import os, openai
import sys
sys.path.append('C:\Program Files\Tesseract-OCR')

os.environ["OPENAI_API_KEY"] = "sess-ehi2sHXZAiJqtT9EtkP5nwfDQ1SmVwYqHtDzBWOj"
openai.api_key = os.environ['OPENAI_API_KEY']
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

path = "./data/"
from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# ====1.加载数据,从pdf中拆分text和table================================
# ====2.生成summary================================  取代了切割 通过uuid来获取
# ====3.加入多向量库================================
# ====4.构建RAG================================
# =====5.执行查询===============================

# TODO: 改为多线程识别多个PDF； 然后把GPT换为我们自己的模型；

# 1.加载数据, 从pdf中拆分text和table
print("====1.加载数据,从pdf中拆分text和table================================")
filename = path + "LLaMA2.pdf"
print(filename)

# from unstructured.partition.pdf import partition_pdf
#
# # infer_table_structure=True automatically selects hi_res strategy
# elements = partition_pdf(filename=filename, infer_table_structure=True)
# tables = [el for el in elements if el.category == "Table"]
#
# print(tables[0].text)
# print('--------------------------------------------------')
# print(tables[0].metadata.text_as_html)


# Get elements
raw_pdf_elements = partition_pdf(
    filename=path + "docker.pdf",
    # Unstructured first finds embedded image blocks
    extract_images_in_pdf=True,
    # 使用YOLOX模型处理表格框和标题
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # # Post processing to aggregate text once we have the title
    # chunking_strategy="by_title",
    # # Chunking params to aggregate text blocks
    # # Attempt to create a new chunk 3800 chars
    # # Attempt to keep chunks > 2000 chars
    # max_characters=4000,
    # new_after_n_chars=3800,
    # combine_text_under_n_chars=2000,
    # image_output_dir_path=path,
)

# Create a dictionary to store counts of each type
category_counts = {}

# PDF中有多少种数据格式
i = 0
for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1
    print(element)

# Unique_categories will have unique elements
unique_categories = set(category_counts.keys())
print("category_counts:", category_counts)

# 对不同格式的数据分别保存
class Element(BaseModel):
    type: str
    text: Any


# 使用类型进行分类
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    # TODO: 更好的解析这个；可不可以把文本单独存到一个txt然后进行分割向量检索
    elif "unstructured.documents.elements.Text" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# 表格
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

# 文本 e就是Element 对象 属性为 type 和 text(str(element))
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

# 2.生成summary
print("====2.生成summary================================")
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)
#
# Summary chain
model = ChatOpenAI(temperature=0)
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# 总结表格
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
print(table_summaries)

# 总结文本
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
print(text_summaries)

# 3.加入多向量库
print("====3.加入多向量库================================")
import uuid

from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
#
# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# 原理: 向量 --> uuid --> 文本/表格(总结的话中有id, 然后id和表格存在映射)
# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]  # 为每个文本添加一个序号
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})  # 把序号添加到这个metadata
    for i, s in enumerate(text_summaries)
]

retriever.vectorstore.add_documents(summary_texts)  # 把文本变为向量 其中包含metadata--> 向量和序号对应
retriever.docstore.mset(list(zip(doc_ids, texts)))  # 序号和文本对应

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
# 表格总结的话和序号对应
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)  # 向量和序号对应上
retriever.docstore.mset(list(zip(table_ids, tables)))  # 序号和表格对应

# 4.构建RAG
print("====4.构建RAG================================")
from langchain_core.runnables import RunnablePassthrough

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(temperature=0)

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 5.执行查询
print("=====5.执行查询===============================")
res = chain.invoke("Driver Version的型号是多少?")
print(res)














