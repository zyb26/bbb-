from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 使用embedding部分来转换为向量
embeddings = HuggingFaceEmbeddings(model_name=r'./multi5')
loader = TextLoader("../data/.txt", encoding="utf-8")
docs = loader.load()
data_loader = [i.page_content for i in docs]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
question = "上海"

# 2. 召回topK
TopK = retriever.invoke(question)
print(TopK)

# 3. 对topK进行rerank
from FlagEmbedding import FlagReranker

# Setting use_fp16 to True speeds up computation with a slight performance degradation
reranker = FlagReranker(model_name_or_path=r'', use_fp16=True)

# 对TopK进行重排
lst1 = []
for i in TopK:
    lst2 = [f'{question}', f'{i.page_content}']
    lst1.append(lst2)
scores = []
for j in lst1:
    score = reranker(j)
    scores.append(j)
scores_list = sorted(scores)

# 判断重排元素在之前的位置
idx = []
for a, b in enumerate(scores_list):
    for w in range(len(scores) - 1):
        if b == scores[w]:
            idx.append(w)

content_rerank = [TopK[i] for i in idx]

# TODO:走向量检索模板  上面还没跑