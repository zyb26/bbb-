import os
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_correctness, context_precision, context_recall

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'  # langchain可视化工具
LANGCHAIN_PROJECT = "defalt"
# http://smith.langchain.com 申请好 proj，生成好 key
os.environ['LANGCHAIN_API_KEY'] = ""
# OPENAI账号
os.environ['OPENAI_API_KEY'] = ""

# 构建正确的问答对
eval_questions = [
    "申请隔夜评标的，发起方、中介机构应当提前几天",
]
# "交易主体、中介机构、评标（评审）专家、市交易中心工作人员应当承担什么义务",
# "发起方和中标人（成交单位）应当在规定的时间内按照交易公告、交易文件和中标人（成交单位）的投标（响应）文件订立书面交易合同，并将交易合同电子件通过什么上传",
# "发起方应当对上传或填报信息材料的哪些方面作出承诺并承担责任",
# "第二十条【投标澄清与说明】主要内容是什么",

eval_answers = [
    "至少五个工作日",
]
# "国家秘密、商业秘密、交易秘密以及个人信息等承担保密",
# "市交易平台",
# "真实性、完整性、准确性",
# "响应方应当根据评标委员会（评审小组）的要求，在不超出投标（响应）文件的范围或改变投标（响应）文件实质性内容的基础上，对投标（响应）文件中含义不明确的内容进行必要的澄清或说明。",
# 检索召回链
llm = ChatOpenAI(temperature=0)
embeddings = HuggingFaceEmbeddings(model_name=r'./model/models--infgrad--stella-mrl-large-zh-v3.5-1792d/snapshots/0cd78d43dfbc6e904b860c938cb79549107cc514')
loader = TextLoader("./data/292-2024_02_20_17_14-【综合采购】上海市公共资源交易中心交易活动实施办法（试行）.txt", encoding="utf-8")
docs = loader.load()
data_loader = [i.page_content for i in docs]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()
print(retriever.invoke("申请隔夜评标的，发起方、中介机构应当提前几天"))
retrievalQA = RetrievalQA.from_llm(
    llm=llm,
    retriever=retriever,
    verbose=bool(os.getenv('VERBOSE'))
)
# print(retrievalQA.invoke("申请隔夜评标的，发起方、中介机构应当提前几天"))

answers = []
contexts = []
for question in eval_questions:
    response = retrievalQA.invoke(question)
    answers.append(response["result"])
    content = retriever.invoke(question)
    contexts.append([context.page_content for context in content])
print(answers)
print(contexts)

# 构建评估数据
response_dataset = Dataset.from_dict({
    "question": eval_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": eval_answers
})

# 更换评估中的embedding模型
context_precision.embeddings = embeddings

score = evaluate(response_dataset, metrics=[context_precision, faithfulness],  llm=llm , embeddings=embeddings)
df = score.to_pandas()
print(df.head())