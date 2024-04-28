from models2lang import BaiChuan
mode_name_or_path="../../models/baichuan-inc/Baichuan2-13B-Chat-4bits"
types_="13Bbit"
model = BaiChuan(mode_name_or_path,types_)


import nest_asyncio

nest_asyncio.apply()

from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.evaluation import generate_question_context_pairs
from llama_index.evaluation import RetrieverEvaluator
from llama_index.llms import OpenAI
from llama_index.embeddings import resolve_embed_model

import os
import pandas as pd

# load data from data directory
documents = SimpleDirectoryReader("data").load_data()

# bge-m3 embedding model
# https://huggingface.co/BAAI/bge-base-en-v1.5/tree/main
embed_model = resolve_embed_model("local:BAAI/bge-base-en-v1.5")

# Load LM Studio LLM model
llm = BaiChuan(mode_name_or_path,types_)

# Index the data 
service_context = ServiceContext.from_defaults(
    embed_model=embed_model, llm=llm,
)

# Transform data to Nodes struct  为所有chunk形成树
node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=128)
nodes = node_parser.get_nodes_from_documents(documents)

# vetorize 向量库
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# 使用 LLamaIndex 构建 query_engine，方便向量查询
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query("What did the author do growing up?")
print(response_vector.response)

# 和构建faiss向量库差不多
# ----------------------------------------------------------------------------------------------

# 评价工作应当成为衡量 RAG 应用表现的重要指标。这关乎系统针对不同数据源和多样的查询是否能够给出准确答案。
# 起初，单独审查每一个查询和相应的响应有助于系统调优，但随着特殊情况和故障数量的增加，这种方式可能行不通。相比之下，建立一整套综合性评价指标或者自动化评估系统则更为高效。这类工具能够洞察系统整体性能并识别哪些领域需要进一步关注。

# 检索评估： 这是对系统检索出的信息的准确性与相关性进行评价的过程。
# 响应评估： 这是基于检索结果对系统生成回答的质量和恰当性进行测量的过程

# 生成 “问题-上下文” 对：

# 每个chunk生成3个问题
qa_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=3)

# 检索评估
# 1. 向量库 -> 检索器
retriever = vector_index.as_retriever(similarity_top_k=2)

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

# Evaluate
eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)

# Get the list of queries from the above created dataset
queries = list(qa_dataset.queries.values())

# 构建向量库
vector_index = VectorStoreIndex(nodes, service_context = service_context)
query_engine = vector_index.as_query_engine()

# 评估器1
from llama_index.evaluation import FaithfulnessEvaluator
faithfulness_raven13b = FaithfulnessEvaluator(service_context=service_context)

eval_query = queries[3]
eval_query

response_vector = query_engine.query(eval_query)

# Compute faithfulness evaluation
eval_result = faithfulness_raven13b.evaluate_response(response=response_vector)

# You can check passing parameter in eval_result if it passed the evaluation.
eval_result.passing

# 评估器2
from llama_index.evaluation import RelevancyEvaluator

relevancy_raven13b = RelevancyEvaluator(service_context=service_context)

query = queries[3]
query

# Generate response.
# response_vector has response and source nodes (retrieved context)
response_vector = query_engine.query(query)

# Relevancy evaluation
eval_result = relevancy_raven13b.evaluate_response(
    query=query, response=response_vector
)

# You can check passing parameter in eval_result if it passed the evaluation.
eval_result.passing

# You can get the feedback for the evaluation.
eval_result.feedback

# 批次评估器
from llama_index.evaluation import BatchEvalRunner

# Let's pick top 10 queries to do evaluation
batch_eval_queries = queries[:10]

# Initiate BatchEvalRunner to compute FaithFulness and Relevancy Evaluation.
runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
    workers=8,
)

# Compute evaluation
eval_results = await runner.aevaluate_queries(
    query_engine, queries=batch_eval_queries
)

# Let's get faithfulness score
faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])
faithfulness_score

# Let's get relevancy score
relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])
relevancy_score




