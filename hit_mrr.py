# import configparser
# def get_config():
#     """
#     获取ini配置对象
#     :return: ConfigParser
#     """
#     # 创建 ConfigParser 对象
#     config_parser = configparser.ConfigParser()
#     config_parser.read('config.ini', encoding='utf8')
#     return config_parser
# config = get_config()
# import os
# # 代理加载
# os.environ["http_proxy"] = config.get('clash', 'http_proxy')
# os.environ["https_proxy"] = config.get('clash', 'https_proxy')
# 生成向量索引的
import json
import os

from langchain_community.document_loaders.text import TextLoader
from langchain_openai import ChatOpenAI
from llama_index.core import QueryBundle
from llama_index.core.evaluation.retrieval.base import RetrievalEvalMode, RetrievalEvalResult
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
# 检索器
from langchain_core.retrievers import BaseRetriever
from llama_index.legacy.indices.vector_store import VectorIndexRetriever

from llama_index.legacy.node_parser import SimpleNodeParser
# 模型的
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.legacy import evaluation
from llama_index.legacy.embeddings import HuggingFaceEmbedding, CohereEmbedding
# LLM
from llama_index.legacy.llms import Anthropic, OpenAI
from llama_index.legacy.schema import Document
# Evaluator
from llama_index.legacy.evaluation import (
    EmbeddingQAFinetuneDataset, generate_question_context_pairs,
)
from llama_index.legacy.evaluation import RetrieverEvaluator
from typing import List, Any
import pandas as pd
import openai
# import voyageai


# import nest_asyncio
#
# nest_asyncio.apply()
# -*- coding: utf-8 -*-
# 加载文档
# a = Document(text='aaaaa', metadata={'a': 'b'})
# print('aaaaa')
# 产生数据集的时候我使用下面的方法读取不了文件
def get_question_answer():
    """
    1. 进行Document类的转换
    2. 分句子
    3. 产生问答对
    """
    # documents = SimpleDirectoryReader(
    #     input_files=[r'C:\Users\86138\Desktop\4组\ifly_llm\data\LLaMA2.pdf'
    #                  ]).load_data()

    def from_langchain_format(doc):
        """Convert struct from LangChain document format."""
        return Document(text=doc.page_content, metadata=doc.metadata)

    from langchain_community.document_loaders.pdf import PyMuPDFLoader
    data_path = [
        "./data/3-2021_10_26_10_56-上海市公共资源交易管理办法.txt",
        "./data/292-2024_02_20_17_14-【综合采购】上海市公共资源交易中心交易活动实施办法（试行）.txt",
        "./data/301-2020_12_22_21_00-【司法拍卖】上海市公共资源拍卖中心关于进一步做好网络拍卖工作的重点提示.txt"
    ]
    documents_all = []
    for x in data_path:
        loader = TextLoader(x, encoding='utf8')
        a = loader.load()
        documents = []
        for i in a:
            doc = from_langchain_format(i)
            doc.excluded_embed_metadata_keys.extend([
                "file_name",
                "file_type",
                "file_size",
                "creation_date",
                "last_modified_date",
                "last_accessed_date",
            ])
            doc.excluded_llm_metadata_keys.extend([
                "file_name",
                "file_type",
                "file_size",
                "creation_date",
                "last_modified_date",
                "last_accessed_date",
            ])
            documents.append(doc)
        documents_all.extend(documents)
    print(documents_all)

    # 产生一个分词器
    node_parser = SimpleNodeParser.from_defaults(chunk_size=512)

    nodes = node_parser.get_nodes_from_documents(documents_all[1:2])
    print(nodes)
    print(len(nodes))
    #
    # 使用下面模型生成对应的真实标签
    # Prompt to generate questions
    qa_generate_prompt_tmpl = """\
    Context information is below.

    ---------------------
    {context_str}
    ---------------------

    Given the context information and not prior knowledge.
    generate only questions based on the below query.

    You are a Professor. Your task is to setup \
    {num_questions_per_chunk} questions for an upcoming \
    quiz/examination. The questions should be diverse in nature \
    across the document. The questions should not contain options, not start with Q1/ Q2. \
    Restrict the questions to the context information provided.\

    the qusetions and context_str must be chinese char.\
    """

    os.environ[
        "OPENAI_API_KEY"] = "sess-pX3QvpNwuwvA2j0qy4jD5uWxgOReP6xL2ivgOBu6"
    llm = OpenAI(max_retries=5)
    qa_dataset = generate_question_context_pairs(
        nodes,
        llm=llm,
        num_questions_per_chunk=1,
        qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
    print(qa_dataset)
    print(type(qa_dataset))

    qa_dataset.save_json("./data/doc_qa_dataset2.json", )
    return nodes, qa_dataset

# nodes, qa_dataset = get_question_answer()


# def from_langchain_format(doc):
#     """Convert struct from LangChain document format."""
#     return Document(text=doc.page_content, metadata=doc.metadata)

# documents = SimpleDirectoryReader(input_files=["./data/3-2021_10_26_10_56-上海市公共资源交易管理办法.txt",
#                  "./data/292-2024_02_20_17_14-【综合采购】上海市公共资源交易中心交易活动实施办法（试行）.txt",
#                  "./data/301-2020_12_22_21_00-【司法拍卖】上海市公共资源拍卖中心关于进一步做好网络拍卖工作的重点提示.txt"]).load_data()
# node_parser = SimpleNodeParser.from_defaults(chunk_size=512)

# from langchain_community.document_loaders.pdf import PyMuPDFLoader
# data_path = ["./data/3-2021_10_26_10_56-上海市公共资源交易管理办法.txt",
#                 "./data/292-2024_02_20_17_14-【综合采购】上海市公共资源交易中心交易活动实施办法（试行）.txt",
#                 "./data/301-2020_12_22_21_00-【司法拍卖】上海市公共资源拍卖中心关于进一步做好网络拍卖工作的重点提示.txt"]
# documents_all = []
# for x in data_path:
#     loader = TextLoader(x, encoding='utf8')
#     a = loader.load()
#     documents = []
#     for i in a:
#         doc = from_langchain_format(i)
#         doc.excluded_embed_metadata_keys.extend(
#             [
#                 "file_name",
#                 "file_type",
#                 "file_size",
#                 "creation_date",
#                 "last_modified_date",
#                 "last_accessed_date",
#             ]
#         )
#         doc.excluded_llm_metadata_keys.extend(
#             [
#                 "file_name",
#                 "file_type",
#                 "file_size",
#                 "creation_date",
#                 "last_modified_date",
#                 "last_accessed_date",
#             ]
#         )
#         documents.append(doc)
#     documents_all.extend(documents)
# print(documents_all)
documents = SimpleDirectoryReader(input_files=["./data/3-2021_10_26_10_56-上海市公共资源交易管理办法.txt",
                 "./data/292-2024_02_20_17_14-【综合采购】上海市公共资源交易中心交易活动实施办法（试行）.txt",
                 "./data/301-2020_12_22_21_00-【司法拍卖】上海市公共资源拍卖中心关于进一步做好网络拍卖工作的重点提示.txt"]).load_data()
# 产生一个分词器
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)

nodes = node_parser.get_nodes_from_documents(documents[1:2])
print(nodes)

# # 对于得到的数据进行过滤
# def filter_qa_dataset(qa_dataset):
#     """
#     Filters out queries from the qa_dataset that contain certain phrases and the corresponding
#     entries in the relevant_docs, and creates a new EmbeddingQAFinetuneDataset object with
#     the filtered data.

#     :param qa_dataset: An object that has 'queries', 'corpus', and 'relevant_docs' attributes.
#     :return: An EmbeddingQAFinetuneDataset object with the filtered queries, corpus and relevant_docs.
#     """

#     # Extract keys from queries and relevant_docs that need to be removed
#     queries_relevant_docs_keys_to_remove = {
#         k for k, v in qa_dataset.queries.items()
#         if '根据上下文信息，提出一个问题。' in v or '根据文本内容，提出一个问题。' in v
#     }

#     # Filter queries and relevant_docs using dictionary comprehensions
#     filtered_queries = {
#         k: v for k, v in qa_dataset.queries.items()
#         if k not in queries_relevant_docs_keys_to_remove
#     }
#     filtered_relevant_docs = {
#         k: v for k, v in qa_dataset.relevant_docs.items()
#         if k not in queries_relevant_docs_keys_to_remove
#     }

#     # Create a new instance of EmbeddingQAFinetuneDataset with the filtered data
#     return EmbeddingQAFinetuneDataset(
#         queries=filtered_queries,
#         corpus=qa_dataset.corpus,
#         relevant_docs=filtered_relevant_docs
#     )

# qa_dataset = filter_qa_dataset(qa_dataset)

# # filter out pairs with phrases `Here are 2 questions based on provided context`
# # 数据过滤
# # qa_dataset = EmbeddingQAFinetuneDataset.from_json("./data/2.json")
# # print(qa_dataset)
# # print(type(qa_dataset))
# # qa_dataset.save_json()
# # print(qa_dataset.queries)
# # print(qa_dataset.corpus)
# # print(qa_dataset.relevant_docs)
# # print(len(qa_dataset.queries.keys()))
# # print(len(qa_dataset.corpus.keys()))

# # 所有的embedding模型 + reanker模型
# EMBEDDINGS = {
#     # "OpenAI": OpenAIEmbedding(),
#     "bge-large": HuggingFaceEmbedding(model_name=r'/home/admin/workspace/group4/zyb/model_bge'),
#     # "bge-large": HuggingFaceEmbedding(model_name='BAAI/bge-large-en'),
#     # "bge-large": HuggingFaceEmbedding(model_name='BAAI/bge-large-en'),
#     # You can use mean pooling by addin pooling='mean' parameter
#     # "llm-embedder": HuggingFaceEmbedding(model_name='BAAI/llm-embedder'),
#     # "CohereV2": CohereEmbedding(cohere_api_key=cohere_api_key, model_name='embed-english-v2.0'),
#     # "CohereV3": CohereEmbedding(cohere_api_key=cohere_api_key, model_name='embed-english-v3.0', input_type='search_document'),
#     # # "Voyage": VoyageEmbeddings(voyage_api_key=voyage_api_key),
#     # "JinaAI-Small": HuggingFaceEmbedding(model_name='jinaai/jina-embeddings-v2-small-en', pooling='mean', trust_remote_code=True),
#     # "JinaAI-Base": HuggingFaceEmbedding(model_name='jinaai/jina-embeddings-v2-base-en', pooling='mean', trust_remote_code=True),
#     # # "Google-PaLM": GooglePaLMEmbeddings(google_api_key=google_api_key)
# }

# RERANKERS = {
#     "WithoutReranker": "None",
#     # "CohereRerank": CohereRerank(api_key=cohere_api_key, top_n=5),
#     # "bge-reranker-base": SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=5),
#     # "bge-reranker-large": SentenceTransformerRerank(model="BAAI/bge-reranker-large", top_n=5)
# }


# # 展示的方法
# def display_results(embedding_name, reranker_name, eval_results):
#     """Display results from evaluate."""

#     metric_dicts = []
#     for eval_result in eval_results:
#         metric_dict = eval_result.metric_vals_dict
#         metric_dicts.append(metric_dict)

#     full_df = pd.DataFrame(metric_dicts)

#     hit_rate = full_df["hit_rate"].mean()
#     mrr = full_df["mrr"].mean()

#     metric_df = pd.DataFrame(
#         {"Embedding": [embedding_name], "Reranker": [reranker_name], "hit_rate": [hit_rate], "mrr": [mrr]}
#     )

#     return metric_df

# embed_model = HuggingFaceEmbedding(model_name=r'/home/admin/workspace/group4/zyb/model_bge')
# service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)
# vector_index = VectorStoreIndex(nodes, service_context=service_context)
# vector_retriever = VectorIndexRetriever(index=vector_index,
#                                         similarity_top_k=10,
#                                         service_context=service_context)
# # # ------------------------------------------------------------------------------------------------

# from typing import Optional, Any, List
# from llama_index.legacy.evaluation.retrieval.metrics_base import BaseRetrievalMetric, RetrievalMetricResult
# # ----------------------------------------------------------------------------------------------------
# # 指标的求解
# class HitRate(BaseRetrievalMetric):
#     """Hit rate metric."""

#     metric_name: str = "hit_rate"

#     def compute(
#             self,
#             query: Optional[str] = None,
#             expected_ids: Optional[List[str]] = None,
#             retrieved_ids: Optional[List[str]] = None,
#             expected_texts: Optional[List[str]] = None,
#             retrieved_texts: Optional[List[str]] = None,
#             **kwargs: Any,
#     ) -> RetrievalMetricResult:
#         """Compute metric."""
#         if retrieved_ids is None or expected_ids is None:
#             raise ValueError("Retrieved ids and expected ids must be provided")
#         is_hit = any(id in expected_ids for id in retrieved_ids)
#         return RetrievalMetricResult(
#             score=1.0 if is_hit else 0.0,
#         )


# class MRR(BaseRetrievalMetric):
#     """MRR metric."""

#     metric_name: str = "mrr"

#     def compute(
#         self,
#         query: Optional[str] = None,
#         expected_ids: Optional[List[str]] = None,
#         retrieved_ids: Optional[List[str]] = None,
#         expected_texts: Optional[List[str]] = None,
#         retrieved_texts: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> RetrievalMetricResult:
#         """Compute metric."""
#         if retrieved_ids is None or expected_ids is None:
#             raise ValueError("Retrieved ids and expected ids must be provided")
#         for i, id in enumerate(retrieved_ids):
#             if id in expected_ids:
#                 return RetrievalMetricResult(score=1.0 / (i + 1), )
#         return RetrievalMetricResult(score=0.0, )

# # ---------------------------------------------------------------------------------------------------------------------
# # 检索器返回最终召回的节点
# class CustomRetriever:
#     """Custom retriever that performs both Vector search and Knowledge Graph search"""

#     def __init__(
#         self,
#         vector_retriever: VectorIndexRetriever,
#     ) -> None:
#         """Init params."""

#         self._vector_retriever = vector_retriever

#     def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Retrieve nodes given query."""

#         retrieved_nodes = self._vector_retriever.retrieve(query_bundle)

#         # if reranker != 'None':
#         #     retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
#         # else:
#         retrieved_nodes = retrieved_nodes[:5]

#         return retrieved_nodes

#     def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Asynchronously retrieve nodes given query.

#         Implemented by the user.

#         """
#         return self._retrieve(query_bundle)

#     def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
#         if isinstance(str_or_query_bundle, str):
#             str_or_query_bundle = QueryBundle(str_or_query_bundle)
#         return self._aretrieve(str_or_query_bundle)


# custom_retriever = CustomRetriever(vector_retriever)


# def aevaluate_dataset(
#     dataset: EmbeddingQAFinetuneDataset,
#     **kwargs: Any,
# ):

#     def eval_worker(query):
#         return custom_retriever.aretrieve(query)

#     response_jobs = []
#     mode = RetrievalEvalMode.from_str(dataset.mode)

#     metric_dict = {"mrr": [], "hit": []}
#     for query_id, query in dataset.queries.items():
#         expected_ids = dataset.relevant_docs[query_id]
#         expected_texts = dataset.corpus[expected_ids[0]]
#         response_jobs.append(eval_worker(query))  # 每个问题召回的节点
#         result = eval_worker(query)
#         retrieved_ids = [i.node.id_ for i in result]  # 召回文本id
#         retrieved_texts = [i.node.text for i in result]  # 召回的文本
#         print(retrieved_ids)

#         eval_result_mrr = MRR.compute(query, expected_ids, retrieved_ids,
#                                       expected_texts, retrieved_texts)

#         eval_result_hit = HitRate.compute(query, expected_ids, retrieved_ids,
#                                           expected_texts, retrieved_texts)

#         metric_dict["mrr"].append(eval_result_mrr)
#         metric_dict["hit"].append(eval_result_hit)
#     return metric_dict


# result = aevaluate_dataset(qa_dataset)
# mrr = [i.score for i in result["mrr"]]
# hit = [i.score for i in result["hit"]]
# print(mrr)
# print(hit)
