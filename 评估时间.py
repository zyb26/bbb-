from typing import Optional, List, Any

from langchain_core.retrievers import BaseRetriever
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.legacy import QueryBundle
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.legacy.evaluation.retrieval.metrics_base import BaseRetrievalMetric, RetrievalMetricResult
from llama_index.legacy.schema import NodeWithScore, QueryType

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision
)

from ragas.llama_index import evaluate

# # 命中率 指我们预期的召回文本(真实值) 在召回文本的前k个文本中会出现
# class HitRate(BaseRetrievalMetric):
#     """Hit rate metric."""
#
#     metric_name: str = "hit_rate"
#
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
#         is_hit = any(id in expected_ids for id in retrieved_ids)
#         return RetrievalMetricResult(
#             score=1.0 if is_hit else 0.0,
#         )
#
# # MRR 是衡量系统在一系列查询中返回相关文档或信息的平均排名的逆数的平均值。
# # 例如，如果一个系统对第一个查询的正确答案排在第二位，
# # 对第二个查询的正确答案排在第一位，则 MRR 为 (1/2 + 1/1) / 2。
# class MRR(BaseRetrievalMetric):
#     """MRR metric."""
#
#     metric_name: str = "mrr"
#
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
#                 return RetrievalMetricResult(
#                     score=1.0 / (i + 1),
#                 )
#         return RetrievalMetricResult(
#             score=0.0,
#         )
#

from typing import List
#
# from elasticsearch import Elasticsearch
# from llama_index.schema import TextNode
# from llama_index import QueryBundle
# from llama_index.schema import NodeWithScore
# from llama_index.retrievers import BaseRetriever
# from llama_index.indices.query.schema import QueryType
#
# from preprocess.get_text_id_mapping import text_node_id_mapping
#
#
# # 计算每个词与文档的相关度
# class CustomBM25Retriever(BaseRetriever):
#     """Custom retriever for elasticsearch with bm25"""
#     def __init__(self, top_k) -> None:
#         """Init params."""
#         super().__init__()
#         self.es_client = Elasticsearch([{'host': 'localhost', 'port': 9200}])
#         self.top_k = top_k
#
#     def _retrieve(self, query: QueryType) -> List[NodeWithScore]:
#         if isinstance(query, str):
#             query = QueryBundle(query)
#         else:
#             query = query
#
#         result = []
#         # 查询数据(全文搜索)
#         dsl = {
#             'query': {
#                 'match': {
#                     'content': query.query_str
#                 }
#             },
#             "size": self.top_k
#         }
#         search_result = self.es_client.search(index='docs', body=dsl)
#         if search_result['hits']['hits']:
#             for record in search_result['hits']['hits']:
#                 text = record['_source']['content']
#                 node_with_score = NodeWithScore(node=TextNode(text=text,
#                                                 id_=text_node_id_mapping[text]),
#                                                 score=record['_score'])
#                 result.append(node_with_score)
#
#         return result
#
# # -*- coding: utf-8 -*-
# # @place: Pudong, Shanghai
# # @file: evaluation_exp.py
# # @time: 2023/12/25 20:01
# import asyncio
# import time
#
# import pandas as pd
# from datetime import datetime
# from faiss import IndexFlatIP
# from llama_index.evaluation import RetrieverEvaluator
# from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
#
# from custom_retriever.bm25_retriever import CustomBM25Retriever
# from custom_retriever.vector_store_retriever import VectorSearchRetriever
# from custom_retriever.ensemble_retriever import EnsembleRetriever
# from custom_retriever.ensemble_rerank_retriever import EnsembleRerankRetriever
# from custom_retriever.query_rewrite_ensemble_retriever import QueryRewriteEnsembleRetriever
#
#
# # Display results from evaluate.
# def display_results(name_list, eval_results_list):
#     pd.set_option('display.precision', 4)
#     columns = {"retrievers": [], "hit_rate": [], "mrr": []}
#     for name, eval_results in zip(name_list, eval_results_list):
#         metric_dicts = []
#         for eval_result in eval_results:
#             metric_dict = eval_result.metric_vals_dict
#             metric_dicts.append(metric_dict)
#
#         full_df = pd.DataFrame(metric_dicts)
#
#         hit_rate = full_df["hit_rate"].mean()
#         mrr = full_df["mrr"].mean()
#
#         columns["retrievers"].append(name)
#         columns["hit_rate"].append(hit_rate)
#         columns["mrr"].append(mrr)
#
#     metric_df = pd.DataFrame(columns)
#
#     return metric_df
#
#
# doc_qa_dataset = EmbeddingQAFinetuneDataset.from_json("../data/doc_qa_test.json")
# metrics = ["mrr", "hit_rate"]
# # bm25 retrieve
# evaluation_name_list = []
# evaluation_result_list = []
# cost_time_list = []
# for top_k in [1, 2, 3, 4, 5]:
#     start_time = time.time()
#     bm25_retriever = CustomBM25Retriever(top_k=top_k)
#     bm25_retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=bm25_retriever)
#     bm25_eval_results = asyncio.run(bm25_retriever_evaluator.aevaluate_dataset(doc_qa_dataset))
#     evaluation_name_list.append(f"bm25_top_{top_k}_eval")
#     evaluation_result_list.append(bm25_eval_results)
#     cost_time_list.append((time.time() - start_time) * 1000)
#
# print("done for bm25 evaluation!")
# df = display_results(evaluation_name_list, evaluation_result_list)
# df['cost_time'] = cost_time_list
# print(df.head())
# df.to_csv(f"evaluation_bm25_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", encoding="utf-8", index=False)
#

# 如何评估我们自己的检索器
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(llm=None, embed_model = embed_model)
vector_index = VectorStoreIndex(nodes, service_context=service_context)
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k = 10)


class CustomRetriever(BaseRetriever):
    ""     "Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
            self,
            vector_retriever: VectorIndexRetriever,
    ) -> None:
        """Init params."""
        # 放入我们自己的检索器
        self._vector_retriever = vector_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        if reranker != 'None':
            retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
            else:
            retrieved_nodes = retrieved_nodes[:5]

        return retrieved_nodes


async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    """Asynchronously retrieve nodes given query.
    Implemented by the user.
    """
    return self._retrieve(query_bundle)


async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
    if isinstance(str_or_query_bundle, str):
        str_or_query_bundle = QueryBundle(str_or_query_bundle)
    return await self._aretrieve(str_or_query_bundle)


custom_retriever = CustomRetriever(vector_retriever)

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=custom_retriever
)
eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)