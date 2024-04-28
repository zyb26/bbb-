import typing as t
import os
from typing import List
from datasets import load_dataset, load_from_disk
from ragas.metrics import faithfulness, context_recall, context_precision
from ragas.metrics import AnswerRelevancy
from ragas import evaluate
from ragas.llms import BaseRagasLLM
from langchain.schema import LLMResult
from langchain.schema import Generation
from langchain.callbacks.base import Callbacks
from langchain.schema.embeddings import Embeddings
from FlagEmbedding import FlagModel


class MyLLM(BaseRagasLLM):

    def __init__(self, llm):
        self.base_llm = llm

    @property
    def llm(self):
        return self.base_llm

    def generate(
            self,
            prompts: List[str],
            n: int = 1,
            temperature: float = 0,
            callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        generations = []
        llm_output = {}
        token_total = 0
        for prompt in prompts:
            content = prompt.messages[0].content
            text = api.main(content)   # 修改为自己的API方式调用即可
            generations.append([Generation(text=text)])
            token_total += len(text)
        llm_output['token_total'] = token_total

        return LLMResult(generations=generations, llm_output=llm_output)


class BaaiEmbedding(Embeddings):

    def __init__(self,model_path, max_length=512, batch_size=256):
        self.model = FlagModel(model_path, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
        self.max_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode_corpus(texts, self.batch_size, self.max_length).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode_queries(text, self.batch_size, self.max_length).tolist()

# fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")
fiqa_eval = load_from_disk("./fiqa_eval")
print(fiqa_eval)

my_llm = MyLLM("")
ans_relevancy = AnswerRelevancy(embeddings=BaaiEmbedding())
faithfulness.llm = my_llm
context_recall.llm = my_llm
context_precision.llm = my_llm
ans_relevancy.llm = my_llm

result = evaluate(
    fiqa_eval["baseline"].select(range(3)),
    metrics=[context_recall, context_precision, ans_relevancy, faithfulness]
)

df = result.to_pandas()
print(df.head())
df.to_csv("result.csv", index=False)