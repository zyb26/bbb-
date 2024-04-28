from copy import deepcopy
from random import randint

import sentence_transformers
from datasets import Dataset


def shuffle(lst):
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while (m):
        m -= 1
        i = randint(0, m)
        temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst

# 首先读取数据
import pandas as pd

df = pd.read_excel(r'C:\Users\86138\Desktop\4组\ifly_llm\Embedding_Model\Embedding_train\1.xlsx')
# df = df.to_csv('output_file.csv', encoding='utf-8', index=False)
#
# print(df[0][0])
import pandas as pd

# 读取 Excel 文件
# df = pd.read_excel(r'./output_file.csv')
# 打印 DataFrame
question = df.iloc[:, 0]
answer = df.iloc[:, 1]


Ko_list = list(question)
Cn_list = list(answer)

shuffle_Cn_list = shuffle(Cn_list)  # 所有的句子打乱排序
shuffle_Ko_list = shuffle(Ko_list)  # 所有的句子打乱排序

# 构造数据集
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from torch.utils.data import DataLoader

train_size = int(len(Ko_list) * 0.8)
eval_size = len(Ko_list) - train_size

# Define your train examples.
train_data = []
for idx in range(train_size):
  train_data.append(InputExample(texts=[Ko_list[idx], Cn_list[idx]], label=1.0))
  train_data.append(InputExample(texts=[shuffle_Ko_list[idx], shuffle_Cn_list[idx]], label=0.0))

# Define your evaluation examples
sentences1 = Ko_list[train_size:]
sentences2 = Cn_list[train_size:]

sentences1.extend(list(shuffle_Ko_list[train_size:]))
sentences2.extend(list(shuffle_Cn_list[train_size:]))

scores = [1.0] * eval_size + [0.0] * eval_size
print(scores)
#pingguqi
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# 加载模型
# Define the model. Either from scratch of by loading a pre-trained model
import sentence_transformers

model = sentence_transformers.SentenceTransformer(model_name_or_path=r'C:\Users\86138\Desktop\4组\ifly_llm\Embedding_Model\model\models--infgrad--stella-mrl-large-zh-v3.5-1792d\snapshots\0cd78d43dfbc6e904b860c938cb79549107cc514', cache_folder='./model')
# Define your train dataset, the dataloader and the train loss

emb1 = model.encode("软件升级服务可以用单一来源采购方式吗？")
emb2 = model.encode("要看具体情形是不是符合《政府采购法》第三十一条单一来源采购的规定。“只能从唯一供应商处采购”是指因货物或者服务使用不可替代的专利、专有技术，或者公共服务项目具有特殊要求，导致只能从某一特定供应商处采购。《政府采购法》第三十一条　符合下列情形之一的货物或者服务，可以依照本法采用单一来源方式采购：（一）只能从唯一供应商处采购的；（二）发生了不可预见的紧急情况不能从其他供应商处采购的；（三）必须保证原有采购项目一致性或者服务配套的要求，需要继续从原供应商处添购，且添购资金总额不超过原合同采购金额百分之十的。",)

from sentence_transformers import SentenceTransformer, util
cos_sim = util.pytorch_cos_sim(emb1, emb2)
print("Cosine-Similarity:", cos_sim)

# 2. Encode
train_dataset = SentencesDataset(train_data, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
train_loss = losses.CosineSimilarityLoss(model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=100, output_path='./Ko2CnModel')

# 测试
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('../Ko2CnModel')

# Sentences are encoded by calling model.encode()
emb1 = model.encode("软件升级服务可以用单一来源采购方式吗？")
emb2 = model.encode("要看具体情形是不是符合《政府采购法》第三十一条单一来源采购的规定。“只能从唯一供应商处采购”是指因货物或者服务使用不可替代的专利、专有技术，或者公共服务项目具有特殊要求，导致只能从某一特定供应商处采购。《政府采购法》第三十一条　符合下列情形之一的货物或者服务，可以依照本法采用单一来源方式采购：（一）只能从唯一供应商处采购的；（二）发生了不可预见的紧急情况不能从其他供应商处采购的；（三）必须保证原有采购项目一致性或者服务配套的要求，需要继续从原供应商处添购，且添购资金总额不超过原合同采购金额百分之十的。",)

cos_sim = util.pytorch_cos_sim(emb1, emb2)
print("Cosine-Similarity:", cos_sim)