import os
import random

import numpy as np
import pandas as pd
import sentence_transformers
import torch
from sentence_transformers.util import cos_sim
from torch import nn
from transformers import AdamW

# 真实数据
data = pd.read_excel(r'./1.xlsx')
question = data.iloc[:, 0]
answer = data.iloc[:, 1]

real_label = [1 for i in range(len(question))]
dump_label = [0 for i in range(len(question))]

dump_answer = []

# 产生负样本
random.seed(333)
for i in range(len(question)):
    j = random.randint(0, len(question) - 1)
    while i == j:
        j = random.randint(0, len(question) - 1)
    dump_answer.append(answer[j])

# 总的数据
total_question = question + question
total_answer = answer + dump_answer
label = real_label + dump_label


# step3. 构建数据集
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.question = total_question
        self.answer = total_answer
        self.label = label

    def __getitem__(self, index):
        return total_question[index], total_answer[index], label[index]

    def __len__(self):
        return len(self.question)

dataset = MyDataset()
for i in range(5):
    print(dataset[i])

from torch.utils.data import random_split

trainset, validset = random_split(dataset, lengths=[0.9, 0.1])

trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
validloader = DataLoader(validset, batch_size=8, shuffle=False)

# 2个元组1个tensor
print(f'数据集批次的形式:{next(enumerate(trainloader))[1]}')


class Trainer:
    def __init__(self, path=None):
        self.path = path if path else r'../model/models--infgrad--stella-mrl-large-zh-v3.5-1792d/snapshots/0cd78d43dfbc6e904b860c938cb79549107cc514'
        self.model = sentence_transformers.SentenceTransformer(
             model_name_or_path=self.path)
        self.optimizer = AdamW(
            self.model.parameters(), lr=0.005, weight_decay=0.001,
            correct_bias=False, no_deprecation_warning=True
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_data_batch = trainloader
        self.test_data_batch = validloader
        self.total_epoch = 1
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        # 防止训练中断
        self.model_dir = "model"
        self.epoch = 0
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            names = os.listdir(self.model_dir)
            if len(names) > 0:
                names.sort()
                name = names[-1]
                checkpoint = torch.load(os.path.join(self.model_dir, name))
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.epoch = checkpoint["epoch"]

    def train(self):
        for epoch in range(self.epoch, self.total_epoch):
            self.model.train(True)
            train_loss = []
            batch = 0
            for Question, Answer, labels in self.train_data_batch:
                q_embeddings = self.model.encode(Question) # N,E
                a_embeddings = self.model.encode(Answer)
                y_pre = []
                for i in range(len(labels)):
                    cosine_scores = cos_sim(q_embeddings[i], a_embeddings[i])#E
                    # print(cosine_scores)
                    y_pre.append(cosine_scores[0][0])#N
                self.optimizer.zero_grad()
                y_pre = torch.tensor(y_pre, dtype=torch.float32)#N,
                labels = labels.to(torch.float)
                # labels.required_grad = True
                y_pre.requires_grad = True
                loss = self.loss_fn(y_pre, labels)
                loss.backward()
                self.optimizer.step()
                batch += 1
                train_loss.append(loss.item())
                if batch % 2 == 0:
                    print(f"第{epoch+1}/{self.total_epoch}轮；第{batch}批次的损失为{loss}")
                    model_path = os.path.join(self.model_dir, f"{epoch}_{batch}.pth")
                    obj = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(obj, model_path)
            # 验证
            batch = 0
            best_acc = 0.0
            test_loss = []
            test_acc_lis = []
            for Question, Answer, labels in self.test_data_batch:
                q_embeddings = self.model.encode(Question)
                a_embeddings = self.model.encode(Answer)
                y_pre = []
                for i in range(len(labels)):
                    cosine_scores = cos_sim(q_embeddings[i], a_embeddings[i])
                    y_pre.append(cosine_scores[0][0])
                y_pre = torch.tensor(y_pre, dtype=torch.float32)
                labels = labels.to(torch.float)
                loss = self.loss_fn(y_pre, labels)
                batch += 1
                test_loss.append(loss.item())
                if batch % 2 == 0:
                    print(f"{epoch + 1}/{self.total_epoch} {batch} train_loss={loss.item()}")
            print(f"{epoch} train_mean_loss {np.mean(train_loss):.4f}, test_mean_loss {np.mean(test_loss)}")
        # 保存最后一轮的
        model_path = os.path.join(self.model_dir, "last.pth")
        obj = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(obj, model_path)

if __name__ == '__main__':
    Trainer().train()



