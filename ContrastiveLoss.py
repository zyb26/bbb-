import torch
import numpy as np
import torch.nn.functional as F

# 对比损失函数
# 计算训练时问题和答案的相似度
# 1. 先算出问题答案的欧式距离 Dw
# 2. Y*1/2(Dw)**2 + (1-Y)1/2{max(0, m - Dw)}**2
# 非1即0  1表示正确答案 0表示负样本
# 正样本只和前面相关 距离越小损失越小
# 负样本只和后面有关 距离超过m就不存在损失(距离足够远)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +     # clamp截断操作
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive