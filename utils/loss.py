import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

__all__ = [
    'SupConLoss_v1',
    'SupConLoss_v2'
]


class SupConLoss_v1(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x1, x2, labels1=None, labels2=None):
        device = x1.device

        # x1 = F.normalize(x1)
        # x2 = F.normalize(x2)
        x = torch.cat((x1, x2))
        # 计算相似度
        # sim_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)/self.temperature
        sim_matrix = torch.matmul(x, x.T) / self.temperature

        if labels1 is not None:
            # 有监督学习labels
            labels = torch.cat((labels1, labels2)).to(device)
        else:
            # 无监督学习labels
            labels = torch.arange(x1.shape[0]).repeat(2).to(device)

        # 正例掩码，相同label的为1
        p_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        p_mask -= torch.eye(p_mask.shape[0], device=device)
        # 负例掩码
        n_mask = torch.not_equal(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        # 正例相似度
        pos_sim = sim_matrix * p_mask
        # 负例相似度
        neg_sim = torch.exp(sim_matrix) * n_mask

        loss = pos_sim - torch.log(neg_sim.sum(1, keepdim=True))

        loss = -(p_mask * loss)
        loss = loss.sum(1)
        loss = loss / p_mask.sum(1)
        # 平均损失
        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss).mean()
        # mask = 1 - torch.eye(sim_matrix.shape[0], device=device)
        # loss = F.cross_entropy(sim_matrix * mask, labels)

        return loss


class SupConLoss_v2(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x1, x2, labels1=None, labels2=None):
        device = x1.device

        size = x1.shape[-1]
        # x1 = F.normalize(x1)
        # x2 = F.normalize(x2)
        x = torch.cat((x1, x2))
        x = x.reshape(-1, 1)
        # 计算相似度
        # sim_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)/self.temperature
        sim_matrix = torch.matmul(x, x.T) / self.temperature

        if labels1 is not None:
            # 有监督学习labels
            labels = torch.cat((labels1, labels2)).to(device)
        else:
            # 无监督学习labels
            labels = torch.arange(x1.shape[0]).repeat(2).to(device)


        labels = labels.repeat(size, 1).T
        labels = labels.flatten()

        # 正例掩码，相同label的为1
        p_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        p_mask -= torch.eye(p_mask.shape[0], device=device)
        # 负例掩码
        n_mask = torch.not_equal(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        # 正例相似度
        pos_sim = sim_matrix * p_mask
        # 负例相似度
        neg_sim = torch.exp(sim_matrix) * n_mask

        loss = pos_sim - torch.log(neg_sim.sum(1, keepdim=True))
        loss = -((loss * p_mask).sum(1) / p_mask.sum(1))
        # 平均损失
        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss).mean()
        # mask = 1 - torch.eye(sim_matrix.shape[0], device=device)
        # loss = F.cross_entropy(sim_matrix * mask, labels)

        return loss



if __name__ == '__main__':

    torch.manual_seed(24)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(24)
        torch.cuda.manual_seed_all(24)

    loss = SupConLoss_v1(0.07)
    x1 = torch.rand((4, 6))
    print(x1)
    y1 = torch.tensor([0, 0, 1, 1])
    x2 = torch.rand((4, 6))
    y2 = torch.tensor([0, 1, 1, 0])

    out = loss(x1, x2, y1, y2)
    # out = loss(x1, x2)
    print(out)