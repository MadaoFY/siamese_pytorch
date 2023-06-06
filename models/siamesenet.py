import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.cspconvnext import cspconvnext_t, cspconvnext_s

__all__ = [
    'ss_cspconvnext_t',
    'ss_cspconvnext_s',
]


class ssnet(nn.Module):
    '''
    计算图片embedding
    '''

    def __init__(self, backbone, embedding_num=256, train=True, num_classes=None):
        super().__init__()
        self.embedding_num = embedding_num
        self.backbone = backbone
        inputs = torch.zeros((1, 3, 160, 160))
        out = self.backbone(inputs).detach()
        self.out_shape = out.shape[1]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(self.out_shape, self.embedding_num, bias=False),
            nn.BatchNorm1d(self.embedding_num)
        )

        if train:
            self.cls = nn.Linear(embedding_num, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear, ):
                nn.init.normal_(m.weight, 0, 0.01)


    def forward(self, x, train=True):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        before_norm = self.fc(x)
        x = F.normalize(before_norm, p=2, dim=1)
        if train:
            cls = self.cls(before_norm)
            return x, cls
        return x



def ss_cspconvnext_t(embedding_num=256, train=True, num_classes=None, pretrain_weights=None):

    if num_classes is None:
        train = False
        print("If num_classes is None, train will be set False!")
        print("If you want to train model, you must set a correct number for num_classes.")

    backbone = cspconvnext_t(pretrain_weights=pretrain_weights)
    model = ssnet(
        backbone,
        embedding_num,
        train=train,
        num_classes=num_classes
    )

    return model


def ss_cspconvnext_s(embedding_num=256, train=True, num_classes=None, pretrain_weights=None):

    if num_classes is None:
        train = False
        print("If num_classes is None, train will be set False!")
        print("If you want to train model, you must set a correct number for num_classes.")

    backbone = cspconvnext_s(pretrain_weights=pretrain_weights)
    model = ssnet(
        backbone,
        embedding_num,
        train=train,
        num_classes=num_classes
    )

    return model


if __name__ == '__main__':
    ssmodel = ss_cspconvnext_t()
    print(ssmodel )