import torch
import numpy as np
import torch.nn as nn

from models.cspconvnext import cspconvnext_t, cspconvnext_s
from models.cspresnet import cspresnet101

__all__ = [
    'ss_cspconvnext_t',
    'ss_cspconvnext_s',
    'ss_cspresnet101'
]


class ssnet(nn.Module):
    '''
    用于判断两张图片相似度的孪生网络模型
    '''

    def __init__(self, encoder, embedding_num=256, embedding_train=False):
        super().__init__()
        self.embedding_num = embedding_num
        self.embedding_train = embedding_train
        self.encoder = encoder(embedding_num=self.embedding_num, embedding_train=self.embedding_train)
        if not self.embedding_train:
            inputs = torch.zeros((1, 3, 224, 224))
            self.encoder.train()
            out = self.encoder(inputs)
            self.out_shape = out.shape[-1]
            self.num_classes = 1
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Sequential(nn.Linear(self.out_shape, self.out_shape, bias=False),
                                     # nn.BatchNorm1d(self.out_shape),
                                     nn.LeakyReLU(inplace=True))
            self.fc2 = nn.Sequential(nn.Linear(self.out_shape, 1024, bias=False),
                                     nn.BatchNorm1d(1024),  nn.LeakyReLU(inplace=True),
                                     nn.Linear(1024, self.num_classes),
                                     )

            for m in self.modules():
                if isinstance(m, nn.Linear, ):
                    nn.init.normal_(m.weight, 0, 0.01)



    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        if self.embedding_train:
            return x1, x2
        else:
            x = torch.abs(x1 - x2)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x



def ss_cspconvnext_t(embedding_num=256, embedding_train=True, pretrain_weights=None):
    # embedded_train的值只能是 True 或者 False
    assert embedding_train in (True, False)
    model = ssnet(
        cspconvnext_t,
        embedding_num,
        embedding_train
    )
    # embedded_train为False并且传入预训练权重,对图片的相似度模型进行训练
    if embedding_train is False and pretrain_weights is not None:
        param_weights = torch.load(pretrain_weights)
        # 筛选出共有层的权重
        load_param_weights = {k: v for k, v in param_weights.items() if k in model.state_dict().keys()}
        model.load_state_dict(load_param_weights, strict=False)

    return model


def ss_cspconvnext_s(embedding_num=256, embedding_train=True, pretrain_weights=None):
    # embedded_train的值只能是 True 或者 False
    assert embedding_train in (True, False)
    model = ssnet(
        cspconvnext_s,
        embedding_num,
        embedding_train
    )
    # embedded_train为False并且传入预训练权重,对图片的相似度模型进行训练
    if embedding_train is False and pretrain_weights is not None:
        param_weights = torch.load(pretrain_weights)
        # 筛选出共有层的权重
        load_param_weights = {k: v for k, v in param_weights.items() if k in model.state_dict().keys()}
        model.load_state_dict(load_param_weights, strict=False)

    return model


def ss_cspresnet101(embedding_num=256, embedding_train=True, pretrain_weights=None):
    # embedded_train的值只能是 True 或者 False
    assert embedding_train in (True, False)
    model = ssnet(
        cspresnet101,
        embedding_num,
        embedding_train
    )
    # embedded_train为False并且传入预训练权重,对图片的相似度模型进行训练
    if embedding_train is False and pretrain_weights is not None:
        param_weights = torch.load(pretrain_weights)
        # 筛选出共有层的权重
        load_param_weights = {k: v for k, v in param_weights.items() if k in model.state_dict().keys()}
        model.load_state_dict(load_param_weights, strict=False)

    return model


if __name__ == '__main__':
    ssmodel = ss_cspconvnext_t()
    print(ssmodel )