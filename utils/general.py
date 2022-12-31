import os
import math
import torch
import cv2 as cv
import numpy as np


__all__ = [
    'same_seeds',
    'SineAnnealingLR',
    'resize_image',
    'letterbox_image'
]


def same_seeds(seed=42):
    """
        固定随机种子
    """
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    # torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    # torch.backends.cudnn.deterministic = True  # 固定网络结构


def resize_image(image, size=256):
    """
        把短边resize到指定值，长边随比例进行resize
    """
    h, w = image.shape[:2]

    if min(h, w) == size:
        return image

    else:
        if min(h,w) < size:
            inter_fn = cv.INTER_AREA
        else:
            inter_fn = cv.INTER_CUBIC

        if h >= w:
            scale = size / w
        else:
            scale = size / h
        image = cv.resize(image, None, fx=scale, fy=scale, interpolation=inter_fn)

        return image


def letterbox_image(image, return_padding=False):
    """
        为保持h,w的一致,对图片短边两侧进行等距离padding
    """
    h, w = image.shape[:2]

    if h > w:
        p = int((h - w) // 2)
        image = cv.copyMakeBorder(image, 0, 0, p, (h - w - p), cv.BORDER_CONSTANT, value=0)
    else:
        p = int((w - h) // 2)
        image = cv.copyMakeBorder(image, p, (w - h - p), 0, 0, cv.BORDER_CONSTANT, value=0)

    if return_padding:
        return image, p
    else:
        return image


def SineAnnealingLR(opt, t_max):
    """
        sine学习率变化
    """

    lr_lambda = lambda x: (1 + math.cos(math.pi * x / t_max + math.pi)) * 0.5
    lr_sine = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    return lr_sine