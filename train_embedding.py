import os
import time
import torch
import pandas as pd
import albumentations as A
from albumentations import pytorch as AT

from utils.loss import SupConLoss_v1
from utils.utils_train import train_embedding
from utils.general import SineAnnealingLR, same_seeds
from utils.dataset import ReadDataSet_random, ReadDataSet_pairs
from models.siamesenet import ss_cspconvnext_t, ss_cspconvnext_s, ss_cspresnet101


# 数据增强操作
def img_transform(train=True):
    transforms = []
    if train:
        transforms.append(A.RandomBrightnessContrast(p=0.3))
        transforms.append(A.GaussianBlur(p=0.15))
        # transforms.append(A.ToGray(p=0.1))
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.Resize(args.img_sz, args.img_sz, interpolation=2, p=1))
    transforms.append(A.Normalize())
    transforms.append(AT.ToTensorV2())
    return A.Compose(transforms)


def main(args):
    device = torch.device(args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    same_seeds(42)

    # 读取训练集验证集
    train_csv = pd.read_csv(args.train_dir)
    valid_csv = pd.read_csv(args.valid_dir)
    img_dir = args.img_dir

    train_dataset = ReadDataSet_random(train_csv, img_dir, img_transform(), positive_rate=1)
    # val_dataset = ReadDataSet_random(valid_csv, img_dir, img_transform(False), positive_rate=0.5)
    val_dataset = ReadDataSet_pairs(valid_csv, img_dir, img_transform(False))

    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

    # 学习率
    lr = args.lr
    weight_decay = args.weight_decay
    # 训练轮次
    epochs = args.epochs
    # 模型权重保存路径
    model_save_dir = args.model_save_dir
    # 创建模型
    model = ss_cspconvnext_t(embedding_train=True).to(device)

    # 划分是否相同类别的cosine阈值
    cosine_thres = args.cosine_thres
    # 创建损失函数
    loss_fn = SupConLoss_v1(0.07).to(device)
    # 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        weight_decay=weight_decay
        )
    # 优化策略cosine
    t_max = 20
    lr_cosine = SineAnnealingLR(optimizer, t_max)
    # lr_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # 是否使用半精度训练
    fp16 = args.fp16
    # 用于计算训练时间
    start = time.time()

    # 训练
    train_embedding(
        model,
        train_loader,
        val_loader,
        cosine_thres,
        loss_fn,
        optimizer,
        lr_cosine,
        epochs,
        model_save_dir,
        log_save_dir=args.log_save_dir,
        model_save_epochs=args.model_save_epochs,
        gpu=args.gpu,
        fp16=fp16
    )
    print(f'{epochs} epochs completed in {(time.time() - start) / 3600:.3f} hours.')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备类型
    parser.add_argument('--gpu', default='cuda', help='训练设备类型')
    # 训练所需图片的根目录
    parser.add_argument('--img_dir', default='./CASIA_WebFace_clean_v1/img/', help='训练所用图片根目录')
    # 训练集
    parser.add_argument('--train_dir', default='./CASIA_WebFace_clean_v1/WebFace_train_v1.csv', help='训练集文档')
    # 验证集
    parser.add_argument('--valid_dir', default='./CASIA_WebFace_clean_v1/LfwPairs.csv', help='测试集文档')
    # 划分是否相同类别的cosine阈值
    parser.add_argument('--cosine_thres', type=float, default=0.3, help='cosine threshold')
    # 图片的size
    parser.add_argument('--img_sz', type=int, default=160, help='train, val image size (pixels)')
    # 训练信息保存位置
    parser.add_argument('--log_save_dir', default=None, help='tensorboard信息保存地址')
    # 模型权重保存地址
    parser.add_argument('--model_save_dir', default='./models_save/ss_cspconvnext_t',
                        help='模型权重保存地址')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.0001 is the default value for training')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    # 优化器的weight_decay参数
    parser.add_argument('--weight-decay', default=0.05, type=float, metavar='W', help='weight decay')
    # 训练的batch_size
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='batch size when training.')
    # 额外指定权重保存epoch
    parser.add_argument('--model_save_epochs', default=[], type=list, metavar='N', help='额外指定epoch进行权重保存')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--fp16", default=True, choices=[True, False], help="Use fp16 for mixed precision training")

    args = parser.parse_args()
    print(args)

    main(args)