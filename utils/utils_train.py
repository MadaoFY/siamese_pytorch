import os
import math
import torch
import cv2 as cv
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'train_embedding',
]


def train_embedding(
        model,
        train_loader,
        val_loader,
        cosine_thres,
        loss,
        optimizer,
        lr_scheduler,
        epochs,
        model_save_dir,
        log_save_dir,
        model_save_epochs=None,
        device='cuda',
        fp16=True
                    ):
    if log_save_dir is not None:
        writer = SummaryWriter(log_save_dir)

    low_loss = 0.0
    best_acc = 0.0
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    loss = loss.to(device)
    if fp16:
        scaler = GradScaler()

    for epoch in range(epochs):
        epoch += 1
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_cls_accs = []
        # Iterate the training set by batches.
        for img1, img2, ID1, ID2, label in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            img1, img2 = img1.float().to(device), img2.float().to(device)
            imgs = torch.cat([img1, img2], dim=0)
            ids = torch.cat([ID1, ID2], dim=0)

            if fp16:
                with autocast():
                    x, cls = model(imgs)
                    x1, x2 = torch.chunk(x, 2, 0)
                    em_losses = loss(x1, x2, ID1, ID2)
                    ce_loss = F.cross_entropy(cls, ids)
                    losses = em_losses + ce_loss

                # Compute the gradients for parameters.
                # 反向传播
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()
            else:
                x, cls = model(imgs)
                x1, x2 = torch.chunk(x, 2, 0)
                em_losses = loss(x1, x2, ID1, ID2)
                ce_loss = F.cross_entropy(cls, ids)
                losses = em_losses + ce_loss

                # Compute the gradients for parameters.
                losses.backward()
                optimizer.step()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()

            # Record the loss and accuracy.
            train_loss.append(losses.item())

            cls_pred = cls.argmax(dim=-1)
            cls_acc = (cls_pred == ids).float().mean()
            train_cls_accs.append(cls_acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_cls_accs = sum(train_cls_accs) / len(train_cls_accs)

        # Print the information.
        # 打印每轮的loss信息
        print(f"[Train | {epoch:03d}/{epochs:03d} ] lr={optimizer.state_dict()['param_groups'][0]['lr']:.5f}, "
              f"loss = {train_loss:.5f},"
              f", cls_acc = {train_cls_accs:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        # Iterate the validation set by batches.
        for img1, img2, ID1, ID2, label in tqdm(val_loader):
            img1, img2 = img1.float().to(device), img2.float().to(device)
            label = label.to(device)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                x1 = model.forward(img1, False)
                x2 = model.forward(img2, False)
                losses = loss(x1, x2, ID1, ID2)

            # Record the loss and accuracy.
            # pred = torch.sum(F.mse_loss(x1, x2, reduction='none'), 1)
            pred = F.cosine_similarity(x1, x2)
            pred = torch.where(pred < cosine_thres, 0.0, 1.0)
            acc = (pred == label).float().mean()
            valid_accs.append(acc)
            valid_loss.append(losses.item())

            # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_acc = sum(valid_accs) / len(valid_accs)
        valid_loss = sum(valid_loss) / len(valid_loss)


        if log_save_dir is not None:
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': valid_loss},
                               global_step=epoch)
            writer.add_scalars('acc', {
                # 'train_acc': train_acc,
                'val_acc': valid_acc
            }, global_step=epoch)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'],
                              global_step=epoch)

        # Print the information.
        print(f"[Valid | {epoch:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if low_loss == 0.0:
            low_loss = valid_loss

        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_save_dir + f'_{epoch}_{best_acc:.5f}.pth')
            print('{}[ saving model with best acc {:.5f}, loss {:.5f} ]{}'.format(
                '-' * 15, best_acc, valid_loss, '-' * 15)
            )
        if valid_loss < low_loss:
            low_loss = valid_loss
            torch.save(model.state_dict(), model_save_dir + f'_{epoch}_{low_loss:.5f}.pth')
            print('{}[ saving model with acc {:.5f}, lowest loss {:.5f} ]{}'.format(
                '-' * 15, valid_acc, low_loss, '-' * 15)
            )

        if epoch in model_save_epochs:
            torch.save(model.state_dict(), model_save_dir + f'_{epoch}_{valid_acc}.pth')
            print(f'saving model with epoch {epoch}')
    if log_save_dir:
        writer.close()
    print(f'Done!!!best acc = {best_acc:.5f}')
