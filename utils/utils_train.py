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
    'train_classify'
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
        gpu='cuda',
        fp16=True
                    ):
    if log_save_dir:
        writer = SummaryWriter(log_save_dir)

    low_loss = 0.0
    best_acc = 0.0
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device('cpu')
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
        train_accs = []
        # Iterate the training set by batches.
        for img1, img2, ID1, ID2, label in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            img1, img2 = img1.float().to(device), img2.float().to(device)
            label = label.to(device)

            if fp16:
                with autocast():
                    x1, x2 = model(img1, img2)
                    x1 = F.normalize(x1)
                    x2 = F.normalize(x2)
                    pred = F.cosine_similarity(x1, x2)
                    losses = loss(x1, x2, ID1, ID2)

                # Compute the gradients for parameters.
                # 反向传播
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()
            else:
                x1, x2 = model(img1, img2)
                x1 = F.normalize(x1)
                x2 = F.normalize(x2)
                pred = F.cosine_similarity(x1, x2)
                losses = loss(x1, x2, ID1, ID2)

                # Compute the gradients for parameters.
                losses.backward()
                optimizer.step()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()

            # Record the loss and accuracy.
            pred = torch.where(pred < cosine_thres, 0.0, 1.0)
            acc = (pred == label).float().mean()
            train_loss.append(losses.item())
            train_accs.append(acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        # 打印每轮的loss信息
        print(f"[Train | {epoch:03d}/{epochs:03d} ] lr={optimizer.state_dict()['param_groups'][0]['lr']:.5f}, "
              f"loss = {train_loss:.5f}, acc = {train_acc:.5f}")

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
                x1, x2 = model(img1, img2)
                x1 = F.normalize(x1)
                x2 = F.normalize(x2)
                pred = F.cosine_similarity(x1, x2)
                losses = loss(x1, x2, ID1, ID2)

            # Record the loss and accuracy.
            pred = torch.where(pred < cosine_thres, 0.0, 1.0)
            acc = (pred == label).float().mean()
            valid_loss.append(losses.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)


        if log_save_dir:
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': valid_loss},
                               global_step=epoch)
            writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': valid_acc},
                               global_step=epoch)
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
            print('{}[ saving model with best_acc {:.5f} ]{}'.format('-' * 15, best_acc, '-' * 15))
        if epoch in model_save_epochs:
            torch.save(model.state_dict(), model_save_dir + f'_{epoch}_{valid_acc}.pth')
            print(f'saving model with epoch {epoch}')
    writer.close()
    print(f'Done!!!best acc = {best_acc:.5f}')


def train_classify(
        model,
        train_loader,
        val_loader,
        sigmoid_thres,
        loss,
        optimizer,
        lr_scheduler,
        epochs,
        model_save_dir,
        log_save_dir,
        model_save_epochs=None,
        gpu='cuda',
        fp16=True
                    ):
    if log_save_dir:
        writer = SummaryWriter(log_save_dir)

    best_acc = 0.0
    device = torch.device(gpu) if torch.cuda.is_available() else torch.device('cpu')
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
        train_losses = []
        train_accs = []
        # train_gt = []
        # train_pr = []

        # Iterate the training set by batches.
        for img1, img2, ID1, ID2, label in tqdm(train_loader):
            img1, img2 = img1.float().to(device), img2.float().to(device)
            label = label.float().to(device)

            if fp16:
                with autocast():
                    pred = model(img1, img2)
                    pred = torch.flatten(pred)
                    train_loss = loss(pred, label)

                # Compute the gradients for parameters.
                # 反向传播
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()
            else:
                pred = model(img1, img2)
                pred = torch.flatten(pred)
                train_loss = loss(pred, label)

                # Compute the gradients for parameters.
                train_loss.backward()
                optimizer.step()
                # Gradients stored in the parameters in the previous step should be cleared out first.
                # 梯度清零
                optimizer.zero_grad()

            # Compute the accuracy for current batch.
            pred = torch.sigmoid(pred)
            pred = torch.where(pred < sigmoid_thres, 0., 1.0)
            acc = (pred == label).float().mean()

            # Record the loss and accuracy.
            train_losses.append(train_loss.item())
            train_accs.append(acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_losses = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        # 打印每轮的loss信息
        print(f"[Train | {epoch:03d}/{epochs:03d} ] lr={optimizer.state_dict()['param_groups'][0]['lr']:.5f}, "
              f"loss = {train_losses:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_losses = []
        valid_accs = []
        # valid_gt = []
        # valid_pr = []
        # Iterate the validation set by batches.
        for img1, img2, ID1, ID2, label in tqdm(val_loader):
            img1, img2 = img1.float().to(device), img2.float().to(device)
            label = label.float().to(device)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                pred = model(img1, img2)
                pred = torch.flatten(pred)
                valid_loss = loss(pred, label)

            # Compute the accuracy for current batch.
            pred = torch.sigmoid(pred)
            pred = torch.where(pred < sigmoid_thres, 0., 1.0)
            acc = (pred == label).float().mean()

            # Record the loss and accuracy.
            valid_losses.append(valid_loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_losses = sum(valid_losses) / len(valid_losses)
        valid_acc = sum(valid_accs) / len(valid_accs)

        if log_save_dir:
            writer.add_scalars('loss', {'train_loss': train_losses, 'val_loss': valid_losses},
                               global_step=epoch)
            writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': valid_acc},
                               global_step=epoch)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'],
                               global_step=epoch)

        # Print the information.
        print(f"[Valid | {epoch:03d}/{epochs:03d} ] loss = {valid_losses:.5f}, acc = {valid_acc:.5f}")

        # if the model improves, save a checkpoint at this epoch
        # 保持训练权值
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_save_dir + f'_{epoch}_{best_acc:.5f}.pth')
            print('{}[ saving model with best_acc {:.5f} ]{}'.format('-' * 15, best_acc, '-' * 15))
        if epoch in model_save_epochs:
            torch.save(model.state_dict(), model_save_dir + f'_{epoch}_{valid_acc}.pth')
            print('saving model with epoch {}'.format(epoch))
    writer.close()
    print(f'Done!!!best acc = {best_acc:.5f}')