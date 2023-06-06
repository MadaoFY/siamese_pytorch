import os
import torch
import onnxruntime
import numpy as np
import pandas as pd
import albumentations as A
from tqdm.auto import tqdm
from torch.nn import functional as F
from albumentations import pytorch as AT
from models.siamesenet import ss_cspconvnext_t, ss_cspconvnext_s

from utils.dataset import ReadDataSet_pairs
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def val_transform():
    transforms = []
    # transforms.append(A.CenterCrop(96, 96))
    transforms.append(A.Resize(args.img_sz, args.img_sz, interpolation=2, p=1))
    transforms.append(A.Normalize())
    transforms.append(AT.ToTensorV2())
    return A.Compose(transforms)


def main(args):
    img_dir = args.img_dir
    val = pd.read_csv(args.val_dir)

    val_dataset = ReadDataSet_pairs(val, img_dir, val_transform())
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        )

    weights = args.weights
    last_name = os.path.splitext(weights)[-1]
    assert last_name in ['.pth', '.pt', '.onnx'], f"weights file attribute is {last_name}, not in [.pth , .pt, .onnx]."

    if last_name == '.onnx':
        model = onnxruntime.InferenceSession(
            weights,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        predictions = np.array([], dtype=np.int16)
        labels = np.array([], dtype=np.int16)
        for img1, img2, _, _, label in tqdm(val_loader):
            img1, img2 = img1.float(), img2.float()
            inputs = {model.get_inputs()[0].name: img1.numpy(), model.get_inputs()[1].name: img2.numpy()}
            pred1, pred2 = model.run(None, inputs)
            pred1, pred2 = torch.as_tensor(pred1[0]), torch.as_tensor(pred2[0])

            pred = F.cosine_similarity(pred1, pred2)
            pred = torch.where(pred < args.cosine_thres, 0.0, 1.0)
            pred = np.asarray(pred)
            labels = np.concatenate((labels, label))
            predictions = np.concatenate((predictions, pred))

    elif last_name in ['.pth', '.pt']:
        models_dict = {
            'ss_cspconvnext_t': ss_cspconvnext_t,
            'ss_cspconvnext_s': ss_cspconvnext_s,
        }
        ss_model = models_dict[args.model]
        model = ss_model(256, False).to(device)
        param_weights = torch.load(weights)
        model.load_state_dict(param_weights, strict=True)

        model.eval()
        predictions = torch.tensor([], device=device, dtype=torch.int16)
        labels = torch.tensor([], dtype=torch.int16)
        for img1, img2, _, _, label in tqdm(val_loader):
            img1, img2 = img1.float().to(device), img2.float().to(device)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                x1 = model.forward(img1, False)
                x2 = model.forward(img2, False)

                pred = F.cosine_similarity(x1, x2)  # 0.4~0.5
                # pred = torch.sum(F.mse_loss(x1, x2, reduction='none'), 1)    # 1.0

            pred = torch.where(pred < args.cosine_thres, 0., 1.)
            predictions = torch.cat([predictions, pred])
            labels = torch.cat([labels, label])
        predictions = predictions.cpu()

    else:
        pass

    acc = sum(predictions == labels) / len(labels) * 100
    val['prediction'] = predictions.tolist()

    if args.submission_save_dir:
        val.to_csv(args.submission_save_dir, index=False)

    print(f"acc={acc:.2f}%")
    print("Done!!!!!!!!!!!!!!!!!!!!")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 模型
    parser.add_argument("--model", type=str, default='ss_cspconvnext_t',
                        choices=['ss_cspconvnext_t', 'ss_cspconvnext_s'], help="模型选择")
    # 推理所需图片的根目录
    parser.add_argument('--img_dir', default='./CASIA_WebFace_clean_v1/img/', help='训练所用图片根目录')
    # 权重
    parser.add_argument('--weights', default='./models_save/ss_cspconvnext_t_29_0.88198.pth',
                        help='模型文件地址; pth,pt,onnx模型')
    # 验证集
    parser.add_argument('--val_dir', default='./CASIA_WebFace_clean_v1/LfwPairs.csv', help='验证集文档')
    # submission保存位置
    parser.add_argument('--submission_save_dir', default=None, help='submission保存地址')
    # batch_size
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size when training')
    # 划分是否相同类别的cosine阈值
    parser.add_argument('--cosine_thres', type=float, default=0.5, metavar='N', help='threshold of cosine')
    # 图片的size
    parser.add_argument('--img_sz', type=int, default=160, help='train, val image size (pixels)')

    args = parser.parse_args()
    print(args)

    main(args)