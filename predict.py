import os
import torch
import onnxruntime
import numpy as np
import pandas as pd
import albumentations as A
from tqdm.auto import tqdm
from torch.nn import functional as F
from albumentations import pytorch as AT
from models.siamesenet import ss_cspconvnext_t, ss_cspconvnext_s, ss_cspresnet101

from utils.dataset import ReadDataSet_pairs
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_transform():
    transforms = []
    transforms.append(A.Resize(args.img_sz, args.img_sz, p=1))
    transforms.append(A.Normalize())
    transforms.append(AT.ToTensorV2())
    return A.Compose(transforms)


def main(args):

    img_dir = args.img_dir
    test = pd.read_csv(args.test_dir)

    test_dataset = ReadDataSet_pairs(test, img_dir, test_transform(), test=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
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
        for img1, img2 in tqdm(test_loader):
            img1, img2 = img1.float(), img2.float()
            inputs = {model.get_inputs()[0].name: img1.numpy(), model.get_inputs()[1].name: img2.numpy()}
            pred1, pred2 = model.run(None, inputs)
            pred1, pred2 = torch.as_tensor(pred1[0]), torch.as_tensor(pred2[0])
            pred1, pred2 = F.normalize(pred1), F.normalize(pred2)

            pred = F.cosine_similarity(pred1, pred2)
            pred = torch.where(pred < args.cosine_thres, 0.0, 1.0)
            pred = np.asarray(pred)
            predictions = np.concatenate((predictions, pred))

    elif last_name in ['.pth', '.pt']:
        models_dict = {
            'ss_cspconvnext_t': ss_cspconvnext_t,
            'ss_cspconvnext_s': ss_cspconvnext_s,
            'ss_cspresnet101': ss_cspresnet101
        }
        ss_model = models_dict[args.model]
        model = ss_model(embedding_train=True).to(device)
        param_weights = torch.load(weights)
        model.load_state_dict(param_weights, strict=True)

        model.eval()
        predictions = torch.tensor([], device=device, dtype=torch.int16)
        for img1, img2 in tqdm(test_loader):
            img1, img2 = img1.float().to(device), img2.float().to(device)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                x1, x2 = model(img1, img2)
                x1 = F.normalize(x1)
                x2 = F.normalize(x2)
                pred = F.cosine_similarity(x1, x2)

            pred = torch.where(pred < args.cosine_thres, 0.0, 1.0)
            predictions = torch.cat([predictions, pred])
        predictions = predictions.cpu()

    else:
        pass

    test['prediction'] = predictions.tolist()

    if args.submission_save_dir:
        test.to_csv(args.submission_save_dir, index=False)
        print('Submission is saved')

    print("Done!!!!!!!!!!!!!!!!!!!!")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # ??????
    parser.add_argument("--model", type=str, default='ss_cspconvnext_t',
                        choices=['ss_cspconvnext_t', 'ss_cspconvnext_s', 'ss_cspresnet101'], help="????????????")
    # ??????
    parser.add_argument('--weights', default='./models_save/ss_cspconvnext_t_29_0.88198.pth',
                        help='??????????????????; pth,pt,onnx??????')
    # ??????????????????????????????
    parser.add_argument('--img_dir', default='./CASIA_WebFace_clean_v1/img/', help='???????????????????????????')
    # ?????????
    parser.add_argument('--test_dir', default='./CASIA_WebFace_clean_v1/LfwPairs.csv', help='???????????????')
    # submission????????????
    parser.add_argument('--submission_save_dir', default=None, help='submission????????????')
    # batch_size
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size when training')
    # ???????????????????????????cosine??????
    parser.add_argument('--cosine_thres', type=float, default=0.23, metavar='N', help='threshold of cosine')
    # ?????????size
    parser.add_argument('--img_sz', type=int, default=160, help='train, val image size (pixels)')

    args = parser.parse_args()
    print(args)

    main(args)