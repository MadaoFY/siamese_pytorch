import cv2 as cv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from general import letterbox_image

__all__ = [
    'ReadDataSet_random',
    'ReadDataSet_pairs',
]


class ReadDataSet_random(Dataset):
    r"""
    使用随机抽取数据的方式进行对比学习。
    要求第一列为图片文件的名称，第二列为图片所属的类别。
    |  img  |   id  |
    |-*.jpg-|-  1  -|
    |-*.jpg-|-  2  -|
    |-*.jpg-|-  2  -|
    |-*.jpg-|-  3  -|

    参数说明：
        data_csv：要求第一列为图片文件的名称，第二列为图片所属的类别
        img_dir：图片所在的文件夹路径
        transform：图片增强方法
        positive_rate：正例图片的选择概率，数值范围0~1。默认值：1.0
    """

    def __init__(self, data_csv, img_dir, transform=None, positive_rate=1.0):
        super().__init__()
        self.data = data_csv
        self.img_dir = img_dir
        self.transform = transform
        assert (0 <= positive_rate <= 1), '0 <= positive_rate <= 1'
        self.pos_rate = positive_rate
        self.id_list = data_csv['id'].unique()
        self.sample_list = np.repeat(self.id_list, 5)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        id_1 = self.sample_list[idx]
        # 正例图片的概率
        p = np.random.rand()
        if p < self.pos_rate:
            id_2 = id_1
            idx_1, idx_2 = np.random.choice(self.data[self.data['id'] == id_1].index, 2)
        else:
            id_2 = np.delete(self.id_list, id_1)
            id_2 = np.random.choice(id_2, 1)[0]
            idx_1 = np.random.choice(self.data[self.data['id'] == id_1].index, 1)[0]
            idx_2 = np.random.choice(self.data[self.data['id'] == id_2].index, 1)[0]

        # 读取第一张图片和所属ID
        img_1 = cv.imread(self.img_dir + self.data.at[idx_1, 'img'])
        img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)

        # 读取第二张图片和所属ID
        img_2 = cv.imread(self.img_dir + self.data.at[idx_2, 'img'])
        img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)

        labels = 1 if id_1 == id_2 else 0

        if self.transform:
            img_1 = self.transform(image=img_1)['image']
            img_2 = self.transform(image=img_2)['image']

        return img_1, img_2, id_1, id_2, labels


class ReadDataSet_pairs(Dataset):
    r"""
    使用已配对的数据进行对比学习。
    训练模式下要求第一列、第二列为图片文件的名称，第三、第四列为图片所属的类别，测试集输出不需要id。
    |  img_1  |  img_2  |   id_1  |   id_2  |
    |- *.jpg -|- *.jpg -|-   1   -|-   1   -|
    |- *.jpg -|- *.jpg -|-   1   -|-   2   -|
    |- *.jpg -|- *.jpg -|-   3   -|-   4   -|
    |- *.jpg -|- *.jpg -|-   3   -|-   3   -|

    参数说明：
        data_csv：第一列为图片1文件名称，第二列为图片2文件名称，第三列的为图片1id，第四列为图片2id
        img_dir：图片所在的文件夹路径
        transform：图片增强方法
    """

    def __init__(self, data_csv, img_dir, transform=None, test=False):
        super().__init__()
        self.data = data_csv
        self.img_dir = img_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 读取第一张图片和所属ID
        img_1 = cv.imread(self.img_dir + self.data.at[idx, 'img_1'])
        img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)

        # 读取第二张图片
        img_2 = cv.imread(self.img_dir + self.data.at[idx, 'img_2'])
        img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)

        if self.transform:
            img_1 = self.transform(image=img_1)['image']
            img_2 = self.transform(image=img_2)['image']

        if not self.test:
            id_1 = self.data.at[idx, 'id_1']
            id_2 = self.data.at[idx, 'id_2']

            labels = 1 if id_1 == id_2 else 0

            return img_1, img_2, id_1, id_2, labels

        else:
            return img_1, img_2
