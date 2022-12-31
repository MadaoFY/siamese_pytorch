# siamese_pytorch
 个人参加阿里天池CVPR2022 Biometrics Workshop - Pet Biometric Challenge比赛时用到的方案，利用pytorch搭建的孪生网络，这是根据大家赛后的交流与分享，优化后的方案。由于官方要求该数据集只作为本次比赛使用，不可进行分享，所以这里人脸识别数据集作为替代，来跑通整个流程。
 
 ### 环境搭建
 ```bash
git clone https://github.com/MadaoFY/siamese_pytorch.git # clone

cd siamese_pytorch

pip install -r requirements.txt  # install
```

### 数据集下载
训练集从CASIA-WebFace中抽取，验证集为LFW，已进行清洗和整合  
从CASIA-WebFace中抽取大约2w张图片，与比赛时用到的数据集数量相当，所提供的数据集里包含划分好训练集、验证集的csv文件。  
CASIA_WebFace_clean_v1：https://pan.baidu.com/s/1HVv7QNsoKLJgZaA3jI8s6A

LFW人脸识别数据集官方下载：http://vis-www.cs.umass.edu/lfw/index.html#download

## 使用演示
### 训练(```train_embedding.py```)
假设你已经下载了我提供的数据集，文件夹里的WebFace_train_v1.csv为训练集、LfwPairs.csv为验证集，打开train_embedding.py脚本确认参数后即可运行，部分参数如下。该数据集在默认设置下训练40个epoch即可达到最佳准确率，大约88%。
```python
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
```

### 验证或预测(```val.py、predict.py```)
val.py脚本用于对训练好的模型进行验证。  
运行后返回预测准确率，若设置```--submission_save_dir```参数，将导出每对样本的预测结果，更多参数可以在脚本中查看。  
```bash
python val.py --model cspconvnext_t --weights ./models_save/embedding/ss_cspconvnext_t_29_0.88198.pth --img_dir ./cropface_web_v1/ --val_dir ./LfwPairs.csv --cosine_thres 0.3
```

predict.py脚本用于测试集没有标签的情况下，导出预测结果，默认导出文件名为sub.csv，这里你必须设置```--submission_save_dir```参数以生成结果表。
```bash
python predict.py --model cspconvnext_t --weights ./models_save/embedding/ss_cspconvnext_t_29_0.88198.pth --img_dir ./cropface_web_v1/ --val_dir ./LfwPairs.csv --cosine_thres 0.3 --submission_save_dir sub.csv
```


## 其他相关
Simclr：https://arxiv.org/pdf/2002.05709.pdf  
https://www.bilibili.com/video/BV1bD4y1S7nZ/?share_source=copy_web&vd_source=a9ba073ab382750dccbe80bc2c691e91  
https://www.bilibili.com/video/BV19S4y1M7hm/?spm_id_from=333.999.0.0&vd_source=23508829e27bce925740f90e5cd28cf3




