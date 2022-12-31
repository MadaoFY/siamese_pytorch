# siamese_pytorch
 个人参加阿里天池CVPR2022 Biometrics Workshop - Pet Biometric Challenge比赛时用到的方案，利用pytorch搭建的孪生网络，这是根据大家赛后的交流与分享，优化后的方案。由于官方要求该数据集只作为本次比赛使用，不可进行分享，所以这里人脸识别数据集作为替代，来跑通整个流程。
 
 ### 环境搭建
 ```bash
git clone https://github.com/MadaoFY/classification_pytorch.git # clone

cd classification_pytorch

pip install -r requirements.txt  # install
```

### 数据集下载
训练集为CASIA-WebFace，验证集为LFW  
本人划分了两个训练集，v1训练集的图片数量与我比赛时用到的数据集数量相当，从CASIA-WebFace中抽取大约2w张图片，v2则为全量的CASIA-WebFace图片。
Caltech_256：https://data.caltech.edu/records/20087

## 使用演示
### 训练(```train_embedding.py```)
假设你已经下载了我提供的数据集，并且生成了train.csv、val.csv文件，打开train.py脚本确认参数后即可运行，部分参数如下。
```python
# 训练设备类型
    parser.add_argument('--gpu', default='cuda', help='训练设备类型')
    # 训练所需图片的根目录
    parser.add_argument('--img_dir', default='./cropface_web_v1/', help='训练所用图片根目录')
    # 训练集
    parser.add_argument('--train_dir', default='./WebFace_train_v1.csv', help='训练集文档')
    # 验证集
    parser.add_argument('--valid_dir', default='./LfwPairs.csv', help='测试集文档')
    # 载入预训练权重
    parser.add_argument('--pretrain_weights', default=None, help='预训练权重')
    # 划分是否相同类别的sigmoid阈值
    parser.add_argument('--sigmoid_thres', type=float, default=0.7, help='cosine threshold')
    # 图片的size
    parser.add_argument('--img_sz', type=int, default=160, help='train, val image size (pixels)')
    # 训练信息保存位置
    parser.add_argument('--log_save_dir', default=None, help='tensorboard信息保存地址')
    # 模型权重保存地址
    parser.add_argument('--model_save_dir', default='./models_save/ss_cspconvnext_t', help='模型权重保存地址')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.001 is the default value for training')
```

### 验证或预测(```val.py、predict.py```)
val.py脚本用于对训练好的模型进行验证(acc1)。  
运行后返回预测准确度，若设置```--submission_save_dir```参数，将导出每对样本的预测结果，更多参数可以在脚本中查看。  
```bash
python val.py --model cspconvnext_t --weights ./models_save/embedding/ss_cspconvnext_t_29_0.88198.pth --img_dir ./cropface_web_v1/ --val_dir ./LfwPairs.csv --cosine_thres 0.3
```

predict.py脚本用于测试集没有标签的情况下，导出预测结果，默认导出文件名为sub.csv，这里你必须设置```--submission_save_dir```参数以生成结果表。
```bash
python predict.py --model cspconvnext_t --weights ./models_save/embedding/ss_cspconvnext_t_29_0.88198.pth --img_dir ./cropface_web_v1/ --val_dir ./LfwPairs.csv --cosine_thres 0.3 --submission_save_dir sub.csv
```


## 其他相关





