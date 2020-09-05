# img2vec
训练图像分类模型, 提取图像embedding

数据集大小: 288w张 256*256大小的图片，类别标签116个

- download_images.py 用于多线程下载图像到本地
- ItemDataset.py 定义数据集
- model.py 定义模型，使用迁移学习，应用resnet152
    * 最后加两层MLP，并只更新这两层参数，第一层用于提取embedding,最后一层输出图像分类结果
    * 通过注册hook，可以提取模型中的参数
- main.py 定义训练过程
    * 支持使用单块GPU
    * 支持小批量数据集跑完一个迭代，检查错误和结果
