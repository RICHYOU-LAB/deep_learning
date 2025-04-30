import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)  # 调整为4维张量 (样本数, 通道数, 高度, 宽度)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 配置CNN模型
cnn_model = nn.models.Model_CNN(
    in_channels=1,  # 输入通道数 (MNIST图像为灰度图，通道数为1)
    conv_channels_list=[64, 128, 256],  # 各卷积层输出通道数序列
    kernel_size=3,  # 卷积核尺寸
    stride=1,  # 步长
    padding=1,  # 填充 (保持特征图尺寸不变)
    act_func='ReLU'  # 激活函数
)

# 配置优化器
optimizer = nn.optimizer.SGD(
    init_lr=0.01,  # 初始学习率 (CNN通常需要更小的学习率)
    model=cnn_model
)

# 配置学习率调度器 (调整里程碑到CNN常用范围)
scheduler = nn.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=[500, 1500, 3000],  # 学习率衰减时机
    gamma=0.5  # 衰减系数
)

# 配置损失函数
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model,max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes = axes.reshape(-1)  # 修正此处的赋值操作
_.set_tight_layout(True)
plot(runner, axes)

plt.show()