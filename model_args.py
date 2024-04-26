from torch import nn
import torch
from model import cmx

# 设置L2正则化的权重
l2_lambda = 0.02

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
lr = 1e-2
optimizer = torch.optim.SGD(cmx.parameters(), lr=lr,weight_decay=l2_lambda)
