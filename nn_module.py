import torch
from torch import nn


class Cmx(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


cmx = Cmx()
# 创建一个tensor
x = torch.tensor(1.0)
# 把Tensor放入神经网络
output=cmx(x)
print(output)
