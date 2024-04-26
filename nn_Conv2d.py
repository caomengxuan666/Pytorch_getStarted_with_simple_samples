import torch
from torch import nn
import torchvision.datasets
from torch.nn import Conv2d

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)

dataloader = DataLoader(dataset, batch_size=64)


class Cmx(nn.Module):
    def __init__(self):
        super(Cmx, self).__init__()
        self.conv1 = Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


cmx = Cmx()
print(cmx)

writer=SummaryWriter("./log3")

step=0
for data in dataloader:
    imgs, targets = data
    output = cmx(imgs)
    print(imgs.shape)
    print(output.shape)

    writer.add_images("input",imgs,step)

    #改变大小,-1表示自动计算维度
    output=torch.reshape(output,(-1,3,30,30))

    writer.add_images("output",output,step)
    step+=1
