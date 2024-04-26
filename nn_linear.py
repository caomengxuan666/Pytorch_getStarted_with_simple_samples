import torch
from torch import nn
import torchvision
from torch.nn import Linear

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)

dataloader = DataLoader(dataset, batch_size=64)

class Cmx(nn.Module):
    def __init__(self):
        super(Cmx, self).__init__()
        self.linear1=Linear(196608,10)


    def forward(self, input):
        output=self.linear1(input)
        return output

cmx=Cmx()

for data in dataloader:
    imgs, targets = data

    print(imgs.shape)


    output=torch.reshape(imgs,(1,1,1,-1))
    #output=torch.flatten(imgs)

    print(output.shape)
