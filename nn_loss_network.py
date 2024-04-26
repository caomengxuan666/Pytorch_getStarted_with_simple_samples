import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)

dataloader = DataLoader(dataset, batch_size=1)


class Cmx(nn.Module):
    def __init__(self):
        super(Cmx, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


cmx = Cmx()
loss = nn.CrossEntropyLoss

for data in dataloader:
    imgs, targets = data
    outputs = cmx(imgs)
    print(outputs)
    print(targets)
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print(result_loss)
