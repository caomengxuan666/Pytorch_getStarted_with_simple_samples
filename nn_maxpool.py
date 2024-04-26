import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)

dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]
                      ])

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)


class Cmx(nn.Module):
    def __init__(self):
        super(Cmx, self).__init__()
        self.maxpool1 = MaxPool2d(3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


cmx = Cmx()
output = cmx(input)
print(output)

writer = SummaryWriter("logs_maxpool")

step=0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output=cmx(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()