import torch
from torch import nn
import torchvision
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5], [-1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)

dataloader = DataLoader(dataset, batch_size=64)


class Cmx(nn.Module):
    def __init__(self):
        super(Cmx, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.relu1(input)
        return output


cmx = Cmx()
output = cmx(input)
print(output)

writer=SummaryWriter("./logs_ReLU")

step=0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output=cmx(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()