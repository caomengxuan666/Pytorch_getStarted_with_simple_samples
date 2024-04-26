import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv



train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=False)

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=False)

# 获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集长度为:{}".format(train_data_size))
print("测试集长度为:{}".format(test_data_size))

# 利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=50)
test_dataloader = DataLoader(test_data, batch_size=50)


# 搭建神经网络
class Cmx(nn.Module):
    def __init__(self):
        super(Cmx, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型
cmx = Cmx()
input = torch.ones((64, 3, 32, 32))
output = cmx(input)
print(output.shape)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
lr = 1e-2
optimizer = torch.optim.SGD(cmx.parameters(), lr=lr)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./log_train")

for i in range(epoch):
    print("——————第{}轮训练开始——————".format(i + 1))

    # 训练步骤
    cmx.train()

    for data in train_dataloader:
        imgs, targets = data
        outputs = cmx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{},Loss:{}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    cmx.eval()

    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = cmx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum().item() / len(targets)
            total_accuracy += accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}.".format(total_accuracy / test_data_size))
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1
    # 保存模型
    torch.save(cmx, "cmx_{}.pth".format(i))
    # torch.save(cmx.state_dict(),"cmx_{}.pth".format(i)
    print("模型已保存")

writer.close()
