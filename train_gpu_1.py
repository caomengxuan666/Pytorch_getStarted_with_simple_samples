import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from model import *
from model_args import *

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

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 100

# 添加tensorboard
writer = SummaryWriter("./log_train")


# 添加噪声函数
def load_image_with_noise(input_img, noise_std=0.1):
    # 添加噪声
    noise = torch.randn_like(input_img) * noise_std  # 使用标准正态分布添加噪声
    noisy_img = input_img + noise

    # 将图像像素值限制在0到1之间
    noisy_img = torch.clamp(noisy_img, 0, 1)

    return noisy_img


start_time = time.time()

for i in range(epoch):
    print("——————第{}轮训练开始——————".format(i + 1))

    # 训练步骤
    cmx.train()
    if i > 20:
        lr = 1e-3
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新优化器中的学习率

    if i > 50:
        lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新优化器中的学习率

    for data in train_dataloader:
        imgs, targets = data
        noisy_imgs = load_image_with_noise(imgs)
        imgs = noisy_imgs.cuda() if torch.cuda.is_available() else noisy_imgs
        targets = targets.cuda() if torch.cuda.is_available() else targets
        outputs = cmx(imgs)
        loss = loss_fn(outputs, targets)

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

        # 记录训练集上的准确率
        writer.add_scalar("train_accuracy", accuracy, total_train_step)
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    if total_train_step % 100 == 0:
        end_time = time.time()
        print("训练用时:{}".format(end_time - start_time))
        print("训练次数{},Loss:{}".format(total_train_step, loss.item()))

    # 测试步骤开始
    cmx.eval()

    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            noisy_imgs = load_image_with_noise(imgs)
            imgs = noisy_imgs.cuda() if torch.cuda.is_available() else noisy_imgs
            targets = targets.cuda() if torch.cuda.is_available() else targets
            outputs = cmx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum().item()
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
