import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 测试集
test_data = torchvision.datasets.CIFAR10("./dataset", False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试集第一张图像的信息，target是标签
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0

for data in test_loader:
    imgs, targets = data


    for i in range(imgs.shape[0]):
        img = imgs[i]
        writer.add_image(f"test_data_{step}", img, step)
        step += 1

writer.close()