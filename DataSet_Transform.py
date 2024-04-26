import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10("./dataset", True, transform=dataset_transform)

test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform)

'''
#查看测试集第一个数据
print(test_set[0])

#查看测试集的属性
print(test_set.classes)

#打印图像和对应的target
img,target=test_set[0]
print(img)
print(target)
#显示图像
img.show()

'''
writer = SummaryWriter("test_set")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_seti", img, i)

writer.close()

