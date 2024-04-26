import torch
import torchvision.models

# 方式1
torch.load('vgg16_method1.pth')

# 方式2

vgg16 = torchvision.models.vgg16(pretrained=False)

vgg16.load_state_dict(torch.load('vgg16_method2.pth'))

# model = torch.load('vgg16_method2.pth')

print(vgg16)
