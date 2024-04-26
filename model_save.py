import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1,保存了模型和参数
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式2，保存网络模型参数为字典，保存参数而不是架构
torch.save(vgg16.state_dict(),'vgg16_method2.pth')
