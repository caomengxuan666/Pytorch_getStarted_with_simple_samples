from torchvision.models import resnet18
from torch import nn
import torch

pretrained_resnet = resnet18(pretrained=True)

def get_class_name():
    class_name = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    return class_name

class Cmx(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Cmx, self).__init__()
        # 加载预训练的ResNet模型，去掉最后一层全连接层
        self.features = nn.Sequential(*list(pretrained_resnet.children())[:-1])
        # 添加你的网络结构
        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),  # 假设输出大小是256，根据你的任务调整
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 10)  # 假设输出大小是10，根据你的任务调整
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 将特征展平
        x = self.fc(x)
        return x


# 创建网络模型
cmx = Cmx()

input = torch.ones((64, 3, 32, 32))
output = cmx(input)
if torch.cuda.is_available():
    cmx = cmx.cuda()


