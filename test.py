from PIL import Image
import torchvision
from model import *
import os
class_name = get_class_name()


class Test:
    def __init__(self, model_path, img_path, device='cuda'):
        # 加载之前训练好的模型
        self.model = torch.load(model_path, map_location=device)  # 将模型加载到 GPU 或 CPU 上
        self.model = self.model.to(device)  # 将模型移动到指定设备上
        self.model.eval()

        self.device = device
        self.class_name = get_class_name()  # 获取类别名称

        if os.path.isdir(img_path):  # 如果传入的是文件夹路径
            self.process_folder(img_path)
        else:
            self.process_image(img_path)  # 如果传入的是单个图片路径

    def process_image(self, img_path):
        # 处理单张图片
        image = Image.open(img_path)
        image = image.convert('RGB')
        self.img = image
        self.trans_img()
        self.classify()

    def process_folder(self, img_folder):
        # 处理文件夹中的所有图片
        image_paths = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder)
                       if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

        for img_path in image_paths:
            print(f"Processing image: {img_path}")
            self.process_image(img_path)

    def trans_img(self):
        # 图像转换
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])
        image = transform(self.img)

        # 将图像移动到 GPU 上
        image = image.to(self.device)

        # 添加批次维度
        self.img = torch.unsqueeze(image, 0)

    def classify(self):
        # 进行预测
        with torch.no_grad():
            output = self.model(self.img)
            predicted_index = output.argmax(1).item()
            predicted_class = self.class_name[predicted_index]
            print("Predicted class:", predicted_class)

