from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        """
            给每一个具体的图片根据他们的index索引来拼接存放图像的目录地址
        :param root_dir: 当前工作目录的地址
        :param label_dir: 标签的地址
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx: int) -> tuple[Image, str]:
        """
            根据他们的index索引来拼接地址,从而获取这个地址上的图像和它的标签
        :param idx:图像索引
        :return: 返回一个图像和一个标签
        """
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        """
        :return: 返回数据集的长度
        """
        return len(self.img_path)


root_dir = os.getcwd()

ants_label_dir = "ants"
ants_dataset = MyData(root_dir, ants_label_dir)

bees_label_dir="bees"
bees_dataset=MyData(root_dir,bees_label_dir)

train_dataset=ants_dataset+bees_dataset