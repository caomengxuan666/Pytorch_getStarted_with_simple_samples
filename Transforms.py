from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2 as cv

img=cv.imread("C://Users//Lenovo//Desktop//pictures//swxg.jpg")

image=cv.cvtColor(img,cv.COLOR_BGR2RGB)

writer=SummaryWriter("log2")


tensor_trans=transforms.ToTensor()

tensor_img=tensor_trans(image)

print(tensor_img)

print(tensor_img[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image("Tensor_img",tensor_img)
writer.add_image("Normalize",img_norm)

writer.close()

