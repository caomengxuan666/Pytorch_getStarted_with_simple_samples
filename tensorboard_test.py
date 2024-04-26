import cv2
from torch.utils.tensorboard import SummaryWriter
import math
import cv2 as cv
import numpy as np

writer=SummaryWriter("log")

image=cv.imread("C://Users//Lenovo//Desktop//pictures//swxg.jpg")

img=cv.cvtColor(image,cv2.COLOR_BGR2RGB)

print(img.shape)

writer.add_image("test1",img,1,dataformats='HWC')

for i in range(100):
   writer.add_scalar("y=x²",math.pow(i,2),i)

writer.close()
