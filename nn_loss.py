import torch
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([1, 2, 3], dtype=torch.float32)

targets = torch.tensor([1, 2, 5])

input = torch.reshape(input, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(input, targets)

loss_mse = MSELoss()
result_mse = loss_mse(input, targets)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))

loss_CrossEntropy = CrossEntropyLoss()
result_CrossEntropy = CrossEntropyLoss(x, y)

print(result)
print(result_mse)
print(result_CrossEntropy)
