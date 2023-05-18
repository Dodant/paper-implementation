import torch
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

path2data = '.'
train_data = datasets.MNIST(path2data, train=True, download=True, transform=data_transform)
val_data = datasets.MNIST(path2data, train=False, download=True, transform=data_transform)

train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
val_dl = DataLoader(val_data, batch_size=32)

# # sample images를 확인합니다.
# # data를 추출합니다.
# x_train, y_train = train_data.data, train_data.targets
# x_val, y_val = val_data.data, val_data.targets
#
# # 차원을 추가하여 B*C*H*W 가 되도록 합니다.
# if len(x_train.shape) == 3: x_train = x_train.unsqueeze(1)
# if len(x_val.shape) == 3: x_val = x_val.unsqueeze(1)
#
# # tensor를 image로 변경하는 함수를 정의합니다.
# def show(img):
#     # tensor를 numpy array로 변경합니다.
#     npimg = img.numpy()
#     # C*H*W를 H*W*C로 변경합니다.
#     npimg_tr = npimg.transpose((1,2,0))
#     plt.imshow(npimg_tr, interpolation='nearest')
#
# # images grid를 생성하고 출력합니다.
# # 총 40개 이미지, 행당 8개 이미지를 출력합니다.
# x_grid = utils.make_grid(x_train[:40], nrow=8, padding=2)
#
# show(x_grid)

from torch import nn
import torch.nn.functional as F

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        # in_features, out_features, bias
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = LeNet_5()
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(next(model.parameters()).device)

from torchsummary import summary
summary(model, input_size=(1,32,32))

loss_fn = nn.CrossEntropyLoss(reduction='sum')

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

opt = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = CosineAnnealingLR(opt, T_max=2, eta_min=1e-05)







