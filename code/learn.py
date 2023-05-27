import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import pdb
torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size = 1,
    shuffle= True)

how_many_to_plot = 20

plt.figure(figsize = (50, 50))
for i, batch in enumerate(train_loader, start = 1):
    image, label = batch
    plt.subplot(10,10,i)
    plt.imshow(image.squeeze(), cmap = 'gray')
    plt.axis('off')
    plt.title(train_set.classes[label.item()], fontsize = 28)
    if (i >= how_many_to_plot): break
plt.show()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        
        self.fc1 = nn.Linear(in_features = 12 * 4 * 4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features= 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)

    def forward(self, t):
        return t
    
    # def __repr__(self):
    #     return 'Parameter containing:\n' + super(Parameter, self).__repr__()

network = Network()

for name, param in  network.named_parameters():
    print(name, '\t\t', param.shape)

    