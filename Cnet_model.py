import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import math
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import shutil
import PIL
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from torchviz import make_dot, make_dot_from_trace
import decimal
import splitfolders
import json

class OuterPart(nn.Module):
  def __init__(self, inc, outc, k1 = 3, k2 = 2, s1 = 1, s2 = 2):
    super(OuterPart, self).__init__()
    outc1 = 64
    self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc1, kernel_size=k1, padding='same')
    outc2 = outc1 * 2
    self.conv2 = nn.Conv2d(in_channels=outc1, out_channels=outc2, kernel_size=k1, padding='same')
    outc3 = outc2 * 2
    self.conv3 = nn.Conv2d(in_channels=outc2, out_channels=outc3, kernel_size=k1, padding='same')
    outc4 = outc3
    self.conv4 = nn.Conv2d(in_channels=outc3, out_channels=outc4, kernel_size=k1, padding='same')
    self.maxp = nn.MaxPool2d(kernel_size=k2, stride=s2)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxp(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxp(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.maxp(x)
    x = self.conv4(x)
    x = self.relu(x)
    return x

class MiddlePart(nn.Module):
  def __init__(self, inc, outc, k1=1, k2=2, k3=3, s=2):
    super(MiddlePart, self).__init__()
    outc = 256
    self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=k3, padding='same')
    self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=k3, padding='same')
    self.conv3 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=k1, padding='same')
    self.maxp = nn.MaxPool2d(kernel_size=k2, stride=s)
    self.drop = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.maxp(x)
    x = self.drop(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.maxp(x)
    x = self.drop(x)
    return x

class InnerPart(nn.Module):
  def __init__(self, inc, k1=1, k2=2, k3=3, s=2):
    super(InnerPart, self).__init__()
    outc = 256
    self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=k3, padding='same')
    self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=k3, padding='same')
    self.conv3 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=k1, padding='same')
    self.maxp = nn.MaxPool2d(kernel_size=k2, stride=s)
    self.drop = nn.Dropout(p=0.5)
    self.fc1 = nn.Linear(in_features=256*3*3, out_features=1024)
    self.fc2 = nn.Linear(in_features=1024, out_features=1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.flat = nn.Flatten()
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.maxp(x)
    x = self.drop(x)
    x = self.flat(x)
    # x = x.view(x.size(0), 256*3*3)
    x = self.fc1(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    x = self.sigmoid(x)
    return x

"""# CNet Model"""

class CNetModel(nn.Module):
  def __init__(self):
    super(CNetModel, self).__init__()
    self.out = OuterPart(3, 256)
    self.mid = MiddlePart(512, 256)
    self.inn = InnerPart(512)

  def forward(self, x):
    outer1 = self.out(x)
    outer2 = self.out(x)
    midinp1 = torch.cat((outer1, outer2), dim=1)
    outer3 = self.out(x)
    outer4 = self.out(x)
    midinp2 = torch.cat((outer3, outer4), dim=1)
    middle1 = self.mid(midinp1)
    middle2 = self.mid(midinp2)
    inninp = torch.cat((middle1, middle2), dim=1)
    inner = self.inn(inninp)
    return inner