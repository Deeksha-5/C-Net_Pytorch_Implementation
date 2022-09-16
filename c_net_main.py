"""# Import Libraries"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from Cnet_model import CNetModel
import copy

class cnet():
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path)
        self.model.to(self.device)
        self.model.eval()

    def test(self, testimg):
        with torch.no_grad():
            torch.cuda.empty_cache()
            data_transform = transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor()])
            testimg = data_transform(testimg)
            testimg = testimg[None, :].to(self.device)
            model_result = self.model(testimg)
            return model_result