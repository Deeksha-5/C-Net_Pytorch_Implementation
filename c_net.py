"""# Import Libraries"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
import time
import math
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
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
from Cnet_model import CNetModel

def get_class_distribution(dataset_obj, idx2class):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    for _, label_id in dataset_obj:
        label = idx2class[label_id]
        count_dict[label] += 1
    return count_dict

def get_class_distribution_loaders(dataloader_obj, dataset_obj, idx2class):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    if dataloader_obj.batch_size == 1:    
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else: 
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
    return count_dict

def plot_from_dict(dict_obj, plot_title, **kwargs):
    barplt = sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)
    fig = barplt.get_figure()
    fig.savefig("out.png")

def sets_loader(datapath, batch_size):
    horizontal_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), # flipping horizontally
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor()
                                          # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # scaling image intensities b/w 0 to 1
                                          ])
    vertical_transforms = transforms.Compose([transforms.RandomVerticalFlip(p=1.0), # flipping vertically
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor()
                                          ])
    shear_transforms = transforms.Compose([transforms.RandomAffine(0, shear=0.2), # shearing with a factor of 0.2
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor()
                                          ])
    height_shift_transforms = transforms.Compose([transforms.RandomAffine(0, translate=(0.2, 0)), # height shifting with a factor of 0.2
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor()
                                          ])
    width_shidt_transforms = transforms.Compose([transforms.RandomAffine(0, translate=(0, 0.2)), # weight shifting with a factor of 0.2
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor()
                                          ])
    rotation_transforms = transforms.Compose([transforms.RandomRotation(degrees=40), # rotation by 40 degrees
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor()
                                          ])
    zooming_transforms = transforms.Compose([transforms.RandomAffine(0, scale=(2, 2)), # zooming by 0.2
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor()
                                          ])
    data_transforms = transforms.Compose([transforms.CenterCrop((224,224)),
                                          transforms.ToTensor()
                                          ])  
    
    t_resized = datasets.ImageFolder(root = datapath+'/train', transform = data_transforms)
    t_horizontal_flipped = datasets.ImageFolder(root = datapath+'/train', transform = horizontal_transforms)
    t_vertical_flipped = datasets.ImageFolder(root = datapath+'/train', transform = vertical_transforms)
    t_sheared = datasets.ImageFolder(root = datapath+'/train', transform = shear_transforms)
    t_height_shifted = datasets.ImageFolder(root = datapath+'/train', transform = height_shift_transforms)
    t_width_shifted = datasets.ImageFolder(root = datapath+'/train', transform = width_shidt_transforms)
    t_rotated = datasets.ImageFolder(root = datapath+'/train', transform = rotation_transforms)
    t_zoomed = datasets.ImageFolder(root = datapath+'/train', transform = zooming_transforms)
    train_data = torch.utils.data.ConcatDataset([t_resized, t_horizontal_flipped, t_vertical_flipped, t_sheared, t_height_shifted, t_width_shifted, t_rotated, t_zoomed])
    # train_data = datasets.ImageFolder(root = datapath+'/train', transform = train_transforms)
    v_resized = datasets.ImageFolder(root = datapath+'/val', transform = data_transforms)
    v_horizontal_flipped = datasets.ImageFolder(root = datapath+'/val', transform = horizontal_transforms)
    v_vertical_flipped = datasets.ImageFolder(root = datapath+'/val', transform = vertical_transforms)
    v_sheared = datasets.ImageFolder(root = datapath+'/val', transform = shear_transforms)
    v_height_shifted = datasets.ImageFolder(root = datapath+'/val', transform = height_shift_transforms)
    v_width_shifted = datasets.ImageFolder(root = datapath+'/val', transform = width_shidt_transforms)
    v_rotated = datasets.ImageFolder(root = datapath+'/val', transform = rotation_transforms)
    v_zoomed = datasets.ImageFolder(root = datapath+'/val', transform = zooming_transforms)
    valid_data = torch.utils.data.ConcatDataset([v_resized, v_horizontal_flipped, v_vertical_flipped, v_sheared, v_height_shifted, v_width_shifted, v_rotated, v_zoomed])
    test_data = datasets.ImageFolder(root = datapath+'/test', transform = data_transforms)

    print("ORIGINAL TRAIN DATA SIZE:", len(t_resized))
    print("AUGMENTED TRAIN DATA SIZE:", len(train_data))
    print("ORIGINAL VALID DATA SIZE:", len(v_resized))
    print("AUGMENTED VALID DATA SIZE:", len(valid_data))
    print("TEST DATA SIZE:", len(test_data))

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
    
    print(t_resized.class_to_idx)
    idx2class = {v: k for k, v in t_resized.class_to_idx.items()}
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
    # plot_from_dict(get_class_distribution_loaders(trainloader, resized, idx2class), plot_title="Train Set", ax=axes[0])
    # plot_from_dict(get_class_distribution_loaders(validloader, valid_data, idx2class), plot_title="Valid Set", ax=axes[1])
    return trainloader,validloader,testloader

"""# CNet Trainer"""

class CNetTrainer(object):
    def __init__(self, net, optimizer, criterion, batch_size, no_epochs):
        self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.no_epochs = no_epochs
        self.batch_size = batch_size

    def train(self, trainset, validset, accuracy_file, loss_file, model_path):
        accuracy_stats = {'train': [],'valid': []}
        loss_stats = {'train': [],'valid': []}
        min_loss_valid = float(100)
        patience = 100
        trigger_times = 0
        for epoch in range(self.no_epochs):
            start_time = time.time()
            train_pred = []
            train_true = []
            val_pred = []
            val_true = []
            train_epoch_loss = 0
            val_epoch_loss = 0
            self.net.train()
            for X_train_batch, y_train_batch in trainset:
                X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                self.optimizer.zero_grad()
                y_train_pred = self.net(X_train_batch.float())
                y_train_batch = torch.unsqueeze(y_train_batch, 1)
                y_train_batch = y_train_batch.float()
                train_loss = self.criterion(y_train_pred, y_train_batch)
                train_loss.backward()
                self.optimizer.step()
                train_epoch_loss += train_loss.item()
                y_train_pred = (np.round(y_train_pred.detach().cpu().numpy())).tolist()
                y_train_batch = (np.round(y_train_batch.detach().cpu().numpy())).tolist()
                train_true.extend(y_train_batch)
                train_pred.extend(y_train_pred)
            end_time = time.time()
            with torch.no_grad():
              self.net.eval()
              for X_val_batch, y_val_batch in validset:
                  X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                  y_val_pred = self.net(X_val_batch.float())
                  y_val_batch = torch.unsqueeze(y_val_batch, 1)
                  y_val_batch = y_val_batch.float()
                  val_loss = self.criterion(y_val_pred, y_val_batch)
                  val_epoch_loss += val_loss.item()
                  y_val_pred = (np.round(y_val_pred.detach().cpu().numpy())).tolist()
                  y_val_batch = (np.round(y_val_batch.detach().cpu().numpy())).tolist()
                  val_true.extend(y_val_batch)
                  val_pred.extend(y_val_pred)
            loss_train = train_epoch_loss/len(trainset)
            loss_valid = val_epoch_loss/len(validset)
            if loss_valid > min_loss_valid:
                trigger_times += 1
                print('Trigger Times: ', trigger_times)
                if trigger_times >= patience:
                    print('Early Stopping!')
                    break
            else:
                trigger_times = 0
                print("Trigger Times: ", trigger_times)
                min_loss_valid = loss_valid
                torch.save(self.net.state_dict(), model_path)
            accuracy_train = float(accuracy_score(train_true, train_pred))
            accuracy_valid = float(accuracy_score(val_true, val_pred))
            print(f'Epoch {epoch+0:02}: | Train Loss: {loss_train:.5f} | Val Loss: {loss_valid:.5f} | Train Acc: {accuracy_train:.5f} | Val Acc: {accuracy_valid:.5f} | Time: {end_time - start_time:.2f}')
            accuracy_stats['train'].append(accuracy_train)
            accuracy_stats['valid'].append(accuracy_valid)
            loss_stats['train'].append(loss_train)
            loss_stats['valid'].append(loss_valid)
            epoch += 1
        with open(accuracy_file, "w") as outfile:
            json.dump(accuracy_stats, outfile)
        with open(loss_file, "w") as outfile:
            json.dump(loss_stats, outfile)

    def test(self, testset, model_path, result_file):
        with torch.no_grad():
            conf_mat = np.asarray([[0, 0], [0, 0]])
            model = self.net
            model.load_state_dict(torch.load(model_path))
            model.eval()
            test_true = []
            test_pred = []
            for X_test_batch, y_test_batch in testset:
                X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(self.device)
                y_test_pred = model(X_test_batch.float())
                y_test_batch = torch.unsqueeze(y_test_batch, 1)
                y_test_batch = y_test_batch.float()
                y_test_pred = (np.round(y_test_pred.detach().cpu().numpy())).tolist()
                y_test_batch = (np.round(y_test_batch.detach().cpu().numpy())).tolist()
                test_true.extend(y_test_batch)
                test_pred.extend(y_test_pred)
            accuracy_test = float(accuracy_score(test_true, test_pred))
            print(accuracy_test)
            conf_mat = confusion_matrix(test_true, test_pred)
            print(conf_mat)
            tn = conf_mat[0][0]
            fn = conf_mat[1][0]
            tp = conf_mat[1][1]
            fp = conf_mat[0][1]
            ACC = (tp + tn) / (tp + tn + fn + fp)
            PPV = tp / (tp + fp)
            NPV = tn / (tn + fn)
            SEN = tp / (tp + fn)
            SPE = tn / (tn + fp)
            F1S = 2 * ((PPV * SEN) / (PPV + SEN))
            MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fn) * (tp + fp) * (fp + tn) * (tn + fn))
            print(f'Accuracy: {ACC:.3f} | PPV: {PPV:.3f} | NPV: {NPV:.3f} | SEN: {SEN:.3f} | SPE: {SPE:.3f} | F1S: {F1S:.3f} | MCC: {MCC:.3f}')
            with open(result_file, 'a') as outfile:
                outfile.write('ACCURACY SCORE: ' + str(np.round(accuracy_test,3))+'\n')
                outfile.write('CONFUSION MATRIX: ' + str(conf_mat)+'\n')
                outfile.write('ACC: ' + str(np.round(ACC,3))+'\n')
                outfile.write('PPV: ' + str(np.round(PPV,3))+'\n')
                outfile.write('NPV: ' + str(np.round(NPV,3))+'\n')
                outfile.write('SEN: ' + str(np.round(SEN,3))+'\n')
                outfile.write('SPE: ' + str(np.round(SPE,3))+'\n')
                outfile.write('F1S: ' + str(np.round(F1S,3))+'\n')
                outfile.write('MCC: ' + str(np.round(MCC,3))+'\n')

"""# CNet Main"""

def train_main(data_folder):
    model = CNetModel().cuda()
    # x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False).cuda()
    # print(model)
    print(summary(model, (3, 224, 224), batch_size=16))
    # x = model(x)
    # dot = make_dot(x, params=dict(list(model.named_parameters())))
    # print(dot)
    # dot.render("cnet_dot.png")
    batch_size = [32, 80]
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-04, betas=(0.9, 0.999), eps=1e-07)
    criterion = nn.BCELoss()
    for bs in batch_size:
        print(bs)
        torch.cuda.empty_cache()
        trainer = CNetTrainer(net=model, optimizer=optim, criterion=criterion, batch_size=bs, no_epochs=5000)
        trainset,validset,testset = sets_loader(data_folder, batch_size=bs)
        trainer.train(trainset=trainset, validset=validset, accuracy_file=data_folder+'/model_accuracy_' + str(bs) + '.json', loss_file=data_folder+'/model_loss_' + str(bs) + '.json', model_path=data_folder+'/model_'+str(bs)+'.pth')
        trainer.test(testset=testset, model_path=data_folder+'/model_'+str(bs)+'.pth', result_file=data_folder+'/model_test_results_' + str(bs) + '.txt')


if __name__ == "__main__":
    data_dir = '/data/BreakHis/400X'
    splitted_data = '/data/BreakHis/400X_Splitted'
    splitfolders.ratio(data_dir, output=splitted_data, ratio=(0.70, 0.15, 0.15), group_prefix=None)
    print("The data lies here =>", splitted_data)
    train_main(data_folder=splitted_data)
