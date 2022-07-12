import math
import numpy as np  
import pickle
from image_utils import *
from pathlib import Path
import torchio as tio
from test_dataset import Datasets
import os   
import sys
import torch


cwd = "C:/Users/faiza/Documents/Code_Brain_Tumor_Classification/Classification-of-brain-tumor-using-Spatiotemporal-models-main"
print(cwd)
dataset_dir = Path(cwd, "dataset", "data")
print(dataset_dir)

dataset = Datasets(dataset_path=dataset_dir)
subjects = dataset.brats()
dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')
one_subject = dataset[1]
one_subject.plot()
print(one_subject.t1.data.shape)
np_arr = one_subject.t1.data.cpu().detach()
print(np_arr.shape)
print(np_arr[0][:,:,0].shape)

# train_set_samples = (int(len(total_samples)-0.3*len(total_samples)))  #train_test_split
# test_set_samples = (int(len(total_samples))-(train_set_samples))

# trainset, testset = torch.utils.data.random_split(subjects_dataset, [train_set_samples, test_set_samples], generator=torch.Generator().manual_seed(config.dataset.train_test_split_seed))

# trainloader = DataLoader(dataset=trainset,  batch_size=config.training.batch_size, shuffle=True)
# testloader = DataLoader(dataset=testset,   batch_size=config.training.batch_size, shuffle=True)

# get mri kspace data from 'kspace_brain.p'
# data are organized in row,column order with R=256 rows and C=256 columns
# with open('kspace_brain.p','rb') as f:
#     kspace = pickle.load(f)
# R,C = kspace.shape

# print(f"{R} , {C}")

# # # image = Image.open('sc.png')
# # # # convert image to numpy array
# data = np.asarray(image)

# print(data.shape)

print(np_arr[0][:,:,70].shape)
print(np_arr[0][:,:,70].type)
torch.set_printoptions(threshold=torch.inf)
X = fft2(np_arr)
print(torch.imag(X))

# show_dft(np_arr[0][:,:,70])