from pathlib import Path
import torchio as tio
from test_dataset import Datasets
import os
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
print(type(one_subject.t1.data))


# train_set_samples = (int(len(total_samples)-0.3*len(total_samples)))  #train_test_split
# test_set_samples = (int(len(total_samples))-(train_set_samples))

# trainset, testset = torch.utils.data.random_split(subjects_dataset, [train_set_samples, test_set_samples], generator=torch.Generator().manual_seed(config.dataset.train_test_split_seed))

# trainloader = DataLoader(dataset=trainset,  batch_size=config.training.batch_size, shuffle=True)
# testloader = DataLoader(dataset=testset,   batch_size=config.training.batch_size, shuffle=True)
