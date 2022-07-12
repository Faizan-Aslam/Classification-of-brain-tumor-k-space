import torchio as tio
from torchio.transforms import (
    CropOrPad,
    OneOf,
    RescaleIntensity,
    RandomAffine,
    RandomElasticDeformation,
    RandomFlip,
    Compose,
)
import torch
from pathlib import Path
from test_dataset import Datasets


cwd = "C:/Users/faiza/Documents/Code_Brain_Tumor_Classification/Classification-of-brain-tumor-using-Spatiotemporal-models-main"
print(cwd)
dataset_dir = Path(cwd, "dataset", "data")
print(dataset_dir)

dataset = Datasets(dataset_path=dataset_dir)
subjects = dataset.return_total_samples()

# class_weights = torch.FloatTensor([3.54,1,1]).cuda()
#for dataset being unbalanced for classes [LGG, HGG, Healthy]

#Transforms

rescale = RescaleIntensity((0.05, 99.5))
randaffine = RandomAffine(scales=(0.9,1.2),degrees=10, isotropic=True, image_interpolation='nearest')
flip = RandomFlip(axes=('LR'), p=0.5)
transforms = [rescale, flip, randaffine]

transform = Compose(transforms)

subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)
print('Dataset size:', len(subjects_dataset), 'subjects')
one_subject = subjects_dataset[1]
one_subject.plot()
print(one_subject.t1.shape)
