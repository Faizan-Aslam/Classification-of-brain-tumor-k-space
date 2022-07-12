import os
from os import getcwd
from pathlib import Path
from scipy.fft import fft
import torchio
from image_utils import fft2

class Datasets:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path


    def brats(self):

        HGG = []  # High-grade glioma samples

        for path, currentDirectory, files in os.walk(os.path.join(self.dataset_path, 'MICCAI_BraTS_2019_Data_Training/HGG/')):
            for file in files:
                if file.endswith('_t1ce.nii'):
                    img_path = Path(path + '/' + os.path.basename(path) + '_t1ce.nii')
                    HGG.append(torchio.Subject(t1=torchio.ScalarImage(img_path), label=0))

        LGG = []  # Low-grade glioma samples

        for path, currentDirectory, files in os.walk(os.path.join(self.dataset_path,'MICCAI_BraTS_2019_Data_Training/LGG/')):
            for file in files:
                if file.endswith('_t1ce.nii'):
                    img_path = Path(path + '/' + os.path.basename(path) + '_t1ce.nii')
                    LGG.append(torchio.Subject(t1 = torchio.ScalarImage(img_path), label=0))

        brats = HGG + LGG

        return brats

    def return_real_samples(self):
            return self.brats()

    def return_kspace_samples(self):
        for subject in self.brats():
            subject.t1.data = fft2()
            return self.brats()


    def return_complex_samples(self):
            return self.brats()