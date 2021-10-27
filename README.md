# Classification-of-brain-tumor-using-Spatiotemporal-models

Classifying tumours using such deep learning methods has made significant progress with the availability of open datasets with reliable annotations. 
Typically those methods
are either 3D models, which use 3D volumetric MRIs or even 2D models considering each slice separately. However, by
treating one spatial dimension separately or by considering the slices as a sequence of images over time, spatiotemporal
models can be employed as "spatiospatial" models for this task. These models have the capabilities of learning specific spatial
and temporal relationship, while reducing computational costs. This paper uses two spatiotemporal models, ResNet (2+1)D
and ResNet Mixed Convolution, to classify different types of brain tumours. It was observed that both these models performed
superior to the pure 3D convolutional model, ResNet18. Furthermore, it was also observed that pre-training the models on a
different, even unrelated dataset before training them for the task of tumour classification improves the performance. Finally,
Pre-trained ResNet Mixed Convolution was observed to be the best model in these experiments, achieving a macro F1-score
of 0.9345 and a test accuracy of 96.98%, while at the same time being the model with the least computational cost.

<p align="center">
<img src="meta/nets.png" alt="Your image title" width="500"/>
</p>


Preprint:

@article{chatterjee2021classification,
  title={Classification of Brain Tumours in MR Images using Deep Spatiospatial Models},
  author={Chatterjee, Soumick and Nizamani, Faraz Ahmed and N{\"u}rnberger, Andreas and Speck, Oliver},
  journal={arXiv preprint arXiv:2105.14071},
  year={2021}
}
