# basic python and ML Libraries
import os
# import random
import numpy as np
# import pandas as pd

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# torchvision libraries
import torch

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import matplotlib.pyplot as plt

import nn_utils


width = 512
height = width

transform = nn_utils.get_balls_transform_single(width, height)


# defining the files directory and testing directory
files_dir = 'data/Balls - 4792 - above, toy table, stripey balls, very size-sensitive (augment)/train'
test_dir = 'data/Balls - 4792 - above, toy table, stripey balls, very size-sensitive (augment)/test'

# the relevant class in the dataset
chosen_class = 0

# construct dataset
dataset = nn_utils.TrainImagesDatasetSingle(files_dir, chosen_class, width, height, transforms=transform)
dataset_test = nn_utils.TrainImagesDatasetSingle(test_dir, chosen_class, width, height, transforms=transform)
print('Length of training dataset:', len(dataset))
print('Length of test dataset:', len(dataset_test))

# # getting the image and target for a test index.  Feel free to change the index.
# img, target = dataset[25]
# print('Image shape:', img.shape)
# print('Label example:', target)
    
# plotting the image with bboxes. Feel free to change the index

# for i in range(len(dataset)):
#     img, target = dataset[i]
#     print(target)
#     print(len(target["boxes"]), len(target["labels"]), len(target["area"]), len(target['iscrowd']))
#     nn_utils.plot_img_bbox(img.permute(1, 2, 0), target)
#     plt.show()

# Load from last checkpoint
checkpoint_file = "./checkpoints/balls_model_single.pth"

# training for 5 epochs
num_epochs = 100

# num_classes = 2

nn_utils.train_nn(dataset, dataset_test, checkpoint_file, num_epochs)
