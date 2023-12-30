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

import nn_utils


width = 244
height = 244

transform = A.Compose(
    [
        A.augmentations.geometric.resize.Resize(width, height, cv2.INTER_AREA, always_apply=True, p=1),
        A.HorizontalFlip(0.5),
        # ToTensorV2 converts image to pytorch tensor without div by 255
        ToTensorV2(p=1.0) 
    ],
    bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
)


# defining the files directory and testing directory
files_dir = 'data/Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball/train'
test_dir = 'data/Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball/test'

# construct dataset
dataset = nn_utils.TrainImagesDataset(files_dir, width, height, transforms=transform)
dataset_test = nn_utils.TrainImagesDataset(test_dir, width, height, transforms=transform)
print('Length of training dataset:', len(dataset))
print('Length of test dataset:', len(dataset_test))

# # getting the image and target for a test index.  Feel free to change the index.
# img, target = dataset[25]
# print('Image shape:', img.shape)
# print('Label example:', target)
    
# # plotting the image with bboxes. Feel free to change the index
# img, target = dataset[25]
# plot_img_bbox(img, target)



# Load from last checkpoint
checkpoint_file = "./checkpoints/model.pth"

# training for 5 epochs
num_epochs = 40

num_classes = 3

nn_utils.train_nn(dataset, dataset_test, num_classes, checkpoint_file, num_epochs)
