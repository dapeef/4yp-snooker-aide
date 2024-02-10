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

transform = A.Compose(
    [
        A.augmentations.geometric.resize.Resize(width, height, cv2.INTER_AREA, always_apply=True, p=1), # Squish aspect ratio
        # A.augmentations.crops.transforms.RandomSizedCrop([1080, 1080], height, width, w2h_ratio=1.0, interpolation=1, always_apply=False, p=1.0), # Correct aspect ratio
        A.augmentations.geometric.transforms.ShiftScaleRotate(
            shift_limit=0.3,
            scale_limit=[-0.8, 0],
            rotate_limit=90,
            interpolation=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=None,
            mask_value=None,
            shift_limit_x=None,
            shift_limit_y=None,
            rotate_method='ellipse',
            always_apply=False,
            p=1
        ),
        # A.augmentations.geometric.resize.SmallestMaxSize(max_size=1024, interpolation=1, always_apply=False, p=1)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # ToTensorV2 converts image to pytorch tensor without div by 255
        ToTensorV2(p=1.0) 
    ],
    bbox_params={'format': 'pascal_voc', 'min_visibility': 0.6, 'label_fields': ['labels']}
)


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
