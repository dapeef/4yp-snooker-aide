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

import measure_process


def run_epoch_test(epoch, training_loss):
    results = []

    for set_num in range(1, 6):
        # Get names of folders in this set
        directory = os.path.join("./validation/supervised", f"set-{set_num}")
        entries = os.listdir(directory)
        device_names = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

        for device_name in device_names:
            print(f"\n\nTrying set {set_num} with device '{device_name}'")

            test = measure_process.Test(set_num, device_name)

            results.append(test.test_ball_detection("nn", custom_file_name=f"balls_training_epoch_{epoch}"))

        # Aggregate results
        set_result = measure_process.TestResults()
        set_result.aggregate_results(results)
        set_result.training_loss = training_loss
        print("Total results:")
        set_result.print_metrics()
        print("")
        set_result.save_to_file(os.path.join(directory, f"balls_training_epoch_{epoch}_results.json"))

width = 512
height = width

transform = nn_utils.get_balls_transform_multiple(width, height)


# defining the files directory and testing directory
files_dir = './data/Balls, all ball types - 651 - custom, mixed English and American/train'
test_dir = './data/Balls, all ball types - 651 - custom, mixed English and American/test'
num_classes = 20

# the relevant class in the dataset
chosen_class = 0

# construct dataset
dataset = nn_utils.TrainImagesDatasetMultiple(files_dir, width, height, num_classes=num_classes, transforms=transform)
dataset_test = nn_utils.TrainImagesDatasetMultiple(test_dir, width, height, num_classes=num_classes, transforms=transform)
print('Length of training dataset:', len(dataset))
print('Length of test dataset:', len(dataset_test))

# # getting the image and target for a test index.  Feel free to change the index.
# img, target = dataset[25]
# print('Image shape:', img.shape)
# print('Label example:', target)
    
# plotting the images with bboxes.
# for i in range(len(dataset)):
#     img, target = dataset[i]
#     print(target)
#     print(len(target["boxes"]), len(target["labels"]), len(target["area"]), len(target['iscrowd']))
#     nn_utils.plot_img_bbox(img.permute(1, 2, 0), target, "Ooh look, a labelled image!")
#     plt.axis('off')
#     plt.show()

# Load from last checkpoint
checkpoint_file = "./checkpoints/balls_model_multiple.pth"

# training for 5 epochs
num_epochs = 100

nn_utils.train_nn(dataset, dataset_test, checkpoint_file, num_epochs, num_classes=20, test_function=run_epoch_test)
