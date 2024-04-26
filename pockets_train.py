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
import measure_process


def run_epoch_test(epoch, training_loss):
    results = []

    for set_num in range(1, 3):
        # Get names of folders in this set
        directory = os.path.join("./validation/supervised", f"set-{set_num}")
        entries = os.listdir(directory)
        device_names = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

        for device_name in device_names:
            print(f"\n\nTrying set {set_num} with device '{device_name}'")
            corner_labels_folder = os.path.join("./validation/supervised", f"set-{set_num}", device_name, "corner_labels")

            if os.path.exists(corner_labels_folder):
                test = measure_process.Test(set_num, device_name)
                results.append(test.test_pocket_detection("nn_corner", custom_file_name=f"pockets_training_epoch_{epoch}"))
            
            else:
                print("No corner labels for this set")

        # Aggregate results
        set_result = measure_process.TestResults()
        set_result.aggregate_results(results)
        set_result.training_loss = training_loss
        print("Total results:")
        set_result.print_metrics()
        print("")
        set_result.save_to_file(os.path.join(directory, f"pockets_training_epoch_{epoch}_results.json"))


width = 512
height = width

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

# the relevant class in the dataset
chosen_class = 1

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
# img, target = dataset[25]
# nn_utils.plot_img_bbox(img.permute(1, 2, 0), target)



# Load from last checkpoint
checkpoint_file = "./checkpoints/pockets_model.pth"

# training for 5 epochs
num_epochs = 100

# num_classes = 3

nn_utils.train_nn(dataset, dataset_test, checkpoint_file, num_epochs, test_function=run_epoch_test)
