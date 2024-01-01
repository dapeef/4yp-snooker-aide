# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import torch
# import torchvision
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
import nn_utils

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')



def get_pockets(image_file):
    width = 244
    height = 244

    transform = A.Compose(
        [
            A.augmentations.geometric.resize.Resize(width, height, cv2.INTER_AREA, always_apply=True, p=1),
            ToTensorV2(p=1.0)
        ],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )

    dataset = nn_utils.EvalImagesDataset(image_file, width, height, transforms=transform)

    model_path = "./checkpoints/pockets_model.pth"

    num_classes = 3

    return nn_utils.get_boxes(model_path, dataset,num_classes, image_file)
