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

    model_path = "./checkpoints/pockets_model5.pth"

    # num_classes = 3

    target = nn_utils.get_boxes(model_path, dataset, image_file)

    
    filtered = nn_utils.filter_boxes(target, confidence_threshold=.1)

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nn_utils.plot_result_img_bbox(img, filtered, "NN pockets")
    # plt.show()

    return filtered
