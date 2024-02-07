import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import torch
# import torchvision
import cv2
# import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# import os
import nn_utils

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')



def get_pockets(image_file):
    evaluator = nn_utils.EvaluateNet("./checkpoints/pockets_model.pth", 2)
    evaluator.create_dataset(image_file)
    target = evaluator.get_boxes(0)

    filtered = nn_utils.filter_boxes(target, confidence_threshold=.1)

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nn_utils.plot_result_img_bbox(img, filtered, "NN pockets")
    # plt.show()

    return filtered
