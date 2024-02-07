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



def get_balls(image_file):
    evaluator = nn_utils.EvaluateNet("./checkpoints/balls_model.pth", 2)
    evaluator.create_dataset(image_file)
    target = evaluator.get_draw_boxes(0, "NN balls")
    # plt.show()

    return target["centres"]
