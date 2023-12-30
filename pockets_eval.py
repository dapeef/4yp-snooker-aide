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


# we create a Dataset class which has a __getitem__ function and a __len__ function
class TableImagesDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.height = height
        self.images_dir = files_dir
        self.width = width

        if os.path.isdir(self.images_dir):
            # sorting the images for consistency
            # To get images, the extension of the filename is checked to be jpg
            self.imgs = [image for image in sorted(os.listdir(self.images_dir)) if image[-4:] in ['.jpg', '.png']]
        
        elif os.path.isfile(self.images_dir):
            self.images_dir, img_name = os.path.split(self.images_dir)
            self.imgs = [img_name]

        else:
            raise Exception(f"\"{self.images_dir}\" is not a valid file path")

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.images_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        target = {}
        target["boxes"] = torch.as_tensor([], dtype=torch.float32)
        target["labels"] = torch.as_tensor([], dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            sample = self.transforms(image = img_res,
                                    bboxes = target['boxes'],
                                    labels = target["labels"])
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
        # print(target)

        return img_res, target

    def __len__(self):
        return len(self.imgs)


def get_pockets(image_file):
    transform = A.Compose(
        [ToTensorV2(p=1.0)],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )

    dataset = TableImagesDataset(image_file, 244, 244, transforms=transform)

    model_path = "./checkpoints/model.pth"

    return nn_utils.get_boxes(model_path, dataset, image_file)
