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

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision.transforms import v2
# from torchvision.transforms import ToTensor
# from torchvision import transforms as torchtrans  
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # helper libraries
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



# we create a Dataset class which has a __getitem__ function and a __len__ function
class TableImagesDataset(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.images_dir = files_dir + "/images"
        self.labels_dir = files_dir + "/labels"
        self.height = height
        self.width = width
        
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(self.images_dir)) if image[-4:]=='.jpg']
        
        # classes: 0 index is reserved for background
        self.classes = ['table', 'pocket', 'cushion']

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.images_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        
        # annotation file
        annot_filename = img_name[:-4] + '.txt'
        annot_file_path = os.path.join(self.labels_dir, annot_filename)
        
        boxes = []
        labels = []
        
        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]
        
        # box coordinates for xml files are extracted and corrected for image size given
        with open(annot_file_path) as f:
            for line in f:
                parsed = [float(x) for x in line.split(' ')]
                labels.append(parsed[0])
                x_center = parsed[1]
                y_center = parsed[2]
                box_wt = parsed[3]
                box_ht = parsed[4]

                xmin = x_center - box_wt/2
                xmax = x_center + box_wt/2
                ymin = y_center - box_ht/2
                ymax = y_center + box_ht/2
                
                xmin_corr = int(xmin*self.width)
                xmax_corr = int(xmax*self.width)
                ymin_corr = int(ymin*self.height)
                ymax_corr = int(ymax*self.height)
                
                if xmax_corr - xmin_corr == 0 or ymax_corr - ymin_corr == 0:
                    print("YIKES, the bounding box has non-positive width or height:", img_name)
                
                boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image = img_res,
                                    bboxes = target['boxes'],
                                    labels = labels)
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return img_res, target

    def __len__(self):
        return len(self.imgs)


# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for i, box in enumerate(target['boxes']):
        x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle(
        (x, y),
        width, height,
        linewidth = 2,
        edgecolor = ['red', 'orange', 'yellow'][target["labels"][i]],
        facecolor = 'none'
        )
        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()


# Send train=True for training transforms and False for val/test transforms
def get_transform(train):
  if train:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(0.5),
            # ToTensorV2 converts image to pytorch tensor without div by 255
            # ToTensorV2(p=1.0), # Depricated
            # v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ],
        # bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )
  else:
    return A.Compose(
        [
            # ToTensorV2(p=1.0), # Depricated
            v2.ToDtype(torch.float32, scale=True)
        ],
        # bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )


# Get pretrained model
def get_object_detection_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model


# defining the files directory and testing directory
files_dir = 'data/Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball/train'
test_dir = 'data/Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball/test'

# construct dataset
dataset = TableImagesDataset(files_dir, 244, 244, transforms=get_transform(True))
dataset_test = TableImagesDataset(test_dir, 244, 244, transforms=get_transform(True))
print('Length of training dataset:', len(dataset))
print('Length of test dataset:', len(dataset_test))

# # getting the image and target for a test index.  Feel free to change the index.
# img, target = dataset[25]
# print('Image shape:', img.shape)
# print('Label example:', target)
    
# # plotting the image with bboxes. Feel free to change the index
# img, target = dataset[25]
# plot_img_bbox(img, target)


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
  	# num_workers=4,
    collate_fn=utils.collate_fn,
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=10,
    shuffle=False,
    # num_workers=4,
    collate_fn=utils.collate_fn,
)


# train on gpu if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

num_classes = 3 # one class (class 0) is dedicated to the "background"

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Load from last checkpoint
checkpoint_file = "./checkpoints/model.pth"

if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

else:
    last_epoch = -1

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)


# training for 5 epochs
num_epochs = 35

if num_epochs <= last_epoch + 1:
    print(f"Number of epochs already exceeded. Current number of epochs is {last_epoch + 1}.")

for epoch in range(last_epoch + 1, num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
            }, checkpoint_file)

    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    print("Woo! Finished an epoch!")

