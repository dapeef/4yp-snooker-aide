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
# from torchvision.transforms import v2
# from torchvision.transforms import ToTensor
# from torchvision import transforms as torchtrans  
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # helper libraries
import engine
import utils
import transforms as T

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


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

# Function to visualize bounding boxes in the image
def plot_result_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min x-max y-max

    plt.figure("Neural net pocket detection")
    plt.title("Neural net pocket detection")
    a = plt.gca()
    a.imshow(img)
    
    for i, box in enumerate(target['boxes']):
        is_pocket = True

        if "labels" in target.keys():
            if target["labels"][i] != 1:
                is_pocket = False
        
        if is_pocket:
            x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle(
                (x, y),
                width, height,
                linewidth = 2,
                edgecolor = 'blue', #['red', 'orange', 'yellow'][target["labels"][i]],
                facecolor = 'none'
            )
            # Draw the bounding box on top of the image
            a.add_patch(rect)

    # plt.show()



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

def train_nn(dataset, dataset_test, checkpoint_file, num_epochs):
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
    print("Training on device:", device)

    num_classes = 3 # one class (class 0) is dedicated to the "background"

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
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


    if num_epochs <= last_epoch + 1:
        print(f"Requested number of epochs ({num_epochs}) already exceeded; {last_epoch + 1} epochs completed")


    for epoch in range(last_epoch + 1, num_epochs):
        # training for one epoch
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss
                }, checkpoint_file)

        # evaluate on the test dataset
        engine.evaluate(model, data_loader_test, device=device)

        print("Woo! Finished an epoch!")





# Send train=True for training transforms and False for val/test transforms
def get_transform(train):
  if train:
    return A.Compose(
        [
            A.HorizontalFlip(0.5),
            # ToTensorV2 converts image to pytorch tensor without div by 255
            ToTensorV2(p=1.0) 
        ],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )
  else:
    return A.Compose(
        [ToTensorV2(p=1.0)],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )

def filter_pockets(target, confidence_threshold=0, max_results=6, remove_overlaps=True):
    def are_boxes_overlapping(box1, box2):
        # box format: [xmin, ymin, xmax, ymax]
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2

        # Check for non-overlapping conditions
        if x2_box1 < x1_box2 or x1_box1 > x2_box2:
            return False  # Boxes do not overlap horizontally

        if y2_box1 < y1_box2 or y1_box1 > y2_box2:
            return False  # Boxes do not overlap vertically

        return True  # Boxes overlap
    
    new_target = {
        "boxes": [],
        "scores": []
    }

    for i in range(len(target['boxes'])):
        box = target['boxes'][i]
        score = target['scores'][i]
        label = target['labels'][i]

        if score >= confidence_threshold and \
           label == 1 and \
           len(new_target["boxes"]) < max_results:
            
            overlapping = False

            if remove_overlaps:
                for box_i in new_target['boxes']:
                    if are_boxes_overlapping(box, box_i):
                        overlapping = True

            if not overlapping:
                new_target['boxes'].append(box)
                new_target['scores'].append(score)
    
    return new_target

def evaluate_item(model, item):
    image = item[0]
    # targets = item[1]

    # plot_result_img_bbox(image.permute(1, 2, 0), targets)

    images = list(img.to("cpu") for img in [image])

    with torch.no_grad():
        pred = model(images)

    # print(pred)

    filtered = filter_pockets(pred[0], confidence_threshold=.1)

    # plot_result_img_bbox(image.permute(1, 2, 0), pred[0])
    # plot_result_img_bbox(image.permute(1, 2, 0), filtered)

    # print(filtered)
    
    return filtered

def scale_boxes(target, image_res, model_res=[244, 244]):
    new_target = target.copy()

    for i in range(len(target["boxes"])):
        new_target["boxes"][i][0] *= image_res[0]/model_res[0]
        new_target["boxes"][i][1] *= image_res[1]/model_res[1]
        new_target["boxes"][i][2] *= image_res[0]/model_res[0]
        new_target["boxes"][i][3] *= image_res[1]/model_res[1]
    
    return new_target

def get_boxes(model_path, dataset, image_file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device:", device)

    num_classes = 3
    model = get_object_detection_model(num_classes)

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    target = evaluate_item(model, dataset[0])

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape

    scaled_target = scale_boxes(target, [width, height])

    plot_result_img_bbox(img, scaled_target)

    return scaled_target