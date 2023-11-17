import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
import utils

# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for i, box in enumerate(target['boxes']):
        high_score = True

        if "scores" in target.keys():
            if target['scores'][i] < 0.7:
                high_score = False

        if high_score and target["labels"][i]==1:
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

# we create a Dataset class which has a __getitem__ function and a __len__ function
class TableImagesDataset(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.images_dir = files_dir + "\\images"
        self.labels_dir = files_dir + "\\labels"
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
                    print("YIKES")
                
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

num_classes = 3

model = get_object_detection_model(num_classes)

checkpoint = torch.load(".\\checkpoints\\model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dataset = TableImagesDataset('data\\Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball\\test_small', 244, 244, transforms=get_transform(False))

with torch.no_grad():
    item = dataset[0]
    image = item[0]
    targets = item[1]

    # plot_img_bbox(image.permute(1, 2, 0), targets)

    images = list(img.to("cpu") for img in [image])
    pred = model(images)

plot_img_bbox(image.permute(1, 2, 0), pred[0])

print(pred)