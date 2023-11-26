import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target):
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

def evaluate_item(item):
    image = item[0]
    # targets = item[1]

    # plot_img_bbox(image.permute(1, 2, 0), targets)

    images = list(img.to("cpu") for img in [image])

    with torch.no_grad():
        pred = model(images)

    # print(pred)

    filtered = filter_pockets(pred[0], confidence_threshold=.1)

    # plot_img_bbox(image.permute(1, 2, 0), pred[0])
    # plot_img_bbox(image.permute(1, 2, 0), filtered)

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

def get_boxes(image_file):
    dataset = TableImagesDataset(image_file, 244, 244, transforms=get_transform(False))

    target = evaluate_item(dataset[0])

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape

    scaled_target = scale_boxes(target, [width, height])

    plot_img_bbox(img, scaled_target)

    return scaled_target



print("Loading pocket detection model...")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

num_classes = 3
model = get_object_detection_model(num_classes)

checkpoint = torch.load("./checkpoints/model.pth", map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Pocket detection model loaded!")



if __name__ == "__main__":
    dataset = TableImagesDataset('data/Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball/real_test/images/', 244, 244, transforms=get_transform(False))
    
    for i in range(len(dataset)):
        print(f"Working on {i+1}/{len(dataset)}")

        evaluate_item(dataset[i])

    plt.show()
    