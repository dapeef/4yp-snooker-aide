
import torch
import torchvision
from torchvision.transforms import v2

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
    return v2.Compose(
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