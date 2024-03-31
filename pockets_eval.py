import cv2
import nn_utils
import matplotlib.pyplot as plt

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')



def get_pockets(image_file):
    evaluator = nn_utils.EvaluateNet("./checkpoints/pockets_model.pth", 2)
    
    evaluator.create_dataset(image_file)
    target = evaluator.get_boxes(0)

    target = nn_utils.filter_boxes(target, confidence_threshold=.1)
    target = nn_utils.get_bbox_centers(target)

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nn_utils.plot_img_bbox(img, target, "NN pockets")
    # plt.show()

    return target
