import nn_utils

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')



def get_balls(image_file):
    evaluator = nn_utils.EvaluateNet("./checkpoints/balls_model_single.pth", 2)
    evaluator.create_dataset(image_file)
    target = evaluator.get_boxes(0)

    target = nn_utils.get_bbox_centers(target)

    img = evaluator.dataset.get_image(0) / 255
    nn_utils.plot_img_bbox(img, target, "NN balls")
    # plt.show()

    return target["centres"]
