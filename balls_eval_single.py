import nn_utils

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')



def get_balls(image_file):
    evaluator = nn_utils.EvaluateNet("./checkpoints/balls_model_single.pth", 2)
    evaluator.create_dataset(image_file)
    target = evaluator.get_draw_boxes(0, "NN balls")
    # plt.show()

    return target["centres"]
