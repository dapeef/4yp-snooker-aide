import nn_utils
import matplotlib.pyplot as plt
import cv2

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')
import time
import cProfile


def evaluate(image_file):
    start_time = time.time()
    evaluator = nn_utils.EvaluateNet("./checkpoints/balls_model_multiple.pth", 20)
    print(f"Model load time: {time.time() - start_time}")
    start_time = time.time()
    evaluator.create_dataset(image_file)
    print(f"Dataset creation time: {time.time() - start_time}")
    start_time = time.time()
    target = evaluator.get_boxes(0)
    print(f"Model eval time: {time.time() - start_time}")
    start_time = time.time()

    return target

def filter_show(target, image_file, confidence_threshold):
    target = nn_utils.filter_boxes(target, max_results=100, confidence_threshold=confidence_threshold, remove_overlaps=False)
    target = nn_utils.get_bbox_centers(target)

    # img = evaluator.dataset.get_image(0) / 255
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    nn_utils.plot_img_bbox(img, target, f"NN balls with confidence of {confidence_threshold}")
    # plt.show()

    return target

def get_balls(image_file, confidence_threshold=0.5):
    target = evaluate(image_file)
    target = filter_show(target, image_file, confidence_threshold)

    return target["centres"]


def _test():
    image_file = "./images/terrace2.jpg"

    target = evaluate(image_file)

    for i in range(0, 10):
        print(f"Getting with confidence >= {i/10}...")
        filter_show(target, image_file, confidence_threshold=i/10)
    
    print(target["scores"])

    plt.show()

    
if __name__ == "__main__":
    # Profiler
    # cProfile.run('for i in range(10): _test()', filename="./temp/balls_eval.prof")

    _test()