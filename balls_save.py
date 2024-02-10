import nn_utils

def create_balls_dataset(image_directory, output_directory):
    evaluator = nn_utils.EvaluateNet("./checkpoints/balls_model_single.pth")
    evaluator.create_dataset(image_directory)

    for i in range(len(evaluator.dataset)):
        print(f"Evaluating image {i+1}/{len(evaluator.dataset)}")
        evaluator.get_save_boxes(i, output_directory, 0.5)

if __name__ == "__main__":
    # create_balls_dataset(
    #     image_directory="./data/terrace/raw_images",
    #     output_directory="./data/terrace/dataset/train")
    
    create_balls_dataset(
        image_directory="./data/ultimate pool/train/images",
        output_directory="./data/ultimate pool/train/")