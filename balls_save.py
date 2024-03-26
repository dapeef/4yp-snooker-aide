import nn_utils
import os

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
    
    # create_balls_dataset(
    #     image_directory="./data/ultimate pool/train/images",
    #     output_directory="./data/ultimate pool/train/")

    # directory = "./validation/set-1/laptop_camera"
    # create_balls_dataset(directory, directory)

    def get_top_level_directories(directory):
        top_level_directories = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Check if the item is a directory
                if not any(os.path.isdir(os.path.join(item_path, sub_item)) for sub_item in os.listdir(item_path)):
                    # If the directory does not contain any subdirectories
                    top_level_directories.append(item_path)
                top_level_directories.extend(get_top_level_directories(item_path))
        return top_level_directories

    top_level_directories = get_top_level_directories("./validation")

    for folder_name in top_level_directories:
        print(f"Working on folder {folder_name}")
        create_balls_dataset(folder_name, folder_name)