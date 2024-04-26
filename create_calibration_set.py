import cv2
import os

def purge_folder(folder_name):
    # Create folder if it doesn't already exist
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name)
            print(f"Directory '{folder_name}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{folder_name}': {e}")

    # Get a list of all files in the directory
    files_in_directory = os.listdir(folder_name)

    # Loop through all files and remove them
    for file_name in files_in_directory:
        file_path = os.path.join(folder_name, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    print("Folder purge complete")

def take_images(folder_name, camera_index):
    print(f"Initialising camera {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    print("Camera initialised")

    num = 0

    while cap.isOpened():
        retval, img = cap.read()

        k = cv2.waitKey(5)

        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            image_file = os.path.join(folder_name, 'img' + str(num) + '.png')
            cv2.imwrite(image_file, img)
            print(f"Image {num} saved to file '{image_file}'")
            num += 1

        cv2.imshow('Camera preview',img)

    # Release and destroy all windows before termination
    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    folder_name = './temp/calibration'
    purge_folder(folder_name)
    take_images(folder_name, 1)
    