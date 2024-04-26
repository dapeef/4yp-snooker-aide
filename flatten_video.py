import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_video_frames(video_file, output_file_name="progression_image", blank_image=None, contrast_exponent=0.3, frame_interval=1, start_time=0, end_time=np.inf):
    # Load the video
    cap = cv2.VideoCapture(video_file)

    # Check if video file is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get the frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an empty image to store the overlaid frames
    pure_result = np.zeros((frame_height, frame_width, 3), dtype=np.float64)
    subtract_result = np.zeros((frame_height, frame_width, 3), dtype=np.float64)
    
    blur_size = 5


    i = 0
    frame_count = 0
    num_frames = 0

    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        if i == 0:
            if blank_image is None:
                blank_image = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)

            cv2.imwrite(f'./temp/{output_file_name}_0.jpg', frame)

        if start_time <= i / 29.874 <= end_time:
            if frame_count % frame_interval == 0:
                # Overlay the current frame onto the result image
                pure_result += frame

                blur_frame = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
                subtract_result += cv2.absdiff(blur_frame, blank_image)

                num_frames += 1
            frame_count += 1
            
        
        i += 1

    # Normalisation
    pure_result /= num_frames

    subtract_result /= np.max(subtract_result)
    # subtract_result[subtract_result < 0.01] = 0
    subtract_result = np.power(subtract_result, contrast_exponent)
    subtract_result *= 255 *4/5
    # subtract_result /= num_frames
    subtract_result += blank_image / 5


    cv2.imwrite(f'./temp/{output_file_name}_pure.jpg', pure_result)
    cv2.imwrite(f'./temp/{output_file_name}_subtract.jpg', subtract_result)

    # Release the video capture object
    cap.release()

    return pure_result


if __name__ == "__main__":
    # Example usage
    # pure_overlay_video_frames("C:/Users/dapee/Downloads/9-ball-test.mp4", 1)

    overlay_video_frames("simulation_validation\cue ball.mp4",
                              output_file_name="cue ball",
                              frame_interval=10,
                              start_time=7.5,
                              end_time=14)