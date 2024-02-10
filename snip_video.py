import cv2
import os

def snip_video(video_path, save_directory, save_interval=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval in frames for one minute
    frame_interval = int(fps * save_interval) # Save interval in seconds

    # Initialize frame counter
    frame_count = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        
        # Check if frame is read successfully
        if not ret:
            break
        
        # Increment frame count
        frame_count += 1
        
        # Check if it's time to capture a frame (every minute)
        if frame_count % frame_interval == 0:
            # Save the frame to a file
            filename = os.path.join(save_directory, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Frame {frame_count} saved as {filename}.")
        
    # Release video capture object
    cap.release()


if __name__ == "__main__":
    snip_video("./data/ultimate pool/ultimate_pool.mp4", "./data/ultimate pool/train/images", 60)