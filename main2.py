import find_edges
import sam
import numpy as np
import matplotlib.pyplot as plt
import pockets_eval
import balls_eval_single
import balls_eval_multiple
import cv2
import json
import os
import calibrate_camera



image_file = "./images/snooker1.png"
image_file = "./images/terrace.jpg"
image_file = './data/Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball/real_test/images/snooker2.jpg'
image_file = "./images/snooker2.jpg"
image_file = "./images/home_table3.jpg"
image_file = "./images/terrace_webcam.jpg"
# image_file = "./images/terrace_laptop.jpg"
image_file = "./images/terrace_phone.jpg"
image_file_masked = "./temp\p25_jpg.rf.9627e784e810a4de1eb96393907f2cc4-masked.png" # Masked version of terrace_phone.jpg
# image_file = "./validation\supervised\set-2\s10+_horizontal\images\p15_jpg.rf.ee6a43fbae79cfa374e83110329bb374.jpg"
# image_file = "./validation\supervised\set-2\s10+_horizontal\images\p27+_jpg.rf.cd125a93197825dcbcef765bd3cfc4b3.jpg"
image_file = "validation\supervised\set-2\s10+_horizontal\images\p18_jpg.rf.b80e47920d7e1cc16439f6262859b266.jpg"


# camera_name = "logitech_camera"
# camera_name = "laptop"
camera_name = "s10+_horizontal"


table_size = np.array([1.854, 3.683]) # Snooker
table_size = np.array([0.903, 1.676]) # English 8 ball
# table_size = np.array([1.676, 0.903]) # English 8 ball (sideways)


mtx = np.load(os.path.join("./calibration", camera_name, "intrinsic_matrix.npy"))
dist_coeffs = np.load(os.path.join("./calibration", camera_name, "distortion.npy"))


undistorted_image_file = "./temp/undistorted.png"
img = cv2.imread(image_file)
img = cv2.undistort(img, mtx, dist_coeffs)
cv2.imwrite(undistorted_image_file, img)
image_file = undistorted_image_file


pockets = pockets_eval.evaluate(image_file)
pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, pockets)
corners = find_edges.get_rect_corners(pocket_lines)


homography = find_edges.get_homography(corners, table_size)
rvec, tvec, projection = find_edges.get_perspective(corners, table_size, mtx, None)


img_balls = find_edges.find_balls_hough(image_file)
img_balls = balls_eval_multiple.get_balls(image_file)
img_balls = img_balls["centers"]


real_balls = []
for ball in img_balls:
    real_balls.append(find_edges.get_world_point(ball, homography))
real_balls_projection = find_edges.get_world_pos_from_perspective(img_balls, mtx, rvec, tvec, -(0.037 - 0.0508/2))


find_edges.display_table(real_balls, table_dims=table_size, ball_diameter=0.0508, title="Homography balls")
find_edges.display_table(real_balls_projection, table_dims=table_size, ball_diameter=0.0508, title="Projection balls")


plt.show()