import numpy as np
import os
import find_edges
import measure_process
import matplotlib.pyplot as plt
import cv2
import calibrate_camera


image_file = "./validation/supervised/set-2/s10+_horizontal/images/p18_jpg.rf.b80e47920d7e1cc16439f6262859b266.jpg"
corner_label_file = "./validation/supervised/set-2/s10+_horizontal/corner_labels/p18_jpg.rf.b80e47920d7e1cc16439f6262859b266.txt"
# image_file = "validation\supervised\set-2\logitech_camera\images\w07_jpg.rf.0c791a84cf53abad273ec736655b291c.jpg"
# corner_label_file = "validation\supervised\set-2\logitech_camera\corner_labels\w07_jpg.rf.0c791a84cf53abad273ec736655b291c.txt"
# image_file = "./validation/supervised\set-1\s10+_vertical\images\p05-_jpg.rf.c7d2efa0d6cdeab027c614496f6fa637.jpg"
# corner_label_file = "./validation/supervised\set-1\s10+_vertical\corner_labels\p05-_jpg.rf.c7d2efa0d6cdeab027c614496f6fa637.txt"
# image_file = "./validation/supervised/set-2/s10+_horizontal/images/p18_undistorted.png"
# corner_label_file = "./validation/supervised/set-2/s10+_horizontal/corner_labels/p18_undistorted.txt"
# corner_label_file = "validation/supervised/set-2/s10+_horizontal/corner_labels/p18_undistorted_tweaked.txt"
expected_output_file = "./validation/supervised/set-2/real-positions.txt"


# image_file = "validation\supervised\set-2\s10+_horizontal\images\p25_jpg.rf.9627e784e810a4de1eb96393907f2cc4.jpg"
# corner_label_file = "validation\supervised\set-2\s10+_horizontal\corner_labels\p25_jpg.rf.9627e784e810a4de1eb96393907f2cc4.txt"

camera_name = "logitech_camera"
# camera_name = "laptop"
camera_name = "s10+_horizontal"
# camera_name = "s10+_vertical"

table_size = np.array([0.903, 1.676]) # English 8 ball
z_plane = -(0.037 - 0.0508/2)

mtx = np.load(os.path.join("./calibration", camera_name, "intrinsic_matrix.npy"))
dist_coeffs = np.load(os.path.join("./calibration", camera_name, "distortion.npy"))
# dist_coeffs = None

corners = measure_process.load_points_from_corner_file(corner_label_file)
world_points = measure_process.load_points_from_real_file(expected_output_file)
table_world_points = np.array([[0, 0], [0, table_size[1]], [table_size[0], table_size[1]], [table_size[0], 0]])

img = cv2.imread(image_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.undistort(img, mtx, dist_coeffs)
corners = cv2.undistortImagePoints(corners, mtx, dist_coeffs)
corners = np.array([x[0] for x in corners])

rvec, tvec, projection = find_edges.get_perspective(corners, table_size, mtx)
image_points = find_edges.get_img_pos_from_perspective(world_points, mtx, rvec, tvec, z_plane)
table_image_points = find_edges.get_img_pos_from_perspective(table_world_points, mtx, rvec, tvec, 0)

plt.imshow(img)
# Draw points
for point in np.concatenate((image_points, table_image_points)):
    plt.plot(*point, "wx")
# Draw ball lines
for i in range(len(image_points)):
    plt.plot([image_points[i][0], image_points[(i+1)%len(image_points)][0]], [image_points[i][1], image_points[(i+1)%len(image_points)][1]])
# Draw table lines
for i in range(len(table_image_points)):
    plt.plot([table_image_points[i][0], table_image_points[(i+1)%len(table_image_points)][0]], [table_image_points[i][1], table_image_points[(i+1)%len(table_image_points)][1]])
# Draw ground truthtable edges
for i in range(len(corners)):
    plt.plot([corners[i][0], corners[(i+1)%len(corners)][0]], [corners[i][1], corners[(i+1)%len(corners)][1]], "r")
plt.show()
