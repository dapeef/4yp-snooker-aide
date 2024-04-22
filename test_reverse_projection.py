import numpy as np
import os
import find_edges
import measure_process
import matplotlib.pyplot as plt
import cv2


# image_file = "./validation/supervised/set-2/s10+_horizontal/images/p18_jpg.rf.b80e47920d7e1cc16439f6262859b266.jpg"
# corner_label_file = "./validation/supervised/set-2/s10+_horizontal/corner_labels/p18_jpg.rf.b80e47920d7e1cc16439f6262859b266.txt"
image_file = "./validation/supervised/set-2/s10+_horizontal/images/p18_undistorted.png"
corner_label_file = "./validation/supervised/set-2/s10+_horizontal/corner_labels/p18_undistorted.txt"
# corner_label_file = "validation/supervised/set-2/s10+_horizontal/corner_labels/p18_undistorted_tweaked.txt"
expected_output_file = "./validation/supervised/set-2/real-positions.txt"

# camera_name = "logitech_camera"
# camera_name = "laptop"
camera_name = "s10+_horizontal"

table_size = np.array([0.903, 1.676]) # English 8 ball
z_plane = -(0.037 - 0.0508/2)

mtx = np.load(os.path.join("./calibration", camera_name, "intrinsic_matrix.npy"))
dist_coeffs = np.load(os.path.join("./calibration", camera_name, "distortion.npy"))

corners = measure_process.load_points_from_corner_file(corner_label_file)
world_points = measure_process.load_points_from_real_file(expected_output_file)

rvec, tvec, projection = find_edges.get_perspective(corners, table_size, mtx, dist_coeffs)

image_points = find_edges.get_img_pos_from_perspective(world_points, mtx, rvec, tvec, z_plane)

img = cv2.imread(image_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
# Draw points
for point in image_points:
    plt.plot(*point, "wx")
# Draw table edges
for i in range(len(corners)):
    plt.plot([corners[i][0], corners[(i+1)%len(corners)][0]], [corners[i][1], corners[(i+1)%len(corners)][1]], "r")
plt.show()
