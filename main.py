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

image = cv2.imread(image_file)



# sam.create_mask(
#     image_file=image_file,
#     input_points = np.array([[600, 600], [1300, 600], [1625, 855]]),
#     input_labels = np.array([1, 1, 0])) # 1=foreground, 0=background

# image_file = "images/snooker3.png"
# sam.create_mask(
#     image_file=image_file,
#     input_points = np.array([[600, 600], [1850, 250]]),
#     input_labels = np.array([1, 0])) # 1=foreground, 0=background
# plt.show()



pockets = pockets_eval.evaluate(image_file)
# plt.show()

# image_file = image_file_masked

# sam_lines, sam_mask = find_edges.get_sam_lines()
# dilation_dist = 5
# edges = find_edges.get_edges(image_file, sam_lines, sam_mask, dilation_dist)

pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, pockets)
# plt.show()
# dilation_dist = max_dist / 15
# pocket_lines = find_edges.get_edges(image_file, pocket_lines, pocket_mask, dilation_dist) # V janky



# edges = np.array([[[ 9.44000000e+02,  1.58079637e+00]],
#     [[ 3.21000000e+02,  1.57079637e+00]],
#     [[-1.31600000e+03,  2.87979317e+00]],
#     [[ 5.68000000e+02,  2.96705961e-01]]]) # lines which don't have any parallel
# edges = np.array([[[ 9.44000000e+02,  1.57079637e+00]],
#     [[ 3.21000000e+02,  1.57079637e+00]],
#     [[ 1.31600000e+03,  0]],
#     [[ 5.68000000e+02,  0]]]) # lines which don't have 2 pairs parallel
corners = find_edges.get_rect_corners(pocket_lines)
# print(f"corners: {corners}")



# Snooker table measurements:
# Internal playing surface from the nose of the cushion rubber: 11 foot 9 inches x 5 foot 9 inches
# Cushion depth: 2"
# Distance to edge of green:
#   long side = 11' 9" + 2*2" = 12' 1" = 3.683m
#  short side =  5' 9" + 2*2" =  6' 1" = 1.854m
#
# Cushion height = 1.75" = 44.45mm
# Ball diameter = 52.5mm

# Pool table measurements:
# table_size = [0.9029999999999999, 1.6760000000000002]

# Intrinsic camera matrix
# with open("camera_matrix.json", 'r') as file:
#     camera_properties = json.load(file)
# print(camera_properties)
# fx, fy = camera_properties["focalLength"]
# x0 = image.shape[1] / 2
# y0 = image.shape[0] / 2
# x0, y0 = camera_properties["principalPoint"]
# K = np.array([[fx, 0 , x0],
#               [0 , fy, y0],
#               [0 , 0 , 1 ]])

table_size = np.array([1.854, 3.683]) # Snooker
table_size = np.array([0.903, 1.676]) # English 8 ball
# table_size = np.array([1.676, 0.903]) # English 8 ball (sideways)


# camera_name = "logitech_camera"
# camera_name = "laptop"
camera_name = "s10+_horizontal"

mtx = np.load(os.path.join("./calibration", camera_name, "intrinsic_matrix.npy"))
dist_coeffs = np.load(os.path.join("./calibration", camera_name, "distortion.npy"))
dist_coeffs = None

# print(f"K: {mtx}")
# print(f"dist_coeffs: {dist_coeffs}")

homography = find_edges.get_homography(corners, table_size)
# print(homography)
rvec, tvec, projection = find_edges.get_perspective(corners, table_size, mtx, dist_coeffs)
# print(projection)

# temp = np.dot(projection, np.array([0, 0, 0, 1]))
# print(temp/temp[2])
# temp = np.dot(projection, np.array([1854, 3683, 0, 1]))
# print(temp/temp[2])

width = table_size[0]
height = table_size[1]
points = np.array([[0, height, 0], [width, height, 0], [width, 0, 0], [0, 0, 0]], dtype=np.float32)
# points = np.array([[0, 0, 0], [*table_size, 0], [*(table_size/2), 0]], dtype=np.float32)
img_points, _ = cv2.projectPoints(points, rvec, tvec, mtx, dist_coeffs)
# print(f"img_points: {img_points}")
# find_edges.plotPoints(img_points)

img_points = corners
world_points = find_edges.get_world_pos_from_perspective(img_points, mtx, rvec, tvec, 0)

# print(f"world_points (should correspond to the corners of the table, with dims {table_size}):\n{world_points}")


balls_homography = find_edges.get_balls_homography(homography, 44.45 - 52.5/2)



[x, y] = find_edges.get_world_point([1448, 321], homography)
# print(f"Transformed World Coordinates: ({x}, {y})")



# img_balls = find_edges.find_balls(image_file[:-4] + "-masked.png")
img_balls = find_edges.find_balls_hough(image_file)
# img_balls = find_edges.find_balls(image_file, single_channel_method="greyscale")
img_balls = balls_eval_multiple.get_balls(image_file)
print(img_balls)
img_balls = img_balls["centers"]

# print(f"Homography ball positions: {img_balls}")


# img_balls = [[500, 325],
#  [1445, 325],
#  [1615.36764586, 944.00007061],
#  [ 305.34316018, 944.00001335]]
real_balls = []
for ball in img_balls:
    real_balls.append(find_edges.get_world_point(ball, homography))


real_balls_projection = find_edges.get_world_pos_from_perspective(img_balls, mtx, rvec, tvec, -(0.037 - 0.0508/2))
# print(f"Real ball locations: {real_balls_projection}")

find_edges.display_table(real_balls, table_dims=table_size, ball_diameter=0.0508, title="Homography balls")
find_edges.display_table(real_balls_projection, table_dims=table_size, ball_diameter=0.0508, title="Projection balls")



plt.show()