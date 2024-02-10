import find_edges
import sam
import numpy as np
import matplotlib.pyplot as plt
import pockets_eval
import balls_eval_single
import cv2
import json



image_file = "./images/snooker1.png"
image_file = "./images/terrace.jpg"
# image_file = './data/Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball/real_test/images/snooker2.jpg'
# image_file = "./images/snooker2.jpg"
# image_file = "./images/home_table3.jpg"



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



pockets = pockets_eval.get_pockets(image_file)
# plt.show()



# sam_lines, sam_mask = find_edges.get_sam_lines()
# dilation_dist = 5
# edges = find_edges.get_edges(image_file, sam_lines, sam_mask, dilation_dist)

pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, pockets)
# plt.show()
dilation_dist = max_dist / 15
edges = find_edges.get_edges(image_file, pocket_lines, pocket_mask, dilation_dist)



# edges = np.array([[[ 9.44000000e+02,  1.58079637e+00]],
#     [[ 3.21000000e+02,  1.57079637e+00]],
#     [[-1.31600000e+03,  2.87979317e+00]],
#     [[ 5.68000000e+02,  2.96705961e-01]]]) # lines which don't have any parallel
# edges = np.array([[[ 9.44000000e+02,  1.57079637e+00]],
#     [[ 3.21000000e+02,  1.57079637e+00]],
#     [[ 1.31600000e+03,  0]],
#     [[ 5.68000000e+02,  0]]]) # lines which don't have 2 pairs parallel
corners = find_edges.get_rect_corners(edges)
# print(corners)



# Snooker table measurements:
# Internal playing surface from the nose of the cushion rubber: 11 foot 9 inches x 5 foot 9 inches
# Cushion depth: 2"
# Distance to edge of green:
#   long side = 11' 9" + 2*2" = 12' 1" = 3.683m
#  short side =  5' 9" + 2*2" =  6' 1" = 1.854m
#
# Cushion height = 1.75" = 44.45mm
# Ball diameter = 52.5mm

# Intrinsic camera matrix
with open("camera_matrix.json", 'r') as file:
    camera_properties = json.load(file)
print(camera_properties)
fx, fy = camera_properties["focalLength"]
x0 = image.shape[1] / 2
y0 = image.shape[0] / 2
x0, y0 = camera_properties["principalPoint"]
K = np.array([[fx, 0 , x0],
              [0 , fy, y0],
              [0 , 0 , 1 ]])

print(K)

homography = find_edges.get_homography(corners, [1854, 3683])
# print(homography)
rvec, tvec = find_edges.get_perspective(corners, [1854, 3683], K)
# print(projection)

# temp = np.dot(projection, np.array([0, 0, 0, 1]))
# print(temp/temp[2])
# temp = np.dot(projection, np.array([1854, 3683, 0, 1]))
# print(temp/temp[2])

points = np.array([[0, 0, 0], [1854, 3683, 0], [927, 1841.5, 0]], dtype=np.float32)
img_points = cv2.projectPoints(points, rvec, tvec, K, None)[0]
print(img_points)
find_edges.plotPoints(img_points)


balls_homography = find_edges.get_balls_homography(homography, 44.45 - 52.5/2)



[x, y] = find_edges.get_world_point([1448, 321], homography)
# print(f"Transformed World Coordinates: ({x}, {y})")



# img_balls = find_edges.find_balls(image_file[:-4] + "-masked.png")
img_balls = balls_eval_single.get_balls(image_file)



# img_balls = [[500, 325],
#  [1445, 325],
#  [1615.36764586, 944.00007061],
#  [ 305.34316018, 944.00001335]]
real_balls = []
for ball in img_balls:
    real_balls.append(find_edges.get_world_point(ball, homography))



find_edges.display_table(real_balls)



plt.show()