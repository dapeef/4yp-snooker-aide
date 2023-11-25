import find_edges
import sam
import numpy as np
import matplotlib.pyplot as plt
import pytorch_detection_eval



image_file = "images\\snooker1.png"
# sam.create_mask(
#     image_file=image_file,
#     input_points = np.array([[600, 600], [1300, 600], [1625, 855]]),
#     input_labels = np.array([1, 1, 0])) # 1=foreground, 0=background

# image_file = "images\\snooker3.png"
# sam.create_mask(
#     image_file=image_file,
#     input_points = np.array([[600, 600], [1850, 250]]),
#     input_labels = np.array([1, 0])) # 1=foreground, 0=background
# plt.show()


pockets = pytorch_detection_eval.get_boxes(image_file)

plt.show()

edges = find_edges.get_edges(image_file)



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

homography = find_edges.get_homography(corners, [1854, 3683])
# print(homography)

projection = find_edges.get_perspective(corners, [1854, 3683])

temp = np.dot(projection, np.array([0, 0, 0, 1]))

# print(temp/temp[2])


balls_homography = find_edges.get_balls_homography(homography, 44.45 - 52.5/2)



[x, y] = find_edges.get_world_point([1448, 321], homography)

# print(f"Transformed World Coordinates: ({x}, {y})")



img_balls = find_edges.find_balls(image_file[:-4] + "-masked.png")



# img_balls = [[500, 325],
#  [1445, 325],
#  [1615.36764586, 944.00007061],
#  [ 305.34316018, 944.00001335]]

real_balls = []
for ball in img_balls:
    real_balls.append(find_edges.get_world_point(ball, homography))



find_edges.display_table(real_balls)



plt.show()