import find_edges
import sam
import numpy as np
import matplotlib.pyplot as plt


# sam.create_mask(
#     image_file="images\\snooker1.png",
#     input_points = np.array([[600, 600], [1300, 600], [1625, 855]]),
#     input_labels = np.array([1, 1, 0])) # 1=foreground, 0=background
edges = find_edges.get_edges("images\\snooker1.png")

# edges = np.array([[[ 9.44000000e+02,  1.58079637e+00]],
#     [[ 3.21000000e+02,  1.57079637e+00]],
#     [[-1.31600000e+03,  2.87979317e+00]],
#     [[ 5.68000000e+02,  2.96705961e-01]]]) # lines which don't have any parallel

# edges = np.array([[[ 9.44000000e+02,  1.57079637e+00]],
#     [[ 3.21000000e+02,  1.57079637e+00]],
#     [[ 1.31600000e+03,  0]],
#     [[ 5.68000000e+02,  0]]]) # lines which don't have 2 pairs parallel

corners = find_edges.get_rect_corners(edges)

plt.show()