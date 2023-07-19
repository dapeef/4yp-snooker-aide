import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

file_name = os.path.join("temp", os.listdir("temp")[-1])

mask = np.uint8(np.loadtxt(file_name))

plt.figure()
plt.imshow(mask)

# Detect edges
low_threshold = 0.33
high_threshold = .66
edges = cv2.Canny(mask, low_threshold, high_threshold)
plt.figure()
plt.imshow(edges)

# Detect lines
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
print(lines)

plt.show()