import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

file_name = os.path.join("temp", os.listdir("temp")[-1])

mask = np.uint8(np.loadtxt(file_name))

plt.figure()
plt.imshow(mask)

sigma = 0.33
v = np.median(mask)
low_threshold = int(max(0, (1.0 - sigma) * v))
high_threshold = int(min(255, (1.0 + sigma) * v))
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

# Loop though the lines and draw them
# plt.figure()
# plt.xlim(0, )
for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        plt.plot([x1, x2], [y1, y2], color="green", linewidth=2)
        # plt.plot([200, 500], [200, 500], color="green", linewidth=2)

plt.show()