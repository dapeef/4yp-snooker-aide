import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



def edge(image):
    sigma = 0.33
    v = np.median(image)
    low_threshold = int(max(0, (1.0 - sigma) * v))
    high_threshold = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, low_threshold, high_threshold)

    return edges

def line_houghlinesp(edges):
    # Detect lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    return lines

def line_hough(edges):
    # Detect lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)

    lines = cv2.HoughLines(edges, rho, theta, threshold)

    return lines

def plotLines(lines):
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            plt.plot([x1, x2], [y1, y2], color="green", linewidth=2)

def plotLinesPolar(lines, image_shape, line_color="green"):
    plt.xlim(0, image_shape[1])
    plt.ylim(0, image_shape[0])
    plt.gca().invert_yaxis()

    for i in range(len(lines)):
        for rho, theta in lines[i]:
            x1 = 0
            x2 = image_shape[1]
            y1 = (rho - x1* np.cos(theta) )/ np.sin(theta)
            y2 = (rho - x2* np.cos(theta) )/ np.sin(theta)

            plt.plot([x1, x2], [y1, y2], color=line_color, linewidth=2)

def addRho(lines, padding):
    padded_lines = np.copy(lines)
    # print(padded_lines)
    # print(np.array(sorted(padded_lines.tolist(), key=lambda line: line[0][1]))) # Sort lines by theta value
    for i in range(len(padded_lines)):
        r = padded_lines[i][0][0]
        padded_lines[i][0][0] = r + r/abs(r) * padding

    return padded_lines

def getIntersectionPolar(line1, line2):
    # Get the intersection of two rho-theta lines

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    # x cos θ1 + y sin θ1 = r1
    # x cos θ2 + y sin θ2 = r2

    # AX = b, where
    # A = [cos θ1  sin θ1]   b = |r1|   X = |x|
    #     [cos θ2  sin θ2]       |r2|       |y|

    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1],
                  [rho2]])
    
    x = np.linalg.solve(A, b)

    return x


# Get lines for SAM mask
file_name = os.path.join("temp", os.listdir("temp")[-1])

mask = np.uint8(np.loadtxt(file_name))
edges = edge(mask)
lines = line_hough(edges)[:4]
padded_lines = addRho(lines, 20)
x = getIntersectionPolar(lines[0], lines[1])


plt.figure("SAM mask")
plt.title("SAM mask")
plt.imshow(mask)
plotLinesPolar(lines, mask.shape)
plotLinesPolar(padded_lines, mask.shape, "red")
plt.plot(x[0], x[1], "b+")


# Get lines for original image
image_file = "images\\snooker1.png"
image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = edge(image)
lines = line_hough(edges)[:10]

plt.figure("Hough on original image")
plt.imshow(image)
plotLinesPolar(lines, image.shape)


# plt.figure()
# plt.imshow(edges)

plt.show()