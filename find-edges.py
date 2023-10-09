import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



def edge(image):
    sigma = .33
    v = np.mean(image)
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
    sorted_lines = np.array(sorted(lines.tolist(), key=lambda line: line[0][1])) # Sort lines by theta value
    lines = sorted_lines[[0, 2, 1, 3]]
    
    # for i in range(len(padded_lines)):
    #     r = padded_lines[i][0][0]
    #     padded_lines[i][0][0] = r + r/abs(r) * padding

    # return padded_lines

def circular_kernel(diameter):
    """
    Create a circular kernel with a given diameter.

    Parameters:
        diameter (int): Diameter of the circle.

    Returns:
        numpy.ndarray: Circular kernel with a diameter, dtype=np.uint8.
    """
    radius = diameter / 2
    kernel = np.zeros((diameter, diameter), dtype=np.uint8)
    y, x = np.ogrid[:diameter, :diameter]
    distance_from_center = np.sqrt((x - radius)**2 + (y - radius)**2)
    kernel[distance_from_center <= radius] = 1  # Set True to 1 (np.uint8)
    return kernel

def dilate(mask, size_increase, iterations=1):
    kernel = circular_kernel(size_increase)

    return cv2.dilate(mask, kernel, iterations=iterations)

def erode(mask, size_increase, iterations=1):
    kernel = circular_kernel(size_increase)

    return cv2.erode(mask, kernel, iterations=iterations)

def enlarge(mask, size_increase):
    mask = erode(mask, size_increase)
    mask = dilate(mask, size_increase, 5)

    return mask

def line_mask(rho, theta, thickness, image_size):
    mask = np.zeros(image_size, dtype=np.uint8)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Calculate the line endpoints based on the diagonal length of the image
    line_length = int(np.sqrt(image_size[0]**2 + image_size[1]**2))
    x0 = rho * cos_theta
    y0 = rho * sin_theta
    x1 = int(x0 - line_length * (-sin_theta))
    y1 = int(y0 - line_length * (cos_theta))
    x2 = int(x0 + line_length * (-sin_theta))
    y2 = int(y0 + line_length * (cos_theta))

    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    return mask

def dilated_line_mask(line, thickness, image_size):
    rho, theta = line[0]

    mask = line_mask(rho, theta, image_size)

    mask = dilate(mask, thickness)

    return mask

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
dilated_mask = enlarge(mask, 10)
edges = edge(mask)
lines = line_hough(edges)[:4]
dilated_line = line_mask(lines[0][0][0], lines[0][0][1], 20, mask.shape)
# sorted_lines = np.array(sorted(lines.tolist(), key=lambda line: line[0][1])) # Sort lines by theta value
# sorted_lines = sorted_lines[[0, 2, 1, 3]]
padded_lines = addRho(lines, 20)
x = getIntersectionPolar(lines[0], lines[1])


plt.figure("SAM mask")
plt.title("SAM mask")
plt.imshow(edges)
plotLinesPolar(lines, mask.shape)
# plotLinesPolar(sorted_lines[:3], mask.shape, "red")
plt.plot(x[0], x[1], "b+")


plt.figure("Dilated SAM mask")
plt.title("Dilated SAM mask")
plt.imshow(dilated_mask)
plotLinesPolar(lines, dilated_mask.shape)




# Get lines for original image
image_file = "images\\snooker1.png"
image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mask image with dilated SAM mask
image = cv2.bitwise_and(image, image, mask=dilated_line)
cv2.imwrite("masked.png", image)

edges = edge(image)
lines = line_hough(edges)[:10]

plt.figure("Hough on original image")
plt.imshow(edges)
plotLinesPolar(lines, image.shape)


# plt.figure()
# plt.imshow(edges)

plt.show()