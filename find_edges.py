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

def line_hough(edges, threshold=15):
    # Detect lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 15  # minimum number of votes (intersections in Hough grid cell)

    lines = cv2.HoughLines(edges, rho, theta, threshold)

    return lines

def filter_lines(lines, rho, rho_error, theta, theta_error):
    new_lines = []

    for line in lines.tolist():
        if line[0][0] < rho + rho_error and \
           line[0][0] > rho - rho_error and \
           line[0][1] < theta + theta_error and \
           line[0][1] > theta - theta_error:
            new_lines.append(line)
    
    return np.array(new_lines)

def filter_lines_multiple(lines, sam_lines, rho_error, theta_error):
    filtered_lines = []

    for sam_line in sam_lines:
        chosen_line = filter_lines(lines, sam_line[0][0], rho_error, sam_line[0][1], theta_error)[0].tolist()

        filtered_lines.append(chosen_line)
    
    return np.array(filtered_lines)

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

def get_edges(image_file, mask_file=""):
    # Get lines for SAM mask
    if mask_file == "":
        mask_file = os.path.join("temp", os.listdir("temp")[-1])

    mask = np.uint8(np.loadtxt(mask_file))
    dilated_mask = enlarge(mask, 10)
    edges = edge(mask)
    lines = line_hough(edges)[:4]
    dilated_line = line_mask(lines[0][0][0], lines[0][0][1], 20, mask.shape)
    # sorted_lines = np.array(sorted(lines.tolist(), key=lambda line: line[0][1])) # Sort lines by theta value
    # sorted_lines = sorted_lines[[0, 2, 1, 3]]
    # padded_lines = addRho(lines, 20)
    # x = getIntersectionPolar(lines[0], lines[1])


    plt.figure("SAM mask")
    plt.title("SAM mask")
    plt.imshow(mask)
    plotLinesPolar(lines, mask.shape)
    # plotLinesPolar(sorted_lines[:3], mask.shape, "red")
    # plt.plot(x[0], x[1], "b+")


    plt.figure("Dilated SAM mask")
    plt.title("Dilated SAM mask")
    plt.imshow(dilated_mask)
    plotLinesPolar(lines, dilated_mask.shape)

    sam_lines = lines


    # Get lines for original image
    image_file = "images\\snooker1.png"
    # image_file = "images\\snooker2.jpg"
    image = cv2.imread(image_file)

    # Mask image with dilated SAM mask
    image = cv2.bitwise_and(image, image, mask=dilated_mask)
    cv2.imwrite(image_file[:-4] + "-masked.png", image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = edge(image)
    lines = line_hough(edges) #[:10]

    lines = filter_lines_multiple(lines, sam_lines, 5, 1/360*np.pi)

    plt.figure("Hough on original image, informed by SAM lines")
    plt.title("Hough on original image, informed by SAM lines")
    plt.imshow(image)
    plotLinesPolar(lines, image.shape, "red")


    # plt.figure()
    # plt.imshow(edges)

    # print(sam_lines - lines)

    # plt.show()

    return lines

def get_rect_corners(lines):
    # following this process: https://stackoverflow.com/a/42904725
    def sort_corners(corners):
        mean_x = (corners[0][0] + \
                  corners[1][0] + \
                  corners[2][0] + \
                  corners[3][0]) / 4
        mean_y = (corners[0][1] + \
                  corners[1][1] + \
                  corners[2][1] + \
                  corners[3][1]) / 4
        
        return np.array(sorted(corners, key=lambda point: np.arctan2((point[1]-mean_y), (point[0]-mean_x))))

    point_owners = [[], [], [], []]
    points = []
    parallel_lines = []

    # Get intersection points
    for i in range(len(lines)):
        r1 = lines[i][0][0]
        t1 = lines[i][0][1]

        ct1 = np.cos(t1);     # matrix element a
        st1 = np.sin(t1);     # b

        for j in range(i+1, len(lines)):
            r2 = lines[j][0][0]
            t2 = lines[j][0][1]

            ct2 = np.cos(t2);     # c
            st2 = np.sin(t2);     # d

            det = ct1*st2-st1*ct2

            if det != 0:
                x = (st2*r1-st1*r2)/det
                y = (-ct2*r1+ct1*r2)/det

                point_owners[i].append([len(points), ""])
                point_owners[j].append([len(points), ""])
                points.append([x, y])
            
            else:
                print("Yikes, these lines are parallel: t1={t1}, t2={t2}".format(t1=t1, t2=t2))
                parallel_lines.append(i)
                parallel_lines.append(j)
    
    # If any lines are parallel, the points on those lines are the 4 corners
    if len(parallel_lines) != 0:
        corners = []

        for i in parallel_lines[:2]:
            for label in point_owners[i]:
                corners.append(points[label[0]])

        return sort_corners(corners)

    # Classify whether points are in the middle or on the end of lines
    for i in range(len(point_owners)):
        if len(point_owners[i]) < 2:
            raise Exception("More than 2 lines parallel")
        
        elif len(point_owners[i]) == 2:
            point_owners[i][0][1] = "e"
            point_owners[i][1][1] = "e"
        
        else:
            [x1, y1] = points[point_owners[i][0][0]]
            [x2, y2] = points[point_owners[i][1][0]]
            [x3, y3] = points[point_owners[i][2][0]]

            scale_factors = []
            if (x3-x1) != 0:
                scale_factors.append((x2-x1) / (x3-x1))
            if y3-y1 != 0:
                scale_factors.append((y2-y1) / (y3-y1))

            if len(scale_factors) == 0:
                raise Exception("Two points are exactly the same")
            
            scale_factor = sum(scale_factors) / len(scale_factors)

            if scale_factor <= 0:
                point_owners[i][0][1] = "m"
                point_owners[i][1][1] = "e"
                point_owners[i][2][1] = "e"
            elif scale_factor <= 1:
                point_owners[i][0][1] = "e"
                point_owners[i][1][1] = "m"
                point_owners[i][2][1] = "e"
            else:
                point_owners[i][0][1] = "e"
                point_owners[i][1][1] = "e"
                point_owners[i][2][1] = "m"

    # Classify points into being both middle, partial, or both end
    point_types = ["", "", "", "", "", ""]

    for line in point_owners:
        for label in line:
            if point_types[label[0]] == "":
                point_types[label[0]] = label[1]
            
            elif sorted([point_types[label[0]], label[1]]) == ["e", "m"]:
                point_types[label[0]] = "p"

    # Choose the right points for the corners
    e = []
    p = []
    m = []

    for ind, i in enumerate(point_types):
        if i == "e":
            e.append(ind)
        if i == "p":
            p.append(ind)
        if i == "m":
            m.append(ind)
    
    def get_lines_on_point(point):
        lines = []
        for line_ind, line in enumerate(point_owners):
            for label in line:
                if label[0] == point:
                    lines.append(line_ind)
        return lines

    m_lines = get_lines_on_point(m[0])

    for ind, i in enumerate(e):
        e_lines = get_lines_on_point(i)

        if not e_lines[0] in m_lines and not e_lines[1] in m_lines:
            e_ind = ind
    
    corner_inds = [p[0], m[0], p[1], e[e_ind]]

    corners = []
    for i in corner_inds:
        corners.append(points[i])

    # print(points)
    # print(point_types)
    # print(point_owners)

    # print(corner_inds)
    # print(corners)

    return sort_corners(corners)

def get_homography(img_corners, table_dims):
    width = table_dims[0]
    height = table_dims[1]

    world_pts = np.array([[0, height], [width, height], [width, 0], [0, 0]])

    homography, _ = cv2.findHomography(img_corners, world_pts)

    return homography

def get_world_point(image_point, homography):
    img_pts_homogeneous = np.array([image_point[0], image_point[1], 1], dtype=np.float32)
    world_pts_homogeneous = np.dot(homography, img_pts_homogeneous)

    # Normalize by the third coordinate
    world_pts_normalized = world_pts_homogeneous / world_pts_homogeneous[2]

    # Extract the x and y coordinates in the world frame
    x_world = world_pts_normalized[0]
    y_world = world_pts_normalized[1]

    return [x_world, y_world]

def find_balls(image_file):
    # Read the image
    color_image = cv2.imread(image_file)

    # Resize image
    resize_factor = 1500 / max(color_image.shape)
    # color_image = cv2.resize(color_image, (0,0), fx=resize_factor, fy=resize_factor)

    # Convert to HSV
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Edge find using val
    val_channel = hsv_image[:, :, 2]
    blur_radius = 3
    val_smooth = cv2.GaussianBlur(val_channel, (blur_radius, blur_radius), 0)
    val_edges = cv2.Canny(val_channel, 100, 200)
    # plt.figure()
    # plt.imshow(val_edges)
    # plt.figure()
    # plt.imshow(val_smooth)

    # Define circle size based on image size
    circle_size = (int(9/resize_factor), int(15/resize_factor))

    # print(circle_size)

    # Find circles
    circles = cv2.HoughCircles(
        val_smooth,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=circle_size[0],
        param1=200,
        param2=12,
        minRadius=circle_size[0],
        maxRadius=circle_size[1]
    )

    
    plt.figure()
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        centers = circles[0, :, :2]
        radii = circles[0, :, 2]

        # print("Centers:")
        # print(centers)
        # print("Radii:")
        # print(radii)

        # Display the image with detected circles
        for i in range(len(centers)):
            plt.gca().add_patch(plt.Circle((centers[i, 0], centers[i, 1]), radii[i], color='b', fill=False))
        # plt.show()
    else:
        print("No circles found.")
        centers = []

    return np.array(centers)

def display_table(ball_centers, table_dims=[1854, 3683], ball_size=52.5, window_height=1000):
    # initialize our canvas as a 300x300 pixel image with 3 channels
    # (Red, Green, and Blue) with a black background

    def trans(x):
        return int(x * (window_height/table_dims[1]))
    
    white = (255, 255, 255)
    green = (34,139,34)
    blue = (255, 0, 0)

    # Create canvas
    canvas = np.zeros((trans(table_dims[1]), trans(table_dims[0]), 3), dtype="uint8")
    canvas = cv2.imread("images\\blank_snooker_table.png")

    # Add cushions (2" = 50.8mm away from edges)
    cushion_t = 50.8
    corners = [
        (trans(cushion_t), trans(cushion_t)),
        (trans(cushion_t), trans(table_dims[1]-cushion_t)),
        (trans(table_dims[0]-cushion_t), trans(table_dims[1]-cushion_t)),
        (trans(table_dims[0]-cushion_t), trans(cushion_t))]

    # cv2.rectangle(canvas, corners[0], corners[2], green)

    # for i in range(4):
    #     cv2.line(canvas, corners[i], corners[(i+1)%4], white)

    # Add balls
    for ball in ball_centers:
        cv2.circle(canvas, (trans(ball[0]), window_height-trans(ball[1])), trans(ball_size/2), blue, -1)

    plt.figure()
    plt.imshow(canvas)
