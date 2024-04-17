import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time


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

    for line in lines:
        if line[0][0] < rho + rho_error and \
           line[0][0] > rho - rho_error and \
           line[0][1] < theta + theta_error and \
           line[0][1] > theta - theta_error:
            new_lines.append(line)
    
    return np.array(new_lines)

def filter_lines_multiple(lines, informing_lines, rho_error, theta_error):
    filtered_lines = []

    for informing_line in informing_lines:
        chosen_line = filter_lines(lines, informing_line[0][0], rho_error, informing_line[0][1], theta_error)[0].tolist()

        filtered_lines.append(chosen_line)
    
    return np.array(filtered_lines)

def plotLines(lines):
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            plt.plot([x1, x2], [y1, y2], color="green", linewidth=2)

def plotPoints(points):
    for point in points:
        plt.plot(point[0][0], point[0][1], "wx")

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

def standardise_lines(lines):
    new_lines = []

    for line in lines:
        rho = line[0][0]
        theta = line[0][1]

        if rho < 0:
            rho *= -1
            theta += np.pi
            if theta > np.pi:
                theta -= 2*np.pi
        
        new_lines.append([[rho, theta]])

    return new_lines

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

def get_sam_lines(mask_file=""):
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


    # plt.figure("SAM mask")
    plt.figure()
    # plt.title("SAM mask")
    plt.imshow(mask)
    plotLinesPolar(lines, mask.shape)
    # plotLinesPolar(sorted_lines[:3], mask.shape, "red")
    # plt.plot(x[0], x[1], "b+")


    # plt.figure("Dilated SAM mask")
    plt.figure()
    # plt.title("Dilated SAM mask")
    plt.imshow(dilated_mask)
    plotLinesPolar(lines, dilated_mask.shape)

    return lines, dilated_mask

def get_lines_from_pockets(image_file, pockets):
    # lines= [[[rho1, theta1]], [[rho2, theta2]], ...]:
    # [[[ 9.4500000e+02  1.5707964e+00]]
    #  [[ 3.2500000e+02  1.5707964e+00]]
    #  [[-1.3160000e+03  2.8797932e+00]]
    #  [[ 5.6600000e+02  2.9670596e-01]]]

    # Get centers of boxes
    pocket_points = []
    for box in pockets["boxes"]:
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        pocket_points.append([x, y])
    

    # Make sure there are enough pockets detected
    if len(pocket_points) != 6:
        print(f"ERROR Not enough pockets detected; {len(pockets)} detected, 6 needed")
        plt.show()
    assert len(pocket_points) == 6, f"Not enough pockets detected; {len(pockets)} detected, 6 needed"
    

    # Identify which pockets are corners
    side_pocket_idx = []

    for i in range(len(pocket_points)):
        x1 = pocket_points[i][0]
        y1 = pocket_points[i][1]

        vecs = []
        vecs_destinations = []

        for j in range(0, len(pocket_points)):
            if i != j:
                x2 = pocket_points[j][0]
                y2 = pocket_points[j][1]

                vec = np.array([x2-x1, y2-y1])

                for k in range(len(vecs)):
                    cos_theta = np.dot(vec, vecs[k]) / (np.linalg.norm(vec) * np.linalg.norm(vecs[k]))
                    dtheta = np.arccos(np.clip(cos_theta, -1, 1))

                    # print(pocket_points[i], pocket_points[j], pocket_points[k+1],180- np.rad2deg(dtheta))

                    dtheta_threshold = 10 # degrees
                    if dtheta > np.deg2rad(180-dtheta_threshold):
                        # These three points are nearly colinear, so either vec or vecs[k] is the vector to the middle
                        if np.linalg.norm(vec) < np.linalg.norm(vecs[k]):
                            side_pocket_idx.append(i)
                        else:
                            side_pocket_idx.append(vecs_destinations[k])

                vecs.append(vec)
                vecs_destinations.append(i)

    # print(side_pocket_idx)
    
    corners = []
    for i in range(len(pocket_points)):
        if not i in side_pocket_idx:
            corners.append(pocket_points[i])
    
    corners = sort_corners(corners)

    dists = []
    for i in range(len(corners)):
        for j in range(i+1, len(corners)):
            x1 = corners[i][0]
            x2 = corners[i][1]
            y1 = corners[j][0]
            y2 = corners[j][1]

            dists.append(np.linalg.norm([x2-x1, y2-y1]))
    
    max_dist = max(dists)

    if len(corners) != 4:
        print(f"ERROR Wrong number of corners detected; detected {len(corners)}, needed 4")
        plt.show()
    assert len(corners) == 4, f"Wrong number of corners detected; detected {len(corners)}, needed 4"

    # Get lines which go through corners
    lines = []

    for i in range(4):
        x1 = corners[i][0]
        y1 = corners[i][1]
        x2 = corners[(i+1)%4][0]
        y2 = corners[(i+1)%4][1]
        # print(f"[{x1}, {y1}], [{x2}, {y2}]")

        theta = np.arctan2(y2-y1, x2-x1) + np.pi/2

        unit_vec = np.array([np.cos(theta), np.sin(theta)])
        rho = np.dot(unit_vec, np.array([x1, y1]))

        # if abs(np.cos(theta)) > abs(np.sin(theta)):
        #     rho = x1/np.cos(theta)
        # else:
        #     rho = y1/np.sin(theta)

        lines.append([[rho, theta]])

    lines = standardise_lines(lines)

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # plt.figure("Lines from NN pockets")
    plt.figure()
    # plt.title("Lines from NN pockets")
    plt.imshow(image)
    plotLinesPolar(lines, image.shape, "red")
    plt.axis("off")
    
    # Make mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    corners = np.array([corners], dtype=np.int32)
    cv2.fillPoly(mask, corners, color=1)  # Set color to 1 for binary mask

    return lines, mask, max_dist

def save_masked_image(image_file, mask):
    # Get lines for original image
    image = cv2.imread(image_file)

    # Mask image with dilated SAM mask
    image = cv2.bitwise_and(image, image, mask=mask)
    masked_file_name = os.path.join("./temp", os.path.basename(image_file)[:-4] + "-masked.png")
    cv2.imwrite(masked_file_name, image)

def get_edges(image_file, informing_lines, mask, dilation_dist):
    # Get lines for original image
    image = cv2.imread(image_file)

    mask = dilate(mask, int(dilation_dist))
    # plt.figure("Dilated mask")
    plt.figure()
    plt.title("Dilated mask")
    plt.imshow(mask)
    plotLinesPolar(informing_lines, mask.shape, "red")

    # Mask image with dilated SAM mask
    image = cv2.bitwise_and(image, image, mask=mask)
    masked_file_name = os.path.join("./temp", os.path.basename(image_file)[:-4] + "-masked.png")
    cv2.imwrite(masked_file_name, image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = edge(image)
    lines = line_hough(edges) #[:10]
    lines = standardise_lines(lines)

    # plt.show()

    lines = filter_lines_multiple(lines, informing_lines, dilation_dist*2, 3/360*np.pi)

    # plt.figure("Hough on original image, informed by informing lines")
    plt.figure()
    plt.title("Hough on original image, informed by informing lines")
    plt.imshow(image)
    plotLinesPolar(lines, image.shape, "red")


    # plt.figure()
    # plt.imshow(edges)

    # print(sam_lines - lines)

    # plt.show()

    return lines

def get_rect_corners(lines):
    # following this process: https://stackoverflow.com/a/42904725

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
                # print("Yikes, these lines are parallel: t1={t1}, t2={t2}".format(t1=t1, t2=t2))
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

def get_perspective(img_corners, table_dims, mtx, dist_coeffs):
    width = table_dims[0]
    height = table_dims[1]

    world_pts = np.array([[0, height, 0], [width, height, 0], [width, 0, 0], [0, 0, 0]], dtype=np.float32)
    image_pts = np.array(img_corners, dtype=np.float32)

    retval, rvec, tvec = cv2.solvePnP(world_pts, image_pts, mtx, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rvec)

    # tvec /= tvec[2]

    projection = np.column_stack((rmat, tvec))


    # print("retval", retval, "rvec", rvec, "tvec", tvec, "rmat", rmat, "projection", projection, sep="\n")

    return rvec, tvec, projection

def get_world_pos_from_perspective(img_points, mtx, rvec, tvec, z_plane):
    K_inv = np.linalg.inv(mtx)
    rmat, _ = cv2.Rodrigues(rvec)
    tvec = np.transpose(tvec)[0]

    world_points = []
    
    for img_point in img_points:
        img_point_homogeneous = np.transpose(np.append(img_point, [1]))

        # print("tvec", tvec)
        # print("img point homogeneous", img_point_homogeneous)

        leftSideMat = np.linalg.inv(rmat).dot(np.linalg.inv(mtx)).dot(img_point_homogeneous)
        rightSideMat = np.linalg.inv(rmat).dot(tvec)

        # print("MATRICES", leftSideMat, rightSideMat)

        scale_factor = (z_plane + rightSideMat[2]) / leftSideMat[2]

        # print("Scale factor: ", scale_factor)

        world_point = leftSideMat * scale_factor - rightSideMat

        # print("world point:", world_point)

        world_points.append(world_point[:2])
    
    return np.array(world_points)

def get_balls_homography(cushion_homography, height_difference):
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(cushion_homography, np.identity(3))

    R = Rs[0]
    T = Ts[0]
    # T_mat 

    # print("H", cushion_homography)
    # print("num", num)
    # print("Rs", Rs)
    # print("Ts", Ts)
    # print("Ns", Ns)

def get_world_point(image_point, homography):
    img_pts_homogeneous = np.array([image_point[0], image_point[1], 1], dtype=np.float32)
    world_pts_homogeneous = np.dot(homography, img_pts_homogeneous)

    # print(world_pts_homogeneous)

    # Normalize by the third coordinate
    world_pts_normalized = world_pts_homogeneous / world_pts_homogeneous[2]

    # Extract the x and y coordinates in the world frame
    x_world = world_pts_normalized[0]
    y_world = world_pts_normalized[1]

    return np.array([x_world, y_world])

def find_balls_hough(image_file, single_channel_method="val", blur_radius=None, hough_threshold=None):
    # Read the image
    color_image_unscaled = cv2.imread(image_file)

    # Resize image
    resize_factor = 1500 / max(color_image_unscaled.shape)
    color_image = cv2.resize(color_image_unscaled, (0,0), fx=resize_factor, fy=resize_factor)


    # Define circle size based on image size
    # circle_size = (int(9/resize_factor), int(25/resize_factor))
    circle_size = (9, 25)

    if blur_radius is None:
        blur_radius = 7

    # print(circle_size)

    if single_channel_method == "val":
        # Convert to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Edge find using val
        val_image = hsv_image[:, :, 2]
        
        # plt.figure()
        # plt.imshow(val_channel, cmap="gray")
        # plt.axis("off")

        val_smooth = cv2.GaussianBlur(val_image, (blur_radius, blur_radius), 0)
        val_edges = cv2.Canny(val_image, 100, 200)
        # plt.figure()
        # plt.imshow(val_edges)
        # plt.figure()
        # plt.imshow(val_smooth)

        img = val_smooth
    
    elif single_channel_method == "greyscale":
        # Convert to greyscale
        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # plt.figure()
        # plt.imshow(grey_image, cmap="gray")
        # plt.axis("off")

        # Edge find using val
        grey_smooth = cv2.GaussianBlur(grey_image, (blur_radius, blur_radius), 0)
        grey_edges = cv2.Canny(grey_image, 100, 200)
        # plt.figure()
        # plt.imshow(val_edges)
        # plt.figure()
        # plt.imshow(val_smooth)

        img = grey_smooth

    if hough_threshold is None:
        hough_threshold = 12

    # Find circles
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=circle_size[0]*2,
        param1=200, # Canny edge sensitivity
        param2=hough_threshold, # Hough accumulator threshold
        minRadius=circle_size[0],
        maxRadius=circle_size[1]
    )
    
    # # Show smoothed image
    # plt.figure()
    # plt.imshow(img)

    

    if circles is not None:
        circles = np.uint16(np.around(circles))
        centers = circles[0, :, :2]
        radii = circles[0, :, 2]

        # print("Centers:")
        # print(centers)
        # print("Radii:")
        # print(radii)

        # # Plot
        # # plt.figure("Hough circle transform to find balls")
        # plt.figure()
        # # plt.title("Hough circle transform to find balls")
        # plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # # Display the image with detected circles
        # for i in range(len(centers)):
        #     plt.gca().add_patch(plt.Circle((centers[i, 0], centers[i, 1]), radii[i], color='b', fill=False))
        # # plt.show()
        
        real_circles = np.around(circles / resize_factor)
        real_centers = np.around(centers / resize_factor)
        real_radii = np.around(radii / resize_factor)

        plt.figure()
        plt.imshow(cv2.cvtColor(color_image_unscaled, cv2.COLOR_BGR2RGB))
        
        for i in range(len(centers)):
            plt.gca().add_patch(plt.Circle((real_centers[i, 0], real_centers[i, 1]), real_radii[i], color='b', fill=False))
    else:
        print("No circles found.")
        real_centers = []

    return np.array(real_centers)

def display_table(ball_centers, table_dims=[1854, 3683], ball_diameter=52.5, window_height=1000, title="Estimated ball positions"):
    # initialize our canvas as a 300x300 pixel image with 3 channels
    # (Red, Green, and Blue) with a black background

    def trans(x):
        return int(x * (window_height/table_dims[1]))
    
    white = (255, 255, 255)
    green = (34,139,34)
    blue = (0, 0, 255)

    # Create canvas
    canvas = np.ones((trans(table_dims[1]), trans(table_dims[0]), 3), dtype="uint8") * 255
    # canvas = cv2.imread("images\\blank_snooker_table.png")

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
        cv2.circle(canvas, (trans(ball[0]), window_height-trans(ball[1])), trans(ball_diameter/2), blue, -1)

    # plt.figure(title)
    plt.figure()
    plt.title(title)
    plt.imshow(canvas)

def create_ellipse_mask(image_shape):
    """
    Create a binary mask with an ellipse region.

    Args:
        image_shape: Tuple representing the shape (height, width) of the image.

    Returns:
        ellipse_mask: Binary mask with ellipse region (dtype: np.uint8).
    """
    # Create a blank mask with zeros
    ellipse_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Calculate the center and axes of the ellipse
    center = (image_shape[1] // 2, image_shape[0] // 2)  # (x, y) format
    axes = (image_shape[1] // 2, image_shape[0] // 2)  # (major axis, minor axis)

    # Draw the ellipse (fill it with white color)
    cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, color=255, thickness=-1)

    return ellipse_mask

def pixel_difference(old_image, new_image, threshold=30, proportion_threshold=0.5, show=False):
    def show_img(img):
        fig = plt.figure()
        fig.set_size_inches(2, 2)
        plt.imshow(img, aspect='auto')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')


    """
    Determine whether motion has occurred between two RGB images within an ellipse region.

    Args:
        old_image: numpy array representing the old image (RGB format).
        new_image: numpy array representing the new image (RGB format).
        threshold: Threshold value for considering a pixel as part of motion.
        pixel_threshold: Proportion of pixels above which motion is considered to have occurred.

    Returns:
        has_motion: Boolean value indicating whether motion has occurred.
    """
    # Convert images to numpy arrays if not already in that format
    old_image = np.array(old_image, dtype=np.int16)
    new_image = np.array(new_image, dtype=np.int16)

    # Compute the absolute difference between corresponding pixels
    diff_image = old_image - new_image

    # Calculate the size of the Gaussian kernel based on 1/4 of the smaller side of the image
    smaller_side = min(old_image.shape[0], old_image.shape[1])
    gaussian_kernel_size = max(3, smaller_side // 4)
    if gaussian_kernel_size % 2 == 0:
        gaussian_kernel_size += 1

    # Apply Gaussian smoothing to the masked difference image
    smoothed_diff_image = cv2.GaussianBlur(diff_image, (gaussian_kernel_size, gaussian_kernel_size), 0)

    # Take the absolute value of the smoothed difference image
    abs_smoothed_diff_image = np.abs(smoothed_diff_image)

    # Convert the absolute difference image to grayscale by averaging RGB channels
    diff_image_gray = np.mean(abs_smoothed_diff_image, axis=2)
    
    # Apply ellipse mask
    ellipse_mask = create_ellipse_mask(old_image.shape[:2])
    diff_image_gray_masked = cv2.bitwise_and(diff_image_gray, diff_image_gray, mask=ellipse_mask)

    # Apply threshold to identify motion
    motion_mask = diff_image_gray_masked > threshold

    # Count the number of pixels above the threshold within the ellipse region
    num_pixels_above_threshold = np.count_nonzero(motion_mask)

    # Calculate the proportion of pixels above the threshold relative to the area of the ellipse
    ellipse_area = np.count_nonzero(ellipse_mask)
    pixel_proportion = num_pixels_above_threshold / ellipse_area

    # Determine whether motion has occurred based on the pixel threshold
    has_motion = pixel_proportion > proportion_threshold

    if show:
        print(pixel_proportion, has_motion)

        show_img(old_image)
        show_img(new_image)
        show_img(ellipse_mask)
        show_img(np.mean(np.abs(diff_image), axis=2)) # Raw diff
        show_img(diff_image_gray)
        show_img(diff_image_gray_masked)
        show_img(motion_mask)

    return has_motion

def check_ball_movement(old_image, new_image, target, threshold=30, proportion_threshold=0.5, show=False):
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = np.array(np.rint(box), dtype=np.int16)

        old_sub_image = old_image[y_min:y_max, x_min:x_max]
        new_sub_image = new_image[y_min:y_max, x_min:x_max]

        if pixel_difference(old_sub_image, new_sub_image, threshold, proportion_threshold, show=False):
            if show:
                print("BALL HAS CHANGED")

                plt.figure()
                plt.imshow(old_sub_image)
                plt.figure()
                plt.imshow(new_sub_image)
                plt.show()

            return True
    
    return False


if __name__ == "__main__":
    # old_image = cv2.cvtColor(cv2.imread("./images/diff_old.jpg"), cv2.COLOR_BGR2RGB)
    # new_image = cv2.cvtColor(cv2.imread("./images/diff_new.jpg"), cv2.COLOR_BGR2RGB)
    # # new_image = cv2.cvtColor(cv2.imread("./images/diff_new2.jpg"), cv2.COLOR_BGR2RGB)
    # pixel_difference(old_image, new_image)
    # plt.show()

    old_image = cv2.cvtColor(cv2.imread("./images/terrace_phone.jpg"), cv2.COLOR_BGR2RGB)
    new_image = cv2.cvtColor(cv2.imread("./images/terrace.jpg"), cv2.COLOR_BGR2RGB)
    target = {'boxes': [[1620.4912,  612.8253, 1703.1621,  696.4203], [2576.4312,  886.6342, 2673.8899,  985.4148], [1918.3392,  620.5417, 1998.0477,  701.9626], [2150.3320, 1243.5760, 2266.0259, 1357.1327], [1985.6053,  440.0798, 2054.6768,  516.1108], [2230.3787,  628.5780, 2312.5488,  712.5162], [2195.4463,  877.0612, 2292.1348,  974.8874], [1468.1367,  853.5551, 1566.0989,  950.1334], [1706.2831, 1225.9446, 1825.3719, 1341.2914], [2523.7449,  451.2814, 2590.8489,  523.3998], [2544.7012,  632.2858, 2626.9685,  713.6154], [2618.5410, 1258.1598, 2742.2588, 1373.9728], [1248.9678, 1212.1758, 1369.5040, 1324.3998], [1725.0883,  436.8719, 1792.7833,  509.9797], [2252.0110,  453.9048, 2312.5576,  521.5685], [1835.8574,  870.7050, 1928.5109,  973.4150], [2839.9385,   43.7762, 3095.5750,  350.3239], [2151.3774,  341.1837, 2187.4722,  370.1146]], 'labels': [18, 18, 17, 18, 18, 17, 18, 17, 18, 17, 17, 17, 17, 18, 15, 13, 17, 17], 'scores': [0.9978, 0.9976, 0.9972, 0.9970, 0.9968, 0.9967, 0.9965, 0.9964, 0.9962, 0.9959, 0.9957, 0.9955, 0.9950, 0.9936, 0.9904, 0.9687, 0.7291, 0.5751], 'centers': [[1661.8267, 654.6228], [2625.1606, 936.0245], [1958.1935, 661.2521], [2208.1790, 1300.3544], [2020.1411, 478.0953], [2271.4639, 670.5471], [2243.7905, 925.9742], [1517.1178, 901.8442], [1765.8275, 1283.6179], [2557.2969, 487.3406], [2585.8350, 672.9506], [2680.3999, 1316.0663], [1309.2358, 1268.2878], [1758.9358, 473.4258], [2282.2842, 487.7366], [1882.1841, 922.0600], [2967.7568, 197.0500], [2169.4248, 355.6492]]}
    
    tick = time.time()
    print(check_ball_movement(old_image, new_image, target, show=False))
    print(time.time() - tick)
    print(f"Executed {len(target['boxes'])/18*16} comparisons in {(time.time() - tick)/18*16} seconds; {(time.time() - tick) / len(target['boxes']) /18*16} secs per comparison")