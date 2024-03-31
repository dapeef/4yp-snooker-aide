import numpy as np
import os
import pockets_eval
import balls_eval_multiple
import find_edges
import cv2
import matplotlib.pyplot as plt



TABLE_SIZE = np.array([0.903, 1.676]) # English 8 ball

def get_closest_point(point_of_interest, points):
    """Find the point in a list closest to a point of interest."""
    closest_dist = np.inf
    closest_point = None
    
    for point in points:
        distance = np.linalg.norm(point, point_of_interest)
        if distance < closest_dist:
            closest_dist = distance
            closest_point = point
    
    return closest_point, closest_dist

def load_points_from_file(file_path):
    """Create a NumPy array of points from a file."""
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 3:
                color, x, y = parts
                points.append(np.array([float(x), float(y)]))
    return np.array(points)

def load_points_from_training_file(file_path, image_file):
    image_res = cv2.imread(image_file).shape[:2]

    points = []

    for line in open(file_path, "r").read().split("\n"):
        line_values = [float(x) for x in line.split(" ")][1:]

        line_values[0] *= image_res[1]
        line_values[1] *= image_res[0]
        line_values[2] *= image_res[1]
        line_values[3] *= image_res[0]

        point = np.array(line_values[:2])
        # point = np.array([line_values[1], line_values[0]])
        points.append(point)

    return np.array(points)

def rotate_points(file_path, points):
    file_path = os.path.basename(file_path)
    file_name = os.path.splitext(file_path)[0]

    rotation_count = file_name.count("-")
    
    new_points = []

    for point in points:
        if rotation_count == 0:
            # No rotation
            new_points.append(point)

        if rotation_count == 1:
            # 180 degree (other end of the table)
            new_points.append(TABLE_SIZE - point)

        if rotation_count == 2:
            # 90 degree (toilet side)
            new_point = np.array([TABLE_SIZE[1] - point[1], point[0]])
            new_points.append(new_point)

        if rotation_count == 2:
            # 270 degree (window side)
            new_point = np.array([point[1], TABLE_SIZE[0] - point[0]])
            new_points.append(new_point)

    return np.array(new_points)



class Test:
    def __init__(self, set, device) -> None:
        self.set = set
        self.device = device
        self.validation_folder = "./validation/supervised"
        self.folder = os.path.join(self.validation_folder, f"set-{self.set}", self.device)

    def test_image_detection(self, detection_method):
        # detection_method values can be "hough", "nn"

        images_folder = os.path.join(self.folder, "images")
        labels_folder = os.path.join(self.folder, "labels")

        for file in os.listdir(images_folder):
            image_file = os.path.join(images_folder, file)
            label_file = os.path.join(labels_folder, f"{os.path.splitext(file)[0]}.txt")

            if detection_method == "hough":
                # Get masked image to reduce noise
                pockets = pockets_eval.get_pockets(image_file)
                pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, pockets)
                find_edges.get_edges(image_file, pocket_lines, pocket_mask, 5)
                masked_image_file = os.path.join("./temp", os.path.basename(image_file)[:-4] + "-masked.png")

                # Evaluate ball positions using hough
                detected_points = find_edges.find_balls(masked_image_file)
                print(f"ball positions: {detected_points}")

                # Get expected ball positions
                expected_points = load_points_from_training_file(label_file, image_file)
                print(f"expected ball positions: {expected_points}")

                self.draw(image_file, detected_points, expected_points)

            elif detection_method == "nn":
                pass

    def test_projection(self):
        expected_output_file = os.path.join(self.validation_folder, f"set-{self.set}", "real-positions.txt")

    def draw(self, image_file, detected_points, expected_points):
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure()
        plt.imshow(img)

        for point in detected_points:
            # cv2.drawMarker(img, point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            plt.plot(*point, marker="x", markersize=10, color="b")
        for point in expected_points:
            # cv2.drawMarker(img, point, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            plt.plot(*point, marker="x", markersize=10, color="g")

        plt.show()

if __name__ == "__main__":
    test = Test(1, "laptop_camera")

    test.test_image_detection(detection_method="hough")

    plt.show()
