import numpy as np
import os
import pockets_eval
import balls_eval_multiple
import find_edges
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json


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


class TestResults:
    def __init__(self):
        pass

    def print_metrics(self):
        print(f"True positives: {self.true_positives}")
        print(f"False positives: {self.false_positives}")
        print(f"True negatives: {self.true_negatives}")
        print(f"False negatives: {self.false_negatives}")
        print(f"Mean error: {self.mean_error}")
        print(f"Mean error normalised: {self.mean_error_normalised}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"F1 score: {self.f1_score}")

    def pickle_metrics(self):
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "mean_error": self.mean_error,
            "mean_error_normalised": self.mean_error_normalised,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score
        }

    def save_to_file(self, file_name):
        with open(file_name, 'w') as json_file:
            json.dump(self.pickle_metrics(), json_file, indent=4)

    def mean_of_list(self, lst):
        if len(lst) == 0:
            return None  # Return None if the list is empty
        return sum(lst) / len(lst)

    def calculate_initial_metrics(self, detected_points, expected_points, min_table_dims):
        match_distance = min_table_dims / 15

        print(match_distance)

        self.detected_points = detected_points
        self.expected_points = expected_points
        self.match_distance = match_distance

        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        error_distances = []

        # Initialize a list to keep track of whether each expected point has been matched
        expected_matched = [False] * len(self.expected_points)

        for i, expected_point in enumerate(self.expected_points):
            found_match = False

            for detected_point in self.detected_points:
                # Calculate distance between detected point and expected point
                distance = np.linalg.norm(detected_point - expected_point)

                if distance <= self.match_distance:
                    # If within match distance, mark as a true positive
                    self.true_positives += 1
                    found_match = True

                    # Mark the expected point as matched
                    expected_matched[i] = True

                    # Add distance to error array
                    error_distances.append(distance)

                    break
            
            if not found_match:
                # If no match found for the expected point, it's a false negative
                self.false_negatives += 1

        # Count unmatched expected points as false negatives
        self.false_positives = len(self.detected_points) - sum(expected_matched)

        # Calculate mean error
        self.mean_error = self.mean_of_list(error_distances)
        self.mean_error_normalised = self.mean_error / min_table_dims

    def calculate_secondary_metrics(self):
        self.precision = self.calculate_precision()
        self.recall = self.calculate_recall()
        self.f1_score = self.calculate_f1_score()

    def calculate_precision(self):
        if self.true_positives + self.false_positives == 0:
            return 0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def calculate_recall(self):
        if self.true_positives + self.false_negatives == 0:
            return 0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    def calculate_f1_score(self):
        if self.precision + self.recall == 0:
            return 0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    

    def aggregate_results(self, results):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        mean_errors = []
        mean_errors_normalised = []

        for result in results:
            self.true_positives += result.true_positives
            self.false_positives += result.false_positives
            self.true_negatives += result.true_negatives
            self.false_negatives += result.false_negatives
            mean_errors.append(result.mean_error)
            mean_errors_normalised.append(result.mean_error_normalised)

        self.mean_error = self.mean_of_list(mean_errors)
        self.mean_error_normalised = self.mean_of_list(mean_errors_normalised)

        self.calculate_secondary_metrics()

class Test:
    def __init__(self, set, device):
        self.set = set
        self.device = device
        self.validation_folder = "./validation/supervised"
        self.folder = os.path.join(self.validation_folder, f"set-{self.set}", self.device)

        if not os.path.exists(self.folder):
            raise Exception(f"Validation set {set} with device '{device}' doesn't exist")

    def test_image_detection(self, detection_method, show=False):
        # detection_method values can be "hough", "nn"

        images_folder = os.path.join(self.folder, "images")
        labels_folder = os.path.join(self.folder, "labels")

        results = []

        for i, file in enumerate(os.listdir(images_folder)):
            image_file = os.path.join(images_folder, file)
            label_file = os.path.join(labels_folder, f"{os.path.splitext(file)[0]}.txt")

            print(f"\nAnalysing image: {image_file}")

            if detection_method == "hough":
                # Get masked image to reduce noise
                pockets = pockets_eval.get_pockets(image_file)
                pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, pockets)
                # find_edges.get_edges(image_file, pocket_lines, pocket_mask, 5)
                find_edges.save_masked_image(image_file, pocket_mask)
                masked_image_file = os.path.join("./temp", os.path.basename(image_file)[:-4] + "-masked.png")

                # Get pocket locations, and min_table_dim
                pocket_locations = find_edges.get_rect_corners(pocket_lines)
                min_table_dims = np.inf
                for j, pocket1 in enumerate(pocket_locations):
                    for k, pocket2 in enumerate(pocket_locations):
                        if j != k:
                            distance = np.linalg.norm(pocket2 - pocket1)
                            min_table_dims = min(min_table_dims, distance)

                print(f"Min table dims: {min_table_dims}")

                # Evaluate ball positions using hough
                detected_points = find_edges.find_balls(masked_image_file)
                # print(f"ball positions: {detected_points}")

                # Get expected ball positions
                expected_points = load_points_from_training_file(label_file, image_file)
                # print(f"expected ball positions: {expected_points}")

                results.append(TestResults())
                results[i].calculate_initial_metrics(detected_points, expected_points, min_table_dims)
                results[i].calculate_secondary_metrics()
                results[i].print_metrics()

                self.draw(image_file, detected_points, expected_points, min_table_dims/15, show=show)

            elif detection_method == "nn":
                pass

        self.result = TestResults()
        self.result.aggregate_results(results)
        print("Total results:")
        self.result.print_metrics()

        self.result.save_to_file(os.path.join(self.folder, f"{detection_method}_results.json"))

    def test_projection(self):
        expected_output_file = os.path.join(self.validation_folder, f"set-{self.set}", "real-positions.txt")

    def draw(self, image_file, detected_points, expected_points, match_radius=0, show=False):
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure()
        plt.imshow(img)
        ax = plt.gca()

        for point in detected_points:
            # cv2.drawMarker(img, point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            plt.plot(*point, marker="x", markersize=10, color="b")
        for point in expected_points:
            # cv2.drawMarker(img, point, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            plt.plot(*point, marker="x", markersize=10, color="g")
            # Draw a circle the size of match_radius
            circle = patches.Circle(point, match_radius, edgecolor='g', facecolor='none')
            ax.add_patch(circle)

        if show:
            plt.show()

def test_all():
    calibration_directory = "./calibration"
    entries = os.listdir(calibration_directory)
    device_names = [entry for entry in entries if os.path.isdir(os.path.join(calibration_directory, entry))]

    for set_num in range(1, 6):
        for device_name in device_names:
            for detection_method in ["hough", "nn"]:
                print(f"Trying set {set_num} with device '{device_name}' and detection method '{detection_method}'")

                try:
                    test = Test(set_num, device_name)
                except Exception as e:
                    print(e)
                    continue

                test.test_image_detection(detection_method)

if __name__ == "__main__":
    # test = Test(2, "s10+_horizontal")
    # test.test_image_detection(detection_method="hough", show=False)

    test_all()

    # plt.show()
