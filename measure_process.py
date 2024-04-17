import numpy as np
import os
import pockets_eval
import balls_eval_multiple
import find_edges
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter
import json
import nn_utils
import time
import sam
import pooltool_test as pt_utils
import pooltool as pt

TABLE_SIZE = np.array([0.903, 1.676]) # English 8 ball
SET_NAMES = ['Close\ndispersed', 'Far\ndispersed', 'Obscured\nby cushion', 'Obscured by\nsame colour', 'Obscured by\nother colour']
METRIC_NAME_MAP = {
    "true_positives": "True positives",
    "false_positives": "False positives",
    "true_negatives": "True negatives",
    "false_negatives": "False negatives",
    "mean_error": "Absolute mean error", # (mm)",
    "mean_error_normalised": "Normalised mean error",
    "max_error": "Absolute max error",
    "max_error_normalised": "Normalised max error",
    "error": "Error (mm)",
    "error_normalised": "Normalised error",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 score",
    "accuracy": "Accuracy",
    "eval_time": "Evaluation time (secs)",
    "one_off_time": "One-off evaluation time (secs)"
}

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

def load_points_from_real_file(file_path):
    """Create a NumPy array of points from a file."""
    scale_factor = 1/100

    points = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()

            if len(parts) == 3:
                ball_type, x, y = parts
                points.append(np.array([float(x)*scale_factor, float(y)*scale_factor]))

    return np.array(points)

def load_points_from_corner_file(file_path):
    """Create a NumPy array of points from a file."""

    points = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()

            if len(parts) == 2:
                x, y = parts
                points.append(np.array([float(x), float(y)]))
            else:
                raise Exception(f"Bad corner file: {file_path}")
            
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

    rotation_symbol = file_path[3]
    
    new_points = []
    rotation_type = None

    for point in points:
        if rotation_symbol == "_" or rotation_symbol == ".":
            # No rotation
            rotation_type = 0
            new_points.append(point)

        if rotation_symbol == "-":
            # 180 degree (other end of the table)
            rotation_type = 1
            new_points.append(TABLE_SIZE - point)

        if rotation_symbol == "+":
            # 90 degree (toilet side)
            rotation_type = 2
            new_point = np.array([TABLE_SIZE[1] - point[1], point[0]])
            new_points.append(new_point)

        if rotation_symbol == "=":
            # 270 degree (window side)
            rotation_type = 3
            new_point = np.array([point[1], TABLE_SIZE[0] - point[0]])
            new_points.append(new_point)

    if rotation_type is None:
        new_points = points

    return np.array(new_points), rotation_type


class TestResults:
    def __init__(self):
        self.img_count = 1

    def print_metrics(self):
        print(f"True positives: {self.true_positives}")
        print(f"False positives: {self.false_positives}")
        print(f"True negatives: {self.true_negatives}")
        print(f"False negatives: {self.false_negatives}")
        print(f"Mean error: {self.mean_error}")
        print(f"Mean error normalised: {self.mean_error_normalised}")
        print(f"Max error: {self.max_error}")
        print(f"Max error normalised: {self.max_error_normalised}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"F1 score: {self.f1_score}")
        print(f"Accuracy: {self.accuracy}")
        print(f"Evaluation time: {self.eval_time}")
        print(f"One off time: {self.one_off_time}")
        print(f"Image count: {self.img_count}")

    def pickle_metrics(self):
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "mean_error": self.mean_error,
            "mean_error_normalised": self.mean_error_normalised,
            "max_error": self.max_error,
            "max_error_normalised": self.max_error_normalised,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "eval_time": self.eval_time,
            "one_off_time": self.one_off_time,
            "img_count": self.img_count
        }

    def unpickle_metrics(self, pickle):
        self.true_positives = pickle["true_positives"]
        self.false_positives = pickle["false_positives"]
        self.true_negatives = pickle["true_negatives"]
        self.false_negatives = pickle["false_negatives"]
        self.mean_error = pickle["mean_error"]
        self.mean_error_normalised = pickle["mean_error_normalised"]
        self.max_error = pickle["max_error"]
        self.max_error_normalised = pickle["max_error_normalised"]
        self.precision = pickle["precision"]
        self.recall = pickle["recall"]
        self.f1_score = pickle["f1_score"]
        self.accuracy = pickle["accuracy"]
        self.eval_time = pickle["eval_time"]
        self.one_off_time = pickle["one_off_time"]
        self.img_count = pickle["img_count"]

    def save_to_file(self, file_name):
        with open(file_name, 'w') as json_file:
            json.dump(self.pickle_metrics(), json_file, indent=4)
        
    def load_from_file(self, file_name):
        with open(file_name, 'r') as json_file:
            return json.load(json_file)

    def mean_of_list(self, lst):
        if len(lst) == 0:
            return None  # Return None if the list is empty
        return sum(lst) / len(lst)

    def calculate_initial_metrics(self, detected_points, expected_points, normalisation, match_radius):
        self.detected_points = detected_points
        self.expected_points = expected_points
        self.match_distance = match_radius

        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        error_distances = []

        # Initialize a list to keep track of whether each expected point has been matched
        expected_matched = [False] * len(self.expected_points)
        detected_matched = [False] * len(self.detected_points)

        for i, expected_point in enumerate(self.expected_points):
            min_distance = np.inf

            for j, detected_point in enumerate(self.detected_points):
                # Calculate distance between detected point and expected point
                distance = np.linalg.norm(detected_point - expected_point)

                if distance <= min_distance and not detected_matched[j]:
                    min_distance = distance
                    closest_point = j
            
            if min_distance <= self.match_distance:
                # Match found
                self.true_positives += 1
                expected_matched[i] = True

                detected_matched[closest_point] = True
                
                error_distances.append(min_distance)
                
            else:
                # If no match found for the expected point, it's a false negative
                self.false_negatives += 1

        # Count unmatched expected points as false negatives
        self.false_positives = len(self.detected_points) - self.true_positives # sum(expected_matched)

        # Calculate mean error
        self.mean_error = self.mean_of_list(error_distances)
        self.mean_error_normalised = self.mean_error / normalisation
        self.max_error = max(error_distances)
        self.max_error_normalised = self.max_error / normalisation

    def calculate_secondary_metrics(self):
        self.precision = self.calculate_precision()
        self.recall = self.calculate_recall()
        self.f1_score = self.calculate_f1_score()
        self.accuracy = self.calculate_accuracy()

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
    
    def calculate_accuracy(self):
        if self.true_positives + self.false_positives + self.true_negatives + self.false_negatives == 0:
            return 0
        return (self.true_positives + self.true_negatives) / (self.true_positives + self.false_positives + self.true_negatives + self.false_negatives)
    
    def aggregate_results(self, results):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        mean_errors = []
        mean_errors_normalised = []
        max_errors = []
        max_errors_normalised = []
        times = []
        one_off_times = []
        self.img_count = 0

        for result in results:
            self.true_positives += result.true_positives
            self.false_positives += result.false_positives
            self.true_negatives += result.true_negatives
            self.false_negatives += result.false_negatives

            mean_errors.append(result.mean_error)
            mean_errors_normalised.append(result.mean_error_normalised)
            max_errors.append(result.max_error)
            max_errors_normalised.append(result.max_error_normalised)
            times.append(result.eval_time)
            one_off_times.append(result.one_off_time)

            self.img_count += result.img_count

        self.calculate_secondary_metrics()

        self.mean_error = self.mean_of_list(mean_errors)
        self.mean_error_normalised = self.mean_of_list(mean_errors_normalised)
        self.max_error = max(max_errors)
        self.max_error_normalised = max(max_errors_normalised)
        self.eval_time = self.mean_of_list(times)
        self.one_off_time = self.mean_of_list(one_off_times)

class Test:
    def __init__(self, set, device):
        self.set = set
        self.device = device
        self.validation_folder = "./validation/supervised"
        self.folder = os.path.join(self.validation_folder, f"set-{self.set}", self.device)

        if not os.path.exists(self.folder):
            raise Exception(f"Validation set {set} with device '{device}' doesn't exist")
        
        self.images_folder = os.path.join(self.folder, "images")
        self.labels_folder = os.path.join(self.folder, "labels")
        self.corner_labels_folder = os.path.join(self.folder, "corner_labels")
        self.sam_labels_folder = os.path.join(self.folder, "sam_labels")

    def test_ball_detection(self, method_name, show=False, blur_radius=None, hough_threshold=None):
        # method_name values can be "hough", "hough_masked", "nn", "nn_masked"

        balls_evaluator_init_time_start = time.time()
        balls_evaluator = nn_utils.EvaluateNet("./checkpoints/balls_model_multiple.pth", 20)
        balls_evaluator_init_time = time.time() - balls_evaluator_init_time_start

        pockets_evaluator_init_time_start = time.time()
        pockets_evaluator = nn_utils.EvaluateNet("./checkpoints/pockets_model.pth", 2)
        pockets_evaluator_init_time = time.time() - pockets_evaluator_init_time_start

        results = []

        for i, file in enumerate(os.listdir(self.images_folder)):
            image_file = os.path.join(self.images_folder, file)
            label_file = os.path.join(self.labels_folder, f"{os.path.splitext(file)[0]}.txt")

            print(f"\nAnalysing image: {image_file} with detection method: {method_name}")

            mask_time_start = time.time()
            # Get masked image to reduce noise
            pockets = pockets_eval.evaluate_from_evaluator(pockets_evaluator, image_file)
            pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, pockets)
            # find_edges.get_edges(image_file, pocket_lines, pocket_mask, 5)
            find_edges.save_masked_image(image_file, pocket_mask)
            masked_image_file = os.path.join("./temp", os.path.basename(image_file)[:-4] + "-masked.png")
            mask_time = time.time() - mask_time_start

            # Get pocket locations, and min_table_dim
            pocket_locations = find_edges.get_rect_corners(pocket_lines)
            min_table_dims = np.inf
            for j, pocket1 in enumerate(pocket_locations):
                for k, pocket2 in enumerate(pocket_locations):
                    if j != k:
                        distance = np.linalg.norm(pocket2 - pocket1)
                        min_table_dims = min(min_table_dims, distance)

            if method_name == "hough":
                method_time_start = time.time()
                # print(f"Min table dims: {min_table_dims}")

                # Evaluate ball positions using hough
                detected_points = find_edges.find_balls_hough(image_file)
                # print(f"ball positions: {detected_points}")

                method_time = time.time() - method_time_start
                one_off_time = method_time

            elif method_name == "hough_masked":
                method_time_start = time.time()
                # print(f"Min table dims: {min_table_dims}")

                # Evaluate ball positions using hough
                detected_points = find_edges.find_balls_hough(masked_image_file, blur_radius=blur_radius, hough_threshold=hough_threshold)
                    
                # print(f"ball positions: {detected_points}")

                method_time = time.time() - method_time_start
                method_time += mask_time
                one_off_time = pockets_evaluator_init_time + method_time
                
            elif method_name == "hough_grey_masked":
                method_time_start = time.time()
                # print(f"Min table dims: {min_table_dims}")

                # Evaluate ball positions using hough
                detected_points = find_edges.find_balls_hough(masked_image_file, single_channel_method="greyscale")
                # print(f"ball positions: {detected_points}")

                method_time = time.time() - method_time_start
                method_time += mask_time
                one_off_time = pockets_evaluator_init_time + method_time

            elif method_name == "nn":
                method_time_start = time.time()
                # print(f"Min table dims: {min_table_dims}")

                # Get detected ball positions
                target = balls_eval_multiple.evaluate_from_evaluator(balls_evaluator, image_file)
                target = nn_utils.filter_boxes(target, max_results=100, confidence_threshold=0.5, remove_overlaps=False)
                target = nn_utils.get_bbox_centers(target)

                detected_points = target["centers"]
                # print(f"ball positions: {detected_points}")

                method_time = time.time() - method_time_start
                one_off_time = method_time + balls_evaluator_init_time

            elif method_name == "nn_masked":
                method_time_start = time.time()
                # print(f"Min table dims: {min_table_dims}")

                # Get detected ball positions
                target = balls_eval_multiple.evaluate_from_evaluator(balls_evaluator, masked_image_file)
                target = nn_utils.filter_boxes(target, max_results=100, confidence_threshold=0.5, remove_overlaps=False)
                target = nn_utils.get_bbox_centers(target)

                detected_points = target["centers"]
                # print(f"ball positions: {detected_points}")

                method_time = time.time() - method_time_start
                method_time += mask_time
                one_off_time = pockets_evaluator_init_time + balls_evaluator_init_time + method_time

            # Get expected ball positions
            expected_points = load_points_from_training_file(label_file, image_file)
            # print(f"expected ball positions: {expected_points}")

            match_radius = min_table_dims/15

            results.append(TestResults())
            results[i].calculate_initial_metrics(detected_points, expected_points, min_table_dims, match_radius)
            results[i].calculate_secondary_metrics()
            results[i].eval_time = method_time
            results[i].one_off_time = one_off_time
            if show:
                results[i].print_metrics()

            self.draw(detected_points, expected_points, image_file=image_file, match_radius=match_radius, show=show)

        self.result = TestResults()
        self.result.aggregate_results(results)
        if show:
            print("Total results:")
            self.result.print_metrics()

        if not blur_radius is None:
            method_file_name = f"{method_name}_rad_{blur_radius}"

        elif not hough_threshold is None:
            method_file_name = f"{method_name}_thresh_{hough_threshold}"

        else:
            method_file_name = method_name

        save_file_name = os.path.join(self.folder, f"{method_file_name}_results.json")

        self.result.save_to_file(save_file_name)

        print(f"Just saved to {save_file_name}")

        return self.result

    def test_projection(self, method_name, show=False):
        # Get expected points
        expected_output_file = os.path.join(self.validation_folder, f"set-{self.set}", "real-positions.txt")
        original_expected_points = load_points_from_real_file(expected_output_file)

        # Get camera calibration data
        camera_load_time_start = time.time()
        mtx = np.load(os.path.join("./calibration", self.device, "intrinsic_matrix.npy"))
        dist_coeffs = np.load(os.path.join("./calibration", self.device, "distortion.npy"))
        camera_load_time = time.time() - camera_load_time_start

        # Table size
        # table_size = np.array([1.854, 3.683]) # Snooker
        # table_size = np.array([0.903, 1.676]) # English 8 ball
        # table_size = np.array([1.676, 0.903]) # English 8 ball
        min_table_dims = TABLE_SIZE[0]

        results = []

        for i, file in enumerate(os.listdir(self.images_folder)):
            image_file = os.path.join(self.images_folder, file)
            label_file = os.path.join(self.labels_folder, f"{os.path.splitext(file)[0]}.txt")
            corner_label_file = os.path.join(self.corner_labels_folder, f"{os.path.splitext(file)[0]}.txt")

            expected_points, rot_type = rotate_points(image_file, original_expected_points)

            if rot_type == 0 or rot_type == 1:
                local_table_size = TABLE_SIZE
            else:
                local_table_size = np.array([TABLE_SIZE[1], TABLE_SIZE[0]])

            image_points = load_points_from_training_file(label_file, image_file)
            corners = load_points_from_corner_file(corner_label_file)

            if method_name == "homography":
                method_time_start = time.time()

                homography = find_edges.get_homography(corners, local_table_size)
                
                detected_points = []
                for ball in image_points:
                    detected_points.append(find_edges.get_world_point(ball, homography))

                method_time = time.time() - method_time_start
                one_off_time = method_time

            elif method_name == "projection":
                method_time_start = time.time()

                rvec, tvec, projection = find_edges.get_perspective(corners, local_table_size, mtx, dist_coeffs)
                detected_points = find_edges.get_world_pos_from_perspective(image_points, mtx, rvec, tvec, -(0.037 - 0.0508/2))

                method_time = time.time() - method_time_start
                one_off_time = method_time + camera_load_time

            match_radius = .1

            img = cv2.imread(image_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure()
            # plt.title(image_file)
            a = plt.gca()
            a.imshow(img)
            plt.axis('off')

            results.append(TestResults())
            results[i].calculate_initial_metrics(detected_points, expected_points, min_table_dims, match_radius)
            results[i].calculate_secondary_metrics()
            results[i].eval_time = method_time
            results[i].one_off_time = one_off_time
            if show:
                results[i].print_metrics()

            if results[i].false_positives != 0 or \
                   results[i].true_negatives != 0 or \
                   results[i].false_negatives != 0:
                print("!!! Not all balls matching, check this is correct")

            self.draw(detected_points, expected_points, image_size=local_table_size, match_radius=match_radius, show=show)

        self.result = TestResults()
        self.result.aggregate_results(results)
        if show:
            print("Total results:")
            self.result.print_metrics()

        self.result.save_to_file(os.path.join(self.folder, f"{method_name}_results.json"))

        return self.result

    def test_pocket_detection(self, method_name, show=False):
        # method_name values can be "sam", "nn_corner"
        
        if method_name == "sam":
            sam_evaluator_init_time_start = time.time()
            sam_evaluator = sam.initialise_sam()
            sam_evaluator_init_time = time.time() - sam_evaluator_init_time_start

        elif method_name == "nn_corner":
            pockets_evaluator_init_time_start = time.time()
            pockets_evaluator = nn_utils.EvaluateNet("./checkpoints/pockets_model.pth", 2)
            pockets_evaluator_init_time = time.time() - pockets_evaluator_init_time_start

        results = []

        for i, file in enumerate(os.listdir(self.images_folder)):
            image_file = os.path.join(self.images_folder, file)
            corner_label_file = os.path.join(self.corner_labels_folder, f"{os.path.splitext(file)[0]}.txt")
            sam_label_file = os.path.join(self.sam_labels_folder, f"{os.path.splitext(file)[0]}.json")

            print(f"\nAnalysing image: {image_file} with detection method: {method_name}")
            
            expected_points = load_points_from_corner_file(corner_label_file)

            # Get pocket locations, and min_table_dim
            pocket_locations = expected_points
            min_table_dims = np.inf
            for j, pocket1 in enumerate(pocket_locations):
                for k, pocket2 in enumerate(pocket_locations):
                    if j != k:
                        distance = np.linalg.norm(pocket2 - pocket1)
                        min_table_dims = min(min_table_dims, distance)

            if method_name == "sam":
                method_time_start = time.time()
                # print(f"Min table dims: {min_table_dims}")

                # image_file = "./images/snooker1.png"
                # input_points = np.array([[600, 600], [1300, 600], [1625, 855]])
                # input_labels = np.array([1, 1, 0]) # 1=foreground, 0=background
                # expected_points = np.array([[494, 321], [1448, 321], [1618, 946], [305, 944]])

                with open(sam_label_file, "r") as f:
                    sam_labels = json.load(f)

                input_points = np.array(sam_labels["input_points"])
                input_labels = np.array(sam_labels["input_labels"])

                mask_file = os.path.join("./temp", f"{os.path.splitext(file)[0]}-mask.txt")

                sam.create_mask_from_model(sam_evaluator, image_file, input_points, input_labels, mask_file)
                lines = find_edges.get_sam_lines(mask_file)
                lines = lines[0]
                detected_points = find_edges.get_rect_corners(lines)

                method_time = time.time() - method_time_start
                one_off_time = method_time + sam_evaluator_init_time

            elif method_name == "nn_corner":
                method_time_start = time.time()
                # print(f"Min table dims: {min_table_dims}")

                pockets = pockets_eval.evaluate_from_evaluator(pockets_evaluator, image_file)
                pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, pockets)
                detected_points = find_edges.get_rect_corners(pocket_lines)

                method_time = time.time() - method_time_start
                one_off_time = method_time + pockets_evaluator_init_time

            match_radius = min_table_dims/5

            results.append(TestResults())
            results[i].calculate_initial_metrics(detected_points, expected_points, min_table_dims, match_radius)
            results[i].calculate_secondary_metrics()
            results[i].eval_time = method_time
            results[i].one_off_time = one_off_time
            if show:
                results[i].print_metrics()

            self.draw(detected_points, expected_points, image_file=image_file, match_radius=match_radius, show=show)

        self.result = TestResults()
        self.result.aggregate_results(results)
        if show:
            print("Total results:")
            self.result.print_metrics()

        self.result.save_to_file(os.path.join(self.folder, f"{method_name}_results.json"))

        return self.result

    def test_end_to_end_detection(self, match_radius=None, show=False):
        def get_pockets(pockets_evaluator, image_file):
            pockets_evaluator.create_dataset(image_file)
            target = pockets_evaluator.get_boxes(0)

            target = nn_utils.filter_boxes(target, confidence_threshold=.1)
            target = nn_utils.get_bbox_centers(target)

            return target
        
        def get_balls(balls_evaluator, image_file):
            balls_evaluator.create_dataset(image_file)
            target = balls_evaluator.get_boxes(0)

            target = nn_utils.filter_boxes(target, max_results=100, confidence_threshold=0.5, remove_overlaps=False)
            target = nn_utils.get_bbox_centers(target)

            return target

        def circle_line_overlap_vector(circle_center, radius, p1, p2):
            """Calculate the vector to move the circle to resolve overlap with the line segment."""
            p1 = p1[:2]
            p2 = p2[:2]

            # Vector representing the line segment
            line_vector = p2 - p1
            # Vector from the start of the line segment to the center of the circle
            start_to_center = circle_center - p1
            # Projection of start_to_center onto the line_vector
            projection = np.dot(start_to_center, line_vector) / np.dot(line_vector, line_vector)

            if projection < 0:
                closest_point = p1
            elif projection > 1:
                closest_point = p2
            else:
                closest_point = p1 + projection * line_vector
            
            # Vector from the center of the circle to the closest point on the line segment
            closest_vector = circle_center - closest_point

            dist = np.linalg.norm(closest_vector)


            if dist > radius:
                return np.array([0, 0])
            
            else:
                # Calculate the vector to move the circle
                move_vector = closest_vector/dist * (radius - dist)
                
                return move_vector
        
        def circle_circle_overlap_vector(c1, r1, c2, r2):
            # Work out how far to move c1 to make it fit next to c2
            c2_c1 = c1 - c2
            dist = np.linalg.norm(c2_c1)

            if dist > r1 + r2:
                return np.array([0, 0])
            else:
                return c2_c1/dist * (r1 + r2 - dist)

        balls_evaluator_init_time_start = time.time()
        balls_evaluator = nn_utils.EvaluateNet("./checkpoints/balls_model_multiple.pth", 20)
        balls_evaluator_init_time = time.time() - balls_evaluator_init_time_start

        pockets_evaluator_init_time_start = time.time()
        pockets_evaluator = nn_utils.EvaluateNet("./checkpoints/pockets_model.pth", 2)
        pockets_evaluator_init_time = time.time() - pockets_evaluator_init_time_start

        table = pt.Table.from_table_specs(pt_utils.english_8_ball_table_specs())
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=table, balls={}, cue=cue)

        cushion_height = 0.037

        min_table_dims = TABLE_SIZE[0]

        results = []
        method_time = 0
        one_off_time = 0

        for i, file in enumerate(os.listdir(self.images_folder)):
            image_file = os.path.join(self.images_folder, file)
            expected_output_file = os.path.join(self.validation_folder, f"set-{self.set}", "real-positions.txt")
            original_expected_points = load_points_from_real_file(expected_output_file)
            expected_points, rot_type = rotate_points(image_file, original_expected_points)
            detected_points = []

            if rot_type != 0:
                print(f"!!! Rejecting file {file} because it has bad orientation")
                # time.sleep(5)
            else:
                print(f"\nAnalysing image: {image_file} end-to-end")

                # Evaluate NNs
                pockets_eval_start = time.time()
                pockets_target = get_pockets(pockets_evaluator, image_file)
                balls_eval_start = time.time()
                balls_target = get_balls(balls_evaluator, image_file)
                one_off_time += time.time() - pockets_eval_start
                method_time += time.time() - balls_eval_start
                
                main_method_start = time.time()
                # Get corner points from pocket output
                pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, pockets_target)
                corners = find_edges.get_rect_corners(pocket_lines)

                # Get homography between pixelspace and tablespace
                cushion_thickness_real = pt_utils.english_8_ball_table_specs().cushion_width

                table_size = [shot.table.w + 2*cushion_thickness_real, shot.table.l + 2*cushion_thickness_real]
                # print(f"Table size: {table_size}")

                calibration_directory = "./calibration"
                mtx = np.load(os.path.join(calibration_directory, self.device, "intrinsic_matrix.npy"))
                dist_coeffs = np.load(os.path.join(calibration_directory, self.device, "distortion.npy"))
                rvec, tvec, projection = find_edges.get_perspective(corners, table_size, mtx, dist_coeffs)


                # Process all balls
                # Inflate ball radius slightly so it moves the balls far enough away from cushions
                simulation_nudge = 0.0001
                ball_radius = pt_utils.english_8_ball_ball_params().R + simulation_nudge

                balls = {}

                red_id = 1
                yellow_id = 9

                for j in range(len(balls_target["labels"])):
                    label = balls_target["labels"][j]
                    # ball_type = self.balls_class_conversion[label]
                    img_center = balls_target["centers"][j]
                    # real_center = find_edges.get_world_point(img_center, homography)
                    real_center = find_edges.get_world_pos_from_perspective([img_center], mtx, rvec, tvec, -(cushion_height - ball_radius))[0]

                    real_center -= np.array([1, 1]) * cushion_thickness_real

                    move_count = 1
                    total_move_count = 0

                    while move_count >= 1:
                        move_count = 0

                        # Jiggle position to remove ball-round_cushion overlaps
                        for ball_id, ball in balls.items():
                            vec = circle_circle_overlap_vector(real_center, ball_radius, ball.state.rvw[0][:2], ball.params.R)
                            if not (vec==np.array([0,0])).all():
                                move_count += 1
                            real_center += vec

                        # Jiggle position to remove ball-round_cushion overlaps
                        for line_id, line_info in shot.table.cushion_segments.circular.items():
                            vec = circle_circle_overlap_vector(real_center, ball_radius, line_info.center[:2], line_info.radius)
                            if not (vec==np.array([0,0])).all():
                                move_count += 1
                            real_center += vec

                        # Jiggle position to remove ball-line_cushion overlaps
                        for line_id, line_info in shot.table.cushion_segments.linear.items():
                            vec = circle_line_overlap_vector(real_center, ball_radius, line_info.p1, line_info.p2)
                            if not (vec==np.array([0,0])).all():
                                move_count += 1
                            real_center += vec

                        # Jiggle position to ensure balls don't fall straight into pockets
                        for pocket_id, pocket_info in shot.table.pockets.items():
                            vec = circle_circle_overlap_vector(real_center, ball_radius, pocket_info.center[:2], pocket_info.radius-ball_radius+simulation_nudge)
                            if not (vec==np.array([0,0])).all():
                                move_count += 1
                            real_center += vec
                        
                        total_move_count += move_count

                        if total_move_count >= 1000:
                            # self.display_info(f"Can't place ball - can't wiggle it into a suitable place. Given up, and placed at {real_center}")
                            break

                    if real_center[0] >= 0 and \
                    real_center[1] >= 0 and \
                    real_center[0] <= table_size[0] and \
                    real_center[1] <= table_size[1]:
                        # #TODO add catches for too many cue balls etc
                        # if ball_type == "cue":
                        #     ball_id = "cue"
                        # elif ball_type == "red":
                        #     ball_id = red_id
                        #     red_id += 1
                        # elif ball_type == "yellow":
                        #     ball_id = yellow_id
                        #     yellow_id += 1
                        # elif ball_type == "8":
                        #     ball_id = 8

                        # ball_id = str(ball_id)

                        # balls[ball_id] = pt_utils.create_ball(ball_id, real_center)

                        detected_points.append((real_center + np.array([1, 1]) * cushion_thickness_real))

                method_time += time.time() - main_method_start
                one_off_time += time.time() - main_method_start

                if match_radius is None:
                    match_radius = 50.8 / 1000 # 1 ball diameter
                    match_radius = 0.13 # Ball separation distance

                self.draw(detected_points, expected_points, image_size=TABLE_SIZE, match_radius=match_radius, show=show)

                results.append(TestResults())
                results[-1].calculate_initial_metrics(detected_points, expected_points, min_table_dims, match_radius)
                results[-1].calculate_secondary_metrics()
                results[-1].eval_time = method_time
                results[-1].one_off_time = one_off_time
                if show:
                    results[-1].print_metrics()


        self.result = TestResults()
        self.result.aggregate_results(results)
        if show:
            print("Total results:")
            self.result.print_metrics()

        save_file_name = os.path.join(self.folder, f"end_to_end_results.json")

        self.result.save_to_file(save_file_name)

        print(f"Just saved to {save_file_name}")

        return self.result

    def draw(self, detected_points, expected_points, image_file=None, image_size=None, match_radius=0, show=False):
        # print(f"detected_points = {list(detected_points)}")
        # print(f"expected_points = {list(expected_points)}")
        # print(f"image_file = {image_file}")
        # print(f"match_radius = {match_radius}")

        plt.figure()

        if not image_file is None:
            img = cv2.imread(image_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.imshow(img)
        
        elif not image_size is None:
            plt.xlim([0, image_size[0]])
            plt.ylim([0, image_size[1]])
            plt.gca().set_aspect('equal', adjustable='box')

        else:
            raise Exception("Neither image_file nor image_size given")
        

        ax = plt.gca()

        for point in detected_points:
            # cv2.drawMarker(img, point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            plt.plot(*point, marker="x", markersize=10, color="b")
        for point in expected_points:
            color = (0, 1, 0)
            # cv2.drawMarker(img, point, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            plt.plot(*point, marker="x", markersize=10, color=color)
            # Draw a circle the size of match_radius
            circle = patches.Circle(point, match_radius, edgecolor=color, facecolor='none')
            ax.add_patch(circle)

        if show:
            plt.show()


def test_all(process_name):
    if process_name == "detection":
        method_names = ["hough", "hough_masked", "hough_grey_masked", "nn", "nn_masked"]
        # method_names = ["hough_masked", "hough_grey_masked"]
    elif process_name == "projection":
        method_names = ["homography", "projection"]

    results = {x: [] for x in method_names}

    for set_num in range(1, 6):
        # Get names of folders in this set
        directory = os.path.join("./validation/supervised", f"set-{set_num}")
        entries = os.listdir(directory)
        device_names = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

        for device_name in device_names:
            for method_name in method_names:
                print(f"\n\nTrying set {set_num} with device '{device_name}' and method '{method_name}'")

                try:
                    test = Test(set_num, device_name)
                except Exception as e:
                    print(e)
                    continue

                if process_name == "detection":
                    results[method_name].append(test.test_ball_detection(method_name))

                elif process_name == "projection":
                    try:
                        results[method_name].append(test.test_projection(method_name))
                    except Exception as e:
                        print(f"Test failed: {e}")
                        continue
        
        # Aggregate results
        for method_name in method_names:
            set_result = TestResults()
            set_result.aggregate_results(results[method_name])
            print("Total results:")
            set_result.print_metrics()
            set_result.save_to_file(os.path.join(directory, f"{method_name}_results.json"))

def test_blur_radius():
    method_name = "hough_masked"

    blur_radii = [x for x in range(1, 22, 2)]

    results = {x: [] for x in blur_radii}

    for set_num in [2]: #range(1, 6):
        # Get names of folders in this set
        directory = os.path.join("./validation/supervised", f"set-{set_num}")
        entries = os.listdir(directory)
        device_names = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

        for device_name in device_names:
            for blur_radius in blur_radii:
                print(f"\n\nTrying set {set_num} with device '{device_name}', method '{method_name}', blur radius {blur_radius}")

                try:
                    test = Test(set_num, device_name)
                except Exception as e:
                    print(e)
                    continue
                
                try:
                    results[blur_radius].append(test.test_ball_detection(method_name, blur_radius=blur_radius))
                except Exception as e:
                    print(f"Test failed: {e}")
                    continue
        
        # Aggregate results
        for blur_radius in blur_radii:
            set_result = TestResults()
            set_result.aggregate_results(results[blur_radius])
            print("Total results:")
            set_result.print_metrics()
            set_result.save_to_file(os.path.join(directory, f"{method_name}_rad_{blur_radius}_results.json"))

def test_hough_threshold():
    method_name = "hough_masked"

    thresholds = [x for x in range(1, 31, 1)]

    results = {x: [] for x in thresholds}

    for set_num in [2]: #range(1, 6):
        # Get names of folders in this set
        directory = os.path.join("./validation/supervised", f"set-{set_num}")
        entries = os.listdir(directory)
        device_names = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

        for device_name in device_names:
            for threshold in thresholds:
                print(f"\n\nTrying set {set_num} with device '{device_name}', method '{method_name}', threshold {threshold}")

                try:
                    test = Test(set_num, device_name)
                except Exception as e:
                    print(e)
                    continue
                
                results[threshold].append(test.test_ball_detection(method_name, hough_threshold=threshold))
        
        # Aggregate results
        for threshold in thresholds:
            set_result = TestResults()
            set_result.aggregate_results(results[threshold])
            print("Total results:")
            set_result.print_metrics()
            set_result.save_to_file(os.path.join(directory, f"{method_name}_thresh_{threshold}_results.json"))

def test_corner_detection(show=False):
    method_names = ["sam", "nn_corner"]

    results = {x: [] for x in method_names}

    set_num = 2

    # Get names of folders in this set
    directory = os.path.join("./validation/supervised", f"set-{set_num}")
    entries = os.listdir(directory)
    device_names = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

    # SAM
    test = Test(set_num, "laptop_camera")
    results["sam"].append(test.test_pocket_detection("sam", show=show))

    # nn
    for device_name in device_names:
        method_name = "nn_corner"
        
        print(f"\n\nTrying set {set_num} with device '{device_name}' and method '{method_name}'")

        try:
            test = Test(set_num, device_name)
        except Exception as e:
            print(e)
            continue

        results["nn_corner"].append(test.test_pocket_detection(method_name))

    # Aggregate results
    for method_name in method_names:
        set_result = TestResults()
        set_result.aggregate_results(results[method_name])
        print("Total results:")
        set_result.print_metrics()
        set_result.save_to_file(os.path.join(directory, f"{method_name}_results.json"))

def test_end_to_end():
    match_radii = [.1, .2, .1 , .1, .1]

    results = []

    for set_num in range(1, 6):
        # Get names of folders in this set
        directory = os.path.join("./validation/supervised", f"set-{set_num}")
        entries = os.listdir(directory)
        device_names = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

        for device_name in device_names:
            print(f"\n\nTrying set {set_num} with device '{device_name}' - end-to-end")

            try:
                test = Test(set_num, device_name)
            except Exception as e:
                print(e)
                continue

            results.append(test.test_end_to_end_detection(match_radius=match_radii[set_num - 1]))
        
        # Aggregate results
        set_result = TestResults()
        set_result.aggregate_results(results)
        print("Total results:")
        set_result.print_metrics()
        set_result.save_to_file(os.path.join(directory, f"end_to_end_results.json"))


def draw_detection_graph(metric_name):
    method_names = ["hough", "hough_masked", "nn", "nn_masked"]
    method_display_names = ['Hough', 'Hough masked', 'NN', 'NN masked']

    values = {x: [] for x in method_names}

    for set_num in range(1, 6):
        directory = os.path.join("./validation/supervised", f"set-{set_num}")

        for method_name in method_names:
            file_name = os.path.join(directory, f"{method_name}_results.json")

            result_object = TestResults()
            result = result_object.load_from_file(file_name)

            values[method_name].append(result[metric_name])
    
    draw_grouped_bar_chart(SET_NAMES, method_names, method_display_names, values, METRIC_NAME_MAP[metric_name])

def draw_greyscale_comparison_graph(metric_name):
    method_names = ["hough_masked", "hough_grey_masked"]
    method_display_names = ['Val from HSV', 'Greyscale']

    draw_single_set_graph(method_names, method_display_names, metric_name, METRIC_NAME_MAP[metric_name], set_num=2)
    
def draw_blur_radius_graph(metric_name):
    method_names = []
    method_display_names = []

    for i in [x for x in range(1, 22, 2)]:
        method_names.append(f"hough_masked_rad_{i}")
        method_display_names.append(f"{i}")

    draw_single_set_graph(method_names, method_display_names, metric_name, METRIC_NAME_MAP[metric_name], set_num=2)

def draw_hough_threshold_graph(metric_name):
    method_names = []
    method_display_names = []

    for i in [x for x in range(1, 31, 1)]:
        method_names.append(f"hough_masked_thresh_{i}")
        method_display_names.append(f"{i if i%2==0 else ' '*i}")

    draw_single_set_graph(method_names, method_display_names, metric_name, METRIC_NAME_MAP[metric_name], set_num=2)

def draw_projection_graphs():
    method_names = ["homography", "projection"]
    method_display_names = ['Homography', 'Pose estimation']

    draw_single_set_graph(method_names, method_display_names, "mean_error", "Mean error (metres)", set_num=2)
    draw_single_set_graph(method_names, method_display_names, "mean_error_normalised", "Normalised mean error (metres)", set_num=2)
    draw_single_set_graph(method_names, method_display_names, "eval_time", "Evaluation time for\n16 balls (secs)", set_num=2)

def draw_pocket_detection_graph(metric_name, set_num=2):
    method_names = ["sam", "nn_corner"]
    method_display_names = ['MetaAI\'s SAM', 'Custom neural network']

    draw_single_set_graph(method_names, method_display_names, metric_name, METRIC_NAME_MAP[metric_name], set_num=set_num)

    # values = {x: [] for x in metric_names}

    # directory = os.path.join("./validation/supervised", f"set-{set_num}")

    # for metric_name in metric_names:
    #     file_name = os.path.join(directory, f"{metric_name}_results.json")

    #     result_object = TestResults()
    #     result = result_object.load_from_file(file_name)

    #     values[metric_name].append(result[metric_name])
    
    # draw_grouped_bar_chart(SET_NAMES, method_names, method_display_names, values, METRIC_NAME_MAP[metric_name])

def draw_end_to_end_graph(metric_name):
    if metric_name == "error_normalised":
        metric_names = ["mean_error_normalised", "max_error_normalised"]
        metric_display_names = ['Normalised\nmean error', 'Normalised\nmax error']
    elif metric_name == "error":
        metric_names = ["mean_error", "max_error"]
        metric_display_names = ['Mean error', 'Max error']
    elif type(metric_name) is str:
        metric_names = [metric_name]
        metric_display_names = [METRIC_NAME_MAP[metric_name]]
    else:
        metric_names = metric_name
        metric_display_names = [METRIC_NAME_MAP[name] for name in metric_name]

    values = {x: [] for x in metric_names}

    for set_num in range(1, 6):
        directory = os.path.join("./validation/supervised", f"set-{set_num}")
        
        file_name = os.path.join(directory, f"end_to_end_results.json")

        result_object = TestResults()
        result = result_object.load_from_file(file_name)

        for _metric_name in metric_names:
            values[_metric_name].append(result[_metric_name]) # *1000 to convert from m to mm
    
    draw_grouped_bar_chart(SET_NAMES, metric_names, metric_display_names, values, METRIC_NAME_MAP[metric_name], "lower right")

def draw_detection_demo():
    detected_points = [[2806, 524], [2596, 648], [1661.8370, 654.5792], [2625.1411, 936.1063], [1958.2749, 661.2714], [2208.2310, 1300.3232], [2020.2266, 478.0728], [2271.5137, 670.5620], [2243.7183, 926.1321], [1517.1267, 901.9069], [1765.8833, 1283.6376], [2557.2925, 487.5326], [2680.3542, 1315.9834], [1309.2239, 1268.2955], [2585.1072, 674.5402], [1758.8634, 473.5060], [1882.0923, 922.1781], [2967.4453, 197.1305], [2169.4390, 355.4602]]
    expected_points = [[2282.07,  488.35], [1307.22, 1271.13], [2206.5 , 1304.28], [1516.22,  904.63], [1880.85,  920.35], [2677.22, 1322.83], [1957.15,  663.63], [1659.72,  656.13], [1763.72, 1286.13], [2621.72,  937.63], [2270.15,  673.2 ], [2581.37,  678.13], [2242.65,  928.63], [2558.35,  491.2 ], [2018.15,  480.7 ], [1756.22,  474.78]]
    image_file = "./validation/supervised\set-2\s10+_horizontal\images\p25_jpg.rf.9627e784e810a4de1eb96393907f2cc4.jpg"
    match_radius = 72.3590974610548

    dummy_test = Test(2, "s10+_horizontal")
    dummy_test.draw(detected_points, expected_points, image_file, match_radius=match_radius, show=True)


def draw_single_set_graph(method_names, method_display_names, metric_name, metric_display_name, set_num=2):
    directory = os.path.join("./validation/supervised", f"set-{set_num}")

    values = []

    for method_name in method_names:
        file_name = os.path.join(directory, f"{method_name}_results.json")
        result_object = TestResults()
        result = result_object.load_from_file(file_name)

        values.append(result[metric_name])

    # Create bar chart
    plt.figure()
    
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_powerlimits((-3,3))
    plt.gca().yaxis.set_major_formatter(yfmt)

    bars = plt.bar(method_display_names, values)

    # Add text
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.3g}", ha='center', va='bottom', fontsize=8)

    # Add labels and title
    # plt.xlabel('Projection algorithm')
    plt.ylabel(metric_display_name)

    # plt.show()

def draw_grouped_bar_chart(group_names, bar_names, bar_display_names, values, y_axis_label="", legend_location="lower right"):
    # Sample data
    # values = {
    #     'Hough': [23, 34, 45, 55, 65],
    #     'NN': [45, 56, 67, 70, 80],
    #     'NN masked': [10, 20, 30, 40, 50],
    #     'Other': [56, 78, 89, 90, 100],
    # }

    plt.figure()
    
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_powerlimits((-3,3))
    plt.gca().yaxis.set_major_formatter(yfmt)

    # Set the width of the bars
    bar_width = 1/(len(bar_names) + 1)

    # Set the positions of the bars on the x-axis
    x_positions = []
    x_positions.append(np.arange(len(group_names)))
    for i in range(len(bar_names)):
        x_positions.append([x + bar_width for x in x_positions[-1]])

    # Create the bars
    bar_sets = []
    for i, method_name in enumerate(bar_names):
        bar_sets.append(plt.bar(x_positions[i], values[method_name], width=bar_width * .8, edgecolor='grey', label=bar_display_names[i]))

    # Add values above the bars
    for bars in bar_sets:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.3g}", ha='center', va='bottom', fontsize=6)

    # Add labels and title
    # plt.xlabel('Test sets', fontweight='bold')
    plt.ylabel(y_axis_label) #, fontweight='bold')
    plt.xticks([r + bar_width*(len(bar_names)-1)/2 for r in range(len(group_names))], group_names)

    # Add legend
    if len(bar_names) != 1:
        plt.legend(loc=legend_location)

    # Show the chart
    # plt.show()


if __name__ == "__main__":
    start_time = time.time()
    # test = Test(2, "laptop_camera")
    # test.test_ball_detection(method_name="hough", show=True)

    # test = Test(2, "s10+_horizontal")
    # test.test_projection("projection", show=True)

    test = Test(2, "laptop_camera")
    test.test_pocket_detection("sam", show=True)

    # test = Test(2, "s10+_horizontal")
    # test.test_ball_detection("hough_masked", blur_radius=10, show=True)

    # test = Test(4, "s10+_vertical")
    # test.test_end_to_end_detection(match_radius=.1, show=True)

    # test_all("detection")
    # test_blur_radius()
    # test_hough_threshold()
    # test_all("projection")
    # test_corner_detection(show=True)
    # test_end_to_end()
    # test_end_to_end()

    plt.close('all')
    elapsed_time = time.time() - start_time
    print(f"Executed all tests in {elapsed_time} secs (={elapsed_time/60} mins)")

    # draw_detection_graph("precision")
    # draw_detection_graph("recall")
    # draw_detection_graph("accuracy")
    # draw_detection_graph("f1_score")
    # draw_detection_graph("mean_error_normalised")
    # draw_detection_graph("eval_time")
    # draw_greyscale_comparison_graph("f1_score")
    # draw_blur_radius_graph("f1_score")
    # draw_hough_threshold_graph("f1_score")
    # draw_projection_graphs()
    # draw_pocket_detection_graph("mean_error_normalised")
    # draw_pocket_detection_graph("f1_score")
    # draw_pocket_detection_graph("eval_time")
    # draw_pocket_detection_graph("one_off_time")
    draw_end_to_end_graph("mean_error")
    draw_end_to_end_graph("f1_score")
    draw_end_to_end_graph("eval_time")
    # draw_detection_demo()

    plt.show()
