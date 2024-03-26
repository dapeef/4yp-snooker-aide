import numpy as np
import os



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

def run_test(set, device):
    # set = [1, 2, 3, 4, 5]
    # device = ["laptop_camera", "s10+_vertical", "s10+_horizontal", "logitech_camera"]

    pass

if __name__ == "__main__":
    points = load_points_from_file()