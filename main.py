import find_edges
import sam
import numpy as np
import matplotlib.pyplot as plt


sam.create_mask(
    image_file="images\\snooker1.png",
    input_points = np.array([[600, 600], [1300, 600], [1625, 855]]),
    input_labels = np.array([1, 1, 0])) # 1=foreground, 0=background
find_edges.get_edges("images\\snooker1.png")

plt.show()