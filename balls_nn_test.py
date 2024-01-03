import matplotlib.pyplot as plt
import balls_eval

# image_file = "data/Balls - 4792 - above, toy table, stripey balls, very size-sensitive (augment)/train/images/Ball2-51-_png.rf.6b0ee4e1dde8b1a71b5ac54a391a9b7e.jpg"
# image_file = "images/terrace.jpg"
image_file = "images/home_table4.jpg"
# image_file = "images/snooker2.jpg"

img_balls = balls_eval.get_balls(image_file)
plt.show()