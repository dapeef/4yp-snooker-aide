from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QWidget
from PyQt5 import uic
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF, QTimer, QStringListModel
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QPolygon, QPolygonF, QPixmap, QImage
import sys
import os
import pooltool as pt
from pooltool.ani.camera import camera_states
from pooltool.ani.image import HDF5Images, ImageZip, NpyImages, image_stack
from pooltool.ani.image.interface import FrameStepper
import math
import pooltool_test as pt_utils
import time
import nn_utils
import find_edges
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.colors as mcolors
from PyQt5.QtWidgets import QVBoxLayout
import cv2
from pygrabber.dshow_graph import FilterGraph
import calibrate_camera


class Calibration(QMainWindow):
    def __init__(self, cap, output_folder, completion_function):
        super().__init__()

        # Load UI
        uic.loadUi("calibration.ui", self)

        # Receive capture object from main window
        self.cap = cap
        self.device = os.path.basename(output_folder)
        self.output_folder = output_folder
        self.completion_function = completion_function

        # Other variables
        self.temp_folder = "./temp/calibration"
        self.num_images = 0
        self.max_images = 60

        # Initialise camera refresh
        self.camera_refresh_timer = QTimer(self)
        self.camera_refresh_timer.timeout.connect(self.camera_update)
        self.camera_refresh_timer.start(int(1000/30)) # milliseconds (30fps)

        # Initialise capture refresh
        self.capture_refresh_timer = QTimer(self)
        self.capture_refresh_timer.timeout.connect(self.capture_update)
        self.capture_refresh_timer.start(1000) # milliseconds (2fps)

        # Link button
        self.start_button.clicked.connect(self.start_clicked)
        self.calibrating = False
        
        # Save initial label size
        self.label_size = [1000, 563]

    def get_camera_image(self):
        ret, img = self.cap.read()

        if ret:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        else:
            print("!!! Calibration camera error!")
    def show_image(self, img):
        img = cv2.flip(img, 1) # Flip left to right
        height, width, channels = img.shape
        bytesPerLine = 3 * width
        image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(image)
        pixmap = pixmap.scaled(*self.label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_label.setPixmap(pixmap)

    def camera_update(self):
        # if not self.calibrating:
        img = self.get_camera_image()

        if not img is None:
            self.show_image(img)
    def capture_update(self):
        if self.calibrating:
            img = self.get_camera_image()

            if not img is None:
                # self.show_image(img)

                file_name = os.path.join(self.temp_folder, f"{time.time()}.jpg")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_name, img)

                self.num_images += 1
                self.progress_bar.setValue(int(self.num_images / self.max_images * 100))

                if self.num_images >= self.max_images:
                    print("BOSH, we're done here")
                    self.image_capture_complete()

    def empty_temp_folder(self):
        # Clear calibration folder
        for file_name in os.listdir(self.temp_folder):
            file_path = os.path.join(self.temp_folder, file_name)
            os.remove(file_path)

    def image_capture_complete(self):
        # Stop timers
        self.camera_refresh_timer.stop()
        self.capture_refresh_timer.stop()

        square_size = float(self.square_size_text.text())

        self.image_label.setText("Calibrating...")
        calibrate_camera.checkerboard_calibrate(self.temp_folder, self.output_folder, square_size=square_size, show=True)
        self.image_label.setText("Calibration complete")

        self.completion_function(self.device)

        self.close()

    def start_clicked(self):
        if not self.calibrating:
            # Start
            self.empty_temp_folder()

            self.start_button.setText("Cancel")
            self.calibrating = True
            
        else:
            # Cancel
            self.start_button.setText("Start")
            self.calibrating = False

            self.progress_bar.setValue(0)
            self.num_images = 0

            self.empty_temp_folder()

    def closeEvent(self, *args, **kwargs):
        super(QMainWindow, self).closeEvent(*args, **kwargs)
        
        # Stop timers
        self.camera_refresh_timer.stop()
        self.capture_refresh_timer.stop()

        self.empty_temp_folder()


class Ui(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI
        uic.loadUi("gui.ui", self)

        # Initialise visualisation parameters
        self.time = 0
        self.V0_max = 7 # m/s
        self.phi_max = 360
        self.theta_max = 90
        self.a_max = 1
        self.b_max = 1

        # Create shot object
        self.create_shot(balls=pt_utils.get_example_balls())
        # self.shot.strike(V0=self.V0, phi=self.phi, a=self.spin_side, b=self.spin_top, theta=self.theta)
        self.update_shot(
            V0=0.8,
            phi=274,
            a=0,
            b=-0.7,
            theta=0,
        )

        # Set final canvas_widget values
        self.canvas_table_width = 1000
        self.canvas_table_height = self.canvas_table_width / self.shot.table.l * self.shot.table.w
        self.canvas_padding = int(self.canvas_table_width / 10)
        self.canvas_widget.setFixedWidth(int(self.canvas_table_width + 2*self.canvas_padding))
        self.canvas_widget.setFixedHeight(int(self.canvas_table_height + 2*self.canvas_padding))
        self.canvas_widget.paintEvent = self.draw_table_canvas
        self.canvas_widget.mousePressEvent = self.canvas_widget_mousedown
        self.canvas_widget.mouseMoveEvent = self.canvas_widget_click
        self.canvas_widget.mouseReleaseEvent = self.canvas_widget_mouseup
        self.is_dragging_canvas = False

        # Link image button events to functions
        self.update_pockets_checkbox.stateChanged.connect(self.update_pockets_checkbox_changed)
        self.auto_refresh_checkbox.stateChanged.connect(self.auto_refresh_checkbox_changed)
        self.auto_refresh_last_image = None
        self.image_name.returnPressed.connect(self.load_image_clicked)
        self.load_image_button.clicked.connect(self.load_image_clicked)
        self.load_webcam_button.clicked.connect(self.load_webcam_clicked)
        self.render_shot_button.clicked.connect(self.render_shot)

        # Temp file for the webcam image
        self.temp_webcam_file_name = "./temp/webcam_image.jpg"

        # Initialise settings
        self.calibration_directory = "./calibration"
        devices = os.listdir(self.calibration_directory)
        calibration_options = [device for device in devices if os.path.isdir(os.path.join(self.calibration_directory, device))]
        self.image_cal_drop_down.addItems(calibration_options)
        self.image_cal_drop_down.setCurrentText("s10+_horizontal")
        self.webcam_cal_drop_down.addItems(calibration_options)
        self.webcam_cal_drop_down.setCurrentText("logitech_gui")
        self.new_calibration_button.clicked.connect(self.new_calibration_clicked)
        self.game_type_drop_down.addItems(["English 8 ball"])
        self.game_type_drop_down.setCurrentText("English 8 ball")

        # Link time button events to functions
        self.time_slider.valueChanged.connect(
            lambda value: self.update_time(set_time=value/self.time_slider.maximum(), slider_set=True))
        self.plus001.clicked.connect(lambda: self.update_time(rel_time=0.01))
        self.plus01.clicked.connect(lambda: self.update_time(rel_time=0.1))
        self.plus1.clicked.connect(lambda: self.update_time(rel_time=1))
        self.minus001.clicked.connect(lambda: self.update_time(rel_time=-0.01))
        self.minus01.clicked.connect(lambda: self.update_time(rel_time=-0.1))
        self.minus1.clicked.connect(lambda: self.update_time(rel_time=-1))
        self.start_time_button.clicked.connect(lambda: self.update_time(set_time=0))
        self.end_time_button.clicked.connect(lambda: self.update_time(set_time=1))
        self.play_button.clicked.connect(self.toggle_play_animation)

        # Link shot button events to functions
        self.V0_slider.valueChanged.connect(
            lambda value: self.update_shot(V0=value/self.V0_slider.maximum()*self.V0_max))
        self.phi_slider.valueChanged.connect(
            lambda value: self.update_shot(phi=value/self.phi_slider.maximum()*self.phi_max))
        self.phi_plus001.clicked.connect(lambda: self.update_shot(phi=self.shot.cue.phi + 0.01))
        self.phi_plus01.clicked.connect(lambda: self.update_shot(phi=self.shot.cue.phi + 0.1))
        self.phi_plus1.clicked.connect(lambda: self.update_shot(phi=self.shot.cue.phi + 1))
        self.phi_minus001.clicked.connect(lambda: self.update_shot(phi=self.shot.cue.phi - 0.01))
        self.phi_minus01.clicked.connect(lambda: self.update_shot(phi=self.shot.cue.phi - 0.1))
        self.phi_minus1.clicked.connect(lambda: self.update_shot(phi=self.shot.cue.phi - 1))
        self.theta_slider.valueChanged.connect(
            lambda value: self.update_shot(theta=value/self.theta_slider.maximum()*self.theta_max))
        self.a_slider.valueChanged.connect(
            lambda value: self.update_shot(a=-value/self.a_slider.maximum()*self.a_max))
        self.b_slider.valueChanged.connect(
            lambda value: self.update_shot(b=value/self.b_slider.maximum()*self.b_max))
        self.center_button.clicked.connect(lambda: self.update_shot(a=0, b=0))

        # Link spin widget click
        self.spin_canvas_widget.paintEvent = self.draw_spin_canvas
        self.spin_canvas_widget.mousePressEvent = self.spin_widget_mousedown
        self.spin_canvas_widget.mouseMoveEvent = self.spin_widget_click
        self.spin_canvas_widget.mouseReleaseEvent = self.spin_widget_mouseup
        self.is_dragging_spin = False

        # Setup play function calls
        self.playing = False
        self.play_start_time = time.time()
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_update)
        self.play_timer.start(16) # milliseconds

        # Set up auto refresh polling
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.auto_refresh)
        self.auto_refresh_interval = 1000 # milliseconds
        self.refresh_timer.start(self.auto_refresh_interval)

        # Initialise info display output
        self.string_list = []
        self.string_model = QStringListModel()
        self.string_model.setStringList(self.string_list)
        self.info_listview.setModel(self.string_model)

        # Set some colour values
        self.color_table = QColor("#1ea625")
        self.color_cushion = QColor("#0b5e0f")
        self.color_top = QColor("#5e400b")
        self.color_pocket = QColor("#000000")
        self.color_path = [ # Path colours for different ball states
            QColor("black"), # stationary = 0
            QColor("blue"),  # spinning = 1
            QColor("#b3b3b3"),   # sliding = 2
            QColor("white"), # rolling = 3
            QColor("green")  # pocketed = 4
        ]
        self.color_ball = {
            "pooltool_pocket": {
                "cue": QColor("white"),
                "1": QColor("red"),
                "2": QColor("red"),
                "3": QColor("red"),
                "4": QColor("red"),
                "5": QColor("red"),
                "6": QColor("red"),
                "7": QColor("red"),
                "8": QColor("black"),
                "9": QColor("yellow"),
                "10": QColor("yellow"),
                "11": QColor("yellow"),
                "12": QColor("yellow"),
                "13": QColor("yellow"),
                "14": QColor("yellow"),
                "15": QColor("yellow"),
            },
            "generic_snooker": {
                "cue": QColor("white"),
                "red_01": QColor("red"),
                "red_02": QColor("red"),
                "red_03": QColor("red"),
                "red_04": QColor("red"),
                "red_05": QColor("red"),
                "red_06": QColor("red"),
                "red_07": QColor("red"),
                "red_08": QColor("red"),
                "red_09": QColor("red"),
                "red_10": QColor("red"),
                "red_11": QColor("red"),
                "red_12": QColor("red"),
                "red_13": QColor("red"),
                "red_14": QColor("red"),
                "red_15": QColor("red"),
                "yellow": QColor("yellow"),
                "green": QColor("green"),
                "brown": QColor("#5e400b"),
                "blue": QColor("blue"),
                "pink": QColor("#FFC0CB"),
                "black": QColor("black")
            }
        }

        # Define cushion polarity
        # 0 draws a cushion to the right of the defining line, 1 to the left
        # Clockwise, starting with the left-most side of the top-left pocket sides
        self.cushion_polarity = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]

        # How much to nudge balls by when fixing overlaps
        self.simulation_nudge = 0.0001

        # Calculate top and cushion thickness
        self.top_thickness = self.canvas_widget.height()
        self.cushion_thickness = 0
        self.cushion_height = 0.037 # mm - height from playing surface to top of cushion

        for line_id, line_info in self.shot.table.cushion_segments.linear.items():
            # Scale the points to fit within the widget dimensions
            start = self.transform_point(line_info.p1[:2])
            end = self.transform_point(line_info.p2[:2])

            # Save lowest x value - this is to draw the top
            self.top_thickness = min([self.top_thickness, *start, *end])
            self.cushion_thickness = self.canvas_padding - self.top_thickness

        # Cushion width in meters
        self.cushion_thickness_real = pt_utils.english_8_ball_table_specs().cushion_width

        # Do initial update of the GUI
        self.update_time(set_time=0)


        # Initialise NN models
        self.pocket_evaluator = nn_utils.EvaluateNet("./checkpoints/pockets_model.pth", 2)
        self.balls_evaluator = nn_utils.EvaluateNet("./checkpoints/balls_model_multiple.pth", 20)

        self.balls_class_conversion = ['1', '10', '11', '12', '13', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9', 'cue', 'rack', 'red', 'yellow']


        # Initialise camera
        self.get_available_cameras()
        self.current_camera_index = min(1, len(self.available_cameras))
        # self.webcam_dro_down.clear()
        self.webcam_drop_down.addItems(self.available_cameras)
        self.webcam_drop_down.setCurrentIndex(self.current_camera_index)
        self.webcam_drop_down.currentIndexChanged.connect(self.change_camera)
        self.change_camera(self.current_camera_index)

        # Start an instance of the pooltool ShotViewer (shows the GUI)
        # self.interface = pt.ShotViewer()

        # Set up from initial image
        self.load_image_clicked()


    def new_calibration_clicked(self):
        output_folder = os.path.join(self.calibration_directory, self.new_calibration_name.text())
        self.calibration_window = Calibration(self.cap, output_folder, self.refresh_calibration_drop_down)
        self.calibration_window.show()
    def refresh_calibration_drop_down(self, value):
        self.image_cal_drop_down.addItem(value)
        # self.image_cal_drop_down.setCurrentText("s10+_horizontal")
        self.webcam_cal_drop_down.addItem(value)
        self.webcam_cal_drop_down.setCurrentText(value)

    def update_pockets_checkbox_changed(self):
        if self.update_pockets_checkbox.isChecked():
            self.auto_refresh_checkbox.setChecked(False)

        self.auto_refresh_checkbox.setEnabled(not self.update_pockets_checkbox.isChecked())
    def auto_refresh_checkbox_changed(self):
        if self.auto_refresh_checkbox.isChecked():
            self.auto_refresh_last_image = None
            self.load_webcam_clicked()
    def auto_refresh(self):
        if self.auto_refresh_checkbox.isChecked():            
            # Capture image
            ret, img = self.cap.read()
            if ret:
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if not self.auto_refresh_last_image is None:
                    if find_edges.check_ball_movement(self.auto_refresh_last_image, img, self.balls_target, threshold=30, proportion_threshold=0.5):
                        self.display_info("Change detected; automatically refreshing...")
                        # Something's moved, so refresh
                        cv2.imwrite(self.temp_webcam_file_name, img)

                        self.process_image(self.temp_webcam_file_name, self.webcam_cal_drop_down.currentText())
                
                self.auto_refresh_last_image = img
            else:
                self.display_info("Error reading from webcam")


    def get_pockets(self, image_file):
        self.pocket_evaluator.create_dataset(image_file)
        target = self.pocket_evaluator.get_boxes(0)

        target = nn_utils.filter_boxes(target, confidence_threshold=.1)
        target = nn_utils.get_bbox_centers(target)

        return target
    def get_balls(self, image_file):
        self.balls_evaluator.create_dataset(image_file)
        target = self.balls_evaluator.get_boxes(0)

        target = nn_utils.filter_boxes(target, max_results=100, confidence_threshold=0.5, remove_overlaps=False)
        target = nn_utils.get_bbox_centers(target)

        return target

    def process_image(self, image_file, calibration_folder):
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

        self.display_info("Processing image")

        # Get camera parameters
        mtx = np.load(os.path.join(self.calibration_directory, calibration_folder, "intrinsic_matrix.npy"))
        dist_coeffs = np.load(os.path.join(self.calibration_directory, calibration_folder, "distortion.npy"))
        
        # Undistort images
        undistorted_image_file = "./temp/undistorted.png"
        img = cv2.imread(image_file)
        img = cv2.undistort(img, mtx, dist_coeffs)
        cv2.imwrite(undistorted_image_file, img)
        image_file = undistorted_image_file

        # Evaluate NNs
        if self.update_pockets_checkbox.isChecked():
            self.pockets_target = self.get_pockets(image_file)
        else:
            self.pockets_target = self.old_pockets_target

        self.balls_target = self.get_balls(image_file)

        
        # Draw NN output for the pockets
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.plot_img_bbox(self.pockets_widget, img, self.pockets_target, "NN pocket locations")
        # And for the balls
        self.plot_img_bbox(self.balls_widget, img, self.balls_target, "NN ball locations")

        # Check that at least one cue ball has been detected
        cue_ball_exists = False
        for i in range(len(self.balls_target["labels"])):
            label = self.balls_target["labels"][i]
            ball_type = self.balls_class_conversion[label]
            if ball_type == "cue":
                cue_ball_exists = True

        if cue_ball_exists:
            try:
                # Get corner points from pocket output
                pocket_lines, pocket_mask, max_dist = find_edges.get_lines_from_pockets(image_file, self.pockets_target)
                corners = find_edges.get_rect_corners(pocket_lines)

                # Get homography between pixelspace and tablespace
                table_size = [self.shot.table.w + 2*self.cushion_thickness_real, self.shot.table.l + 2*self.cushion_thickness_real]
                # print(f"Table size: {table_size}")

                rvec, tvec, projection = find_edges.get_perspective(corners, table_size, mtx)



                # Process all balls
                # Inflate ball radius slightly so it moves the balls far enough away from cushions
                ball_radius = pt_utils.english_8_ball_ball_params().R + self.simulation_nudge

                balls = {}

                red_id = 1
                yellow_id = 9

                for i in range(len(self.balls_target["labels"])):
                    label = self.balls_target["labels"][i]
                    ball_type = self.balls_class_conversion[label]
                    img_center = self.balls_target["centers"][i]
                    real_center = find_edges.get_world_pos_from_perspective([img_center], mtx, rvec, tvec, -(self.cushion_height - ball_radius))[0]

                    real_center -= np.array([1, 1]) * self.cushion_thickness_real

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
                        for line_id, line_info in self.shot.table.cushion_segments.circular.items():
                            vec = circle_circle_overlap_vector(real_center, ball_radius, line_info.center[:2], line_info.radius)
                            if not (vec==np.array([0,0])).all():
                                move_count += 1
                            real_center += vec

                        # Jiggle position to remove ball-line_cushion overlaps
                        for line_id, line_info in self.shot.table.cushion_segments.linear.items():
                            vec = circle_line_overlap_vector(real_center, ball_radius, line_info.p1, line_info.p2)
                            if not (vec==np.array([0,0])).all():
                                move_count += 1
                            real_center += vec

                        # Jiggle position to ensure balls don't fall straight into pockets
                        for pocket_id, pocket_info in self.shot.table.pockets.items():
                            vec = circle_circle_overlap_vector(real_center, ball_radius, pocket_info.center[:2], pocket_info.radius-ball_radius+self.simulation_nudge)
                            if not (vec==np.array([0,0])).all():
                                move_count += 1
                            real_center += vec
                        
                        total_move_count += move_count

                        if total_move_count >= 1000:
                            self.display_info(f"Can't place ball - can't wiggle it into a suitable place. Given up, and placed at {real_center}")
                            break

                    if real_center[0] >= 0 and \
                    real_center[1] >= 0 and \
                    real_center[0] <= table_size[0] and \
                    real_center[1] <= table_size[1]:

                        #TODO add catches for too many cue balls etc
                        if ball_type == "cue":
                            ball_id = "cue"
                        elif ball_type == "red":
                            ball_id = red_id
                            red_id += 1
                        elif ball_type == "yellow":
                            ball_id = yellow_id
                            yellow_id += 1
                        elif ball_type == "8":
                            ball_id = 8

                        ball_id = str(ball_id)

                        balls[ball_id] = pt_utils.create_ball(ball_id, real_center)

                self.shot.balls = balls
                self.update_shot()

                self.update_time(set_time=0)

                self.old_pockets_target = self.pockets_target

            except AssertionError as e:
                print(e)
                self.display_info(f"Failed to process pockets: {e}")
        else:
            self.display_info("No cue ball found in image")

    def load_image_clicked(self):
        # Load image
        image_file = self.image_name.text()
        if os.path.exists(image_file):
            self.display_info(f"Loading image from image: {image_file}")
            self.process_image(image_file, self.image_cal_drop_down.currentText())
        else:
            self.display_info(f"Invalid file name: {image_file}")
    def load_webcam_clicked(self):
        self.display_info(f"Loading image from webcam: {self.available_cameras[self.current_camera_index]}")
        # Save webcam image
        # Read twice to make sure the camera is up to date
        ret, img = self.cap.read()
        ret, img = self.cap.read()

        if ret:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.temp_webcam_file_name, img)

        self.process_image(self.temp_webcam_file_name, self.webcam_cal_drop_down.currentText())

    def get_available_cameras(self):
        self.available_cameras = FilterGraph().get_input_devices()
    def change_camera(self, new_camera_index):
        self.current_camera_index = new_camera_index

        try:
            self.cap.release()
        except AttributeError:
            pass # First time running

        self.display_info(f"Initialising camera {self.available_cameras[self.current_camera_index]} at index {self.current_camera_index}. This may take a while")
        self.cap = cv2.VideoCapture(self.current_camera_index)
        self.display_info(f"Camera initialised: {self.available_cameras[self.current_camera_index]}")
        self.cap.read()


    def plot_img_bbox(self, widget, img, target, title=""):
        # Clear the layout of the widget
        layout = widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                widget.layout().removeItem(item)
                widget.layout().removeWidget(item.widget())
        else:
            layout = QVBoxLayout()

        colors = ["silver", "blue", "green", "gray", "orange", "purple", "pink", "brown", "navy",
                  "cyan", "magenta", "lime", "teal", "black", "maroon", "white", "olive", "red", "yellow"]

        figure = plt.figure()
        ax = figure.add_subplot(111)

        ax.set_title(title)
        ax.imshow(img)

        for i, box in enumerate(target['boxes']):
            color = colors[target["labels"][i]]

            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            x_mid, y_mid = x + width / 2, y + height / 2

            rect = patches.Rectangle(
                (x, y),
                width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            # If confidence scores exist, display these as text
            if "scores" in target:
                # invert colour
                rgb = mcolors.to_rgb(color)
                opposite_color = tuple(1.0 - val for val in rgb)

                plt.text(x_mid, y_mid,
                        round(float(target["scores"][i]), 3),
                        ha="center", # text alignment
                        va="center",
                        color=opposite_color,
                        fontsize=6
                )

        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, widget)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        widget.setLayout(layout)
        canvas.draw()

        plt.tight_layout()


    def update_time(self, set_time=None, rel_time=None, slider_set=False):
        if not set_time is None:
            self.time = set_time * self.shot.t
        elif not rel_time is None:
            self.time += rel_time

        self.time = min(max(self.time, 0), self.shot.t)
        
        if self.shot.t != 0:
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(int(self.time / self.shot.t * self.time_slider.maximum()))
            self.time_slider.blockSignals(False)

        self.time_label.setText("Shot time: {:.2f}s".format(self.time))

        self.canvas_widget.update()
    def toggle_play_animation(self, event):
        if not self.playing:
            self.playing = True
            self.play_start_time = time.time() - self.time

            self.disable_enable_all(False)
            self.play_button.setEnabled(True)
            self.play_button.setText("Pause")
        
        else:
            self.playing = False
            
            self.disable_enable_all(True)
            self.play_button.setEnabled(True)
            self.play_button.setText("Play")
    def play_update(self):
        if self.playing:
            self.time = (time.time() - self.play_start_time) % self.shot.t
            self.update_time()


    def display_info(self, string):
        print(string)
        
        self.string_list.append(string)
        self.string_model.setStringList(self.string_list)

        self.info_listview.scrollToBottom()


    def disable_enable_all(self, enabled):
        self.image_name.setEnabled(enabled)
        self.load_image_button.setEnabled(enabled)
        self.load_webcam_button.setEnabled(enabled)

        self.time_slider.setEnabled(enabled)
        self.plus001.setEnabled(enabled)
        self.plus01.setEnabled(enabled)
        self.plus1.setEnabled(enabled)
        self.minus001.setEnabled(enabled)
        self.minus01.setEnabled(enabled)
        self.minus1.setEnabled(enabled)
        self.start_time_button.setEnabled(enabled)
        self.end_time_button.setEnabled(enabled)
        self.play_button.setEnabled(enabled)

        self.V0_slider.setEnabled(enabled)
        self.phi_slider.setEnabled(enabled)
        self.phi_plus001.setEnabled(enabled)
        self.phi_plus01.setEnabled(enabled)
        self.phi_plus1.setEnabled(enabled)
        self.phi_minus001.setEnabled(enabled)
        self.phi_minus01.setEnabled(enabled)
        self.phi_minus1.setEnabled(enabled)
        self.theta_slider.setEnabled(enabled)
        self.a_slider.setEnabled(enabled)
        self.b_slider.setEnabled(enabled)
        self.center_button.setEnabled(enabled)


    def create_shot(self, balls):
        table = pt.Table.from_table_specs(pt_utils.english_8_ball_table_specs())
        cue = pt.Cue(cue_ball_id="cue")

        self.shot = pt.System(table=table, balls=balls, cue=cue)

    def spin_widget_mousedown(self, event):
        if not self.playing:
            self.is_dragging_spin = True
            self.spin_widget_click(event)
    def spin_widget_mouseup(self, event):
        self.is_dragging_spin = False
    def spin_widget_click(self, event):
        if self.is_dragging_spin:
            # Update the point coordinates based on mouse click
            circle_radius = min(self.spin_canvas_widget.width(), self.spin_canvas_widget.height()) / 2
            circle_center = self.spin_canvas_widget.rect().center()
            a = -(event.x() - circle_center.x()) / circle_radius
            b = -(event.y() - circle_center.y()) / circle_radius
            self.update_shot(a=a, b=b)

    def canvas_widget_mousedown(self, event):
        if not self.playing:
            self.is_dragging_canvas = True
            self.canvas_widget_click(event)
    def canvas_widget_mouseup(self, event):
        self.is_dragging_canvas = False
    def canvas_widget_click(self, event):
        if self.is_dragging_canvas:
            real_cue_pos = self.shot.balls["cue"].history_cts.states[0].rvw[0][:2]
            cue_ball_pos = self.transform_point(real_cue_pos)

            dx = event.x() - cue_ball_pos[0]
            dy = event.y() - cue_ball_pos[1]

            phi = np.arctan2(-dx, -dy) * 180/np.pi + 180

            self.update_shot(phi=phi)

    def update_shot(self, V0=None, phi=None, a=None, b=None, theta=None):
        # if not V0 is None:
        #     self.V0 = V0
        #     self.shot.cue.V0 = V0
        # if not phi is None:
        #     self.phi = phi
        # if not a is None:
        #     self.spin_side = a
        # if not b is None:
        #     self.spin_top = b
        # if not theta is None:
        #     self.theta = theta
        # Reset simulation


        self.shot.reset_balls()
        self.shot.reset_history()

        # Clamp spin values
        if not a is None:
            a = min(1, max(-1, a))
        if not b is None:
            b = min(1, max(-1, b))


        # Update simulation parameters
        self.shot.strike(V0=V0, phi=phi, a=a, b=b, theta=theta)

        #Update text
        self.V0_label.setText("Shot speed: {:.1f}m/s".format(self.shot.cue.V0))
        self.phi_label.setText("Angle: {:.1f}°".format(self.shot.cue.phi))
        self.theta_label.setText("Elevation: {:.1f}°".format(self.shot.cue.theta))

        # Update sliders
        self.V0_slider.blockSignals(True)
        self.V0_slider.setValue(int(self.shot.cue.V0 / self.V0_max * self.V0_slider.maximum()))
        self.V0_slider.blockSignals(False)
        self.phi_slider.blockSignals(True)
        self.phi_slider.setValue(int(self.shot.cue.phi / self.phi_max * self.phi_slider.maximum()))
        self.phi_slider.blockSignals(False)
        self.a_slider.blockSignals(True)
        self.a_slider.setValue(int(-self.shot.cue.a / self.a_max * self.a_slider.maximum()))
        self.a_slider.blockSignals(False)
        self.b_slider.blockSignals(True)
        self.b_slider.setValue(int(self.shot.cue.b / self.b_max * self.b_slider.maximum()))
        self.b_slider.blockSignals(False)
        self.theta_slider.blockSignals(True)
        self.theta_slider.setValue(int(self.shot.cue.theta / self.theta_max * self.theta_slider.maximum()))
        self.theta_slider.blockSignals(False)

        # Update spin widget
        self.spin_canvas_widget.update()


        self.recalculate_shot()
        self.update_time(set_time=0)

    def recalculate_shot(self):
        # Evolve the shot
        self.shot = pt.simulate(self.shot)
        # Continuize the shot
        self.shot = pt.continuize(self.shot)

        self.canvas_widget.update()

    def _render_shot(self):
        stepper = FrameStepper()

        # Make an dump dir
        path = "./temp"

        # These camera states can be found in pooltool/ani/camera/camera_states. You can
        # make your own by creating a new JSON in that directory. Reach out if you want to
        # create a camera state from within the interactive interface (this is also
        # possible).
        for camera_state in [
            "7_foot_overhead",
            "7_foot_offcenter",
        ]:
            exporter = ImageZip(os.path.join(path, f"{camera_state}.zip"), ext="png")

            imgs = image_stack(
                system=self.shot,
                interface=stepper,
                size=(360 * 1.6, 360),
                fps=10,
                camera_state=camera_states[camera_state],
                show_hud=False,
                gray=False,
            )
            exporter.save(imgs)

            # Verify the images can be read back
            read_from_disk = exporter.read(exporter.path)
            assert np.array_equal(imgs, read_from_disk)
    def render_shot(self):
        # Open up the shot in the GUI
        self.interface.show(self.shot)

    def transform_distanceF(self, real_distance):
        table_width = self.shot.table.w
        table_length = self.shot.table.l

        return real_distance / table_length * self.canvas_table_width
    def transform_distance(self, real_distance):
        return int(self.transform_distanceF(real_distance))
    
    def transform_pointF(self, real_xy):
        return (
            self.transform_distanceF(real_xy[1]) + self.canvas_padding,
            self.transform_distanceF(real_xy[0]) + self.canvas_padding
        )
    def transform_point(self, real_xy):
        point = self.transform_pointF(real_xy)
        return (
            int(point[0]),
            int(point[1])
        )

    def draw_table_canvas(self, event):
        painter = QPainter(self.canvas_widget)
        pen = QPen()
        painter.setRenderHint(QPainter.Antialiasing)  # Optional: Enable antialiasing for smoother circle
        painter.fillRect(self.rect(), self.color_table) # Set background colour
        painter.setBrush(QBrush(Qt.SolidPattern))  # Set the brush style

        # Cushions
        pen.setColor(self.color_cushion)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.color_cushion))  # Set the brush color for the cushions

        # Cushion lines
        for line_id, line_info in self.shot.table.cushion_segments.linear.items():
            # Scale the points to fit within the widget dimensions
            start = self.transform_point(line_info.p1[:2])
            end = self.transform_point(line_info.p2[:2])

            width = int(math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2))
            vector = (end[0] - start[0], end[1] - start[1])

            # Calculate the angle of the line from the x-axis
            angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) + self.cushion_polarity[int(line_info.id)-1] * 180

            painter.save()
            # Move the painter to the center of the rectangle
            painter.translate(start[0] + vector[0] / 2, start[1] + vector[1] / 2)

            # Rotate the painter
            painter.rotate(angle)

            # Draw the rotated rectangle with thickness
            painter.drawRect(int(-width / 2), 0, width, self.cushion_thickness)

            painter.restore()

        # Cushion circles
        for line_id, line_info in self.shot.table.cushion_segments.circular.items():
            # Scale the points to fit within the widget dimensions
            center = self.transform_point(line_info.center[:2])

            radius = self.transform_distance(line_info.radius)
            painter.setBrush(QBrush(self.color_cushion))  # Set the brush color for the circle
            # painter.setBrush(Qt.NoBrush)  # Set the brush color for the circle
            painter.drawEllipse(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)

        
        # Top
        # Draw rectangles forming a frame with padding
        frame_rect_top = QRect().adjusted(0, 0, self.canvas_widget.width(), self.top_thickness)
        frame_rect_left = QRect().adjusted(0, 0, self.top_thickness, self.canvas_widget.height())
        frame_rect_right = QRect().adjusted(self.canvas_widget.width() - self.top_thickness, 0, self.canvas_widget.width(), self.canvas_widget.height())
        frame_rect_bottom = QRect().adjusted(0, self.canvas_widget.height() - self.top_thickness, self.canvas_widget.width(), self.canvas_widget.height())

        painter.fillRect(frame_rect_top, self.color_top)
        painter.fillRect(frame_rect_left, self.color_top)
        painter.fillRect(frame_rect_right, self.color_top)
        painter.fillRect(frame_rect_bottom, self.color_top)


        # Pockets
        for pocket_id, pocket_info in self.shot.table.pockets.items():
            center = self.transform_point(pocket_info.center[:2])

            radius = self.transform_distance(pocket_info.radius)
            painter.setBrush(QBrush(self.color_pocket))  # Set the brush color for the circle
            pen.setWidth(0)
            pen.setColor(self.color_pocket)
            painter.setPen(pen)
            painter.drawEllipse(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)

        
        # Draw ball paths
        for ball_id, ball_info in self.shot.balls.items():
            points = []
            current_state_id = ball_info.history_cts.states[0].s

            for i, state in enumerate(ball_info.history_cts.states):
                points.append(self.transform_point(state.rvw[0][:2]))

                if current_state_id != state.s or i == len(ball_info.history_cts.states):
                    # Set the pen color based on the value of s
                    if state.s == 3: # Rolling
                        if ball_info.id in self.color_ball[ball_info.ballset.name].keys():
                            color = self.color_ball[ball_info.ballset.name][ball_info.id]
                        else:
                            # If not valid colour, then display as grey
                            color = QColor("gray")
                        pen.setColor(color.lighter(180))
                    else:
                        pen.setColor(self.color_path[state.s])
                    pen.setWidth(2)
                    painter.setPen(pen)

                    # Draw the path
                    polyline = QPolygon([QPoint(*point) for point in points])
                    painter.drawPolyline(polyline)

                    points = [points[-1]]


        # Draw balls
        for ball_id, ball_info in self.shot.balls.items():
            if ball_info.id in self.color_ball[ball_info.ballset.name].keys():
                color = self.color_ball[ball_info.ballset.name][ball_info.id]
            else:
                # If not valid colour, then display as grey
                color = QColor("gray")

            # Work out the most appropriate state
            best_state = ball_info.history_cts.states[0]
            for state in ball_info.history_cts.states:
                if state.t > self.time:
                    break
                best_state = state
            center = self.transform_point(best_state.rvw[0][:2])
            radius = self.transform_distance(ball_info.params.R)

            painter.setBrush(QBrush(color))  # Set the brush color for the circle
            pen.setWidth(2)
            pen.setColor(QColor("black"))
            painter.setPen(pen)
            painter.drawEllipse(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)


        # Draw cue
        cue_angle = self.shot.cue.phi
        cue_tangent = np.array([np.cos(np.radians(cue_angle)), np.sin(np.radians(cue_angle))]) # Unit vector pointing in direction of shot
        cue_normal = np.array([np.sin(np.radians(cue_angle)), -np.cos(np.radians(cue_angle))]) # Unit vector pointing to the right of the shot

        cue_length = self.shot.cue.specs.length
        cue_tip_radius = self.shot.cue.specs.tip_radius
        cue_butt_radius = self.shot.cue.specs.butt_radius

        cue_ball_position = np.array(self.shot.balls["cue"].history.states[0].rvw[0][:2])
        cue_ball_radius = self.shot.balls["cue"].params.R

        cue_separation = self.shot.cue.V0 * 0.03

        cue_tip = cue_ball_position - cue_tangent * (cue_ball_radius + cue_separation)
        cue_butt = cue_tip - cue_tangent * cue_length

        polygon = QPolygonF()
        polygon.append(QPointF(*self.transform_point(cue_tip - cue_normal*cue_tip_radius)))
        polygon.append(QPointF(*self.transform_point(cue_tip + cue_normal*cue_tip_radius)))
        polygon.append(QPointF(*self.transform_point(cue_butt + cue_normal*cue_butt_radius)))
        polygon.append(QPointF(*self.transform_point(cue_butt - cue_normal*cue_butt_radius)))

        painter.setBrush(QColor("#F5DEB3"))  # Set the brush color to brown
        painter.setPen(QColor("black"))  # Disable the pen (border)
        painter.drawPolygon(polygon)


        # Draw legend
        legend_position = QPoint(self.canvas_widget.width(), self.canvas_widget.height()) + QPoint(-50, -50) * 10
        legend_position = QPoint(800, 657)
        painter.setFont(QFont("Arial", 12))
        painter.setPen(self.color_path[2])
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.setPen(QColor("white"))
        painter.drawText(legend_position + QPoint(60, 6), "Sliding")
        legend_position += QPoint(0, 20)
        painter.setPen(self.color_path[3])
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.setPen(QColor("white"))
        painter.drawText(legend_position + QPoint(60, 6), "Rolling")
        legend_position += QPoint(0, 20)
        painter.setPen(self.color_path[4])
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.drawLine(legend_position, legend_position + QPoint(50, 0))
        painter.setPen(QColor("white"))
        painter.drawText(legend_position + QPoint(60, 6), "Potting")


    def draw_spin_canvas(self, event):
        painter = QPainter(self.spin_canvas_widget)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen()
        pen.setColor(QColor("black"))
        pen.setWidth(1)
        painter.setPen(pen)  # White color
        painter.setBrush(QBrush(QColor("white")))  # White color

        # Draw the circle
        circle_radius = min(self.spin_canvas_widget.width(), self.spin_canvas_widget.height()) / 2 - 2
        circle_center = self.spin_canvas_widget.rect().center()
        painter.drawEllipse(circle_center, circle_radius, circle_radius)

        # Draw crosshairs at the given point
        point_x = int(circle_center.x() + circle_radius * -self.shot.cue.a)
        point_y = int(circle_center.y() + circle_radius *  -self.shot.cue.b)
        crosshair_length = 10
        pen.setColor(QColor("red"))
        painter.setPen(pen)
        painter.drawLine(point_x, point_y - crosshair_length, point_x, point_y + crosshair_length)
        painter.drawLine(point_x - crosshair_length, point_y, point_x + crosshair_length, point_y)


class Hri():
    def __init__(self):
        self.app = QApplication(sys.argv)

        self.window = Ui()
        self.window.show()

    def mainloop(self):
        sys.exit(self.app.exec())


if __name__ == "__main__":
    hri = Hri()

    hri.mainloop()
