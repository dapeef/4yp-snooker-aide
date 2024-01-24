from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QPolygon
import sys
import os
import json
import pooltool as pt


class Ui(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI
        uic.loadUi("gui.ui", self)

        # Set final canvas_widget values
        self.canvas_table_width = 1000
        self.canvas_table_height = self.canvas_table_width / 2
        self.canvas_padding = int(self.canvas_table_width / 10)
        self.canvas_widget.setFixedWidth(int(self.canvas_table_width + 2*self.canvas_padding))
        self.canvas_widget.setFixedHeight(int(self.canvas_table_height + 2*self.canvas_padding))
        self.canvas_widget.paintEvent = self.draw_canvas

        # Set some colour values
        self.color_table = QColor("#1ea625")
        self.color_cushion = QColor("#0b5e0f")
        self.color_top = QColor("#5e400b")
        self.color_pocket = QColor("#000000")
        
        self.color_path = [
            Qt.black, # stationary = 0
            Qt.blue,  # spinning = 1
            Qt.red,   # sliding = 2
            Qt.white, # rolling = 3
            Qt.green  # pocketed = 4
        ]


        self.shot = pt.System.load("temp/pool_tool_output.json")

    def transform_distance(self, real_distance):
        table_width = self.shot.table.w
        table_length = self.shot.table.l

        return int(real_distance / table_length * self.canvas_table_width)
    
    def transform_point(self, real_xy):
        return (
            int(self.transform_distance(real_xy[1]) + self.canvas_padding),
            int(self.transform_distance(real_xy[0]) + self.canvas_padding)
        )

    def draw_canvas(self, event):
        painter = QPainter(self.canvas_widget)
        pen = QPen()
        painter.setRenderHint(QPainter.Antialiasing)  # Optional: Enable antialiasing for smoother circle
        painter.fillRect(self.rect(), self.color_table) # Set background colour
        painter.setBrush(QBrush(Qt.SolidPattern))  # Set the brush style

        cushion_edge = self.canvas_widget.height()

        # Cushions
        pen.setColor(self.color_cushion)
        pen.setWidth(1)
        painter.setPen(pen)
        
        # Cushion lines
        for line_id, line_info in self.shot.table.cushion_segments.linear.items():
            p1 = line_info.p1[:2]  # Discard the z-value
            p2 = line_info.p2[:2]  # Discard the z-value

            # Scale the points to fit within the widget dimensions
            scaled_p1 = self.transform_point(p1)
            scaled_p2 = self.transform_point(p2)

            painter.drawLine(scaled_p1[0], scaled_p1[1], scaled_p2[0], scaled_p2[1])

            # Save lowest x value - this is to draw the top
            cushion_edge = min([cushion_edge, scaled_p1[1], scaled_p2[1]])

            # radius = 5  # Adjust the radius as needed
            # painter.setBrush(QBrush(Qt.red))  # Set the brush color for the circle
            # painter.drawEllipse(scaled_p1[0] - radius, scaled_p1[1] - radius, 2 * radius, 2 * radius)

            # # Draw text at the center of the line
            # text = line_info.id + ", " + str(line_info.direction)
            # text_position = (int((scaled_p1[0] + scaled_p2[0]) / 2), int((scaled_p1[1] + scaled_p2[1]) / 2))
            # painter.setBrush(QBrush(Qt.white))  # Set the brush color for the text
            # font = QFont()
            # font.setPointSize(10)  # Adjust the font size as needed
            # painter.setFont(font)
            # painter.drawText(text_position[0] - 20, text_position[1] - 10, text)

        # Cushion circles
        for line_id, line_info in self.shot.table.cushion_segments.circular.items():
            # Scale the points to fit within the widget dimensions
            center = self.transform_point(line_info.center[:2])

            radius = self.transform_distance(line_info.radius)
            # painter.setBrush(QBrush(self.color_cushion))  # Set the brush color for the circle
            painter.setBrush(Qt.NoBrush)  # Set the brush color for the circle
            painter.drawEllipse(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)

        
        # Top
        # Draw rectangles forming a frame with padding
        frame_rect_top = QRect().adjusted(0, 0, self.canvas_widget.width(), cushion_edge)
        frame_rect_left = QRect().adjusted(0, 0, cushion_edge, self.canvas_widget.height())
        frame_rect_right = QRect().adjusted(self.canvas_widget.width() - cushion_edge, 0, self.canvas_widget.width(), self.canvas_widget.height())
        frame_rect_bottom = QRect().adjusted(0, self.canvas_widget.height() - cushion_edge, self.canvas_widget.width(), self.canvas_widget.height())

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
                    pen.setColor(self.color_path[state.s])
                    pen.setWidth(2)
                    painter.setPen(pen)

                    # Draw the path
                    polyline = QPolygon([QPoint(*point) for point in points])
                    painter.drawPolyline(polyline)

                    points = [points[-1]]


        # Draw balls
        for ball_id, ball_info in self.shot.balls.items():
            center = self.transform_point(ball_info.state.rvw[0][:2])

            radius = self.transform_distance(ball_info.params.R)
            painter.setBrush(QBrush(Qt.white))  # Set the brush color for the circle
            pen.setWidth(0)
            pen.setColor(Qt.white)
            painter.setPen(pen)
            painter.drawEllipse(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)

class Hri():
    def __init__(self):
        self.app = QApplication(sys.argv)

        self.window = Ui()
        self.window.show()

    def mainloop(self):
        self.app.exec()


if __name__ == "__main__":
    hri = Hri()

    hri.mainloop()
