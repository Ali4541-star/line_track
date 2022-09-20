# TODO: General Note=> Make functions Cython

from enum import Enum, auto
import numpy as np
import cv2
from dataclasses import dataclass
from math import tan, radians


class Errors(Enum):
    BadParameter = "Bad parameter initialized"
    message = "Value of parameters are wrong"
    NO_ERROR = "Fine!!!"

@dataclass
class Line(object):

    point1: tuple
    point2: tuple
    slope: float

class BGRColorPalette(Enum):

    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    PURPLE = (255, 0, 255)
    YELLOW = (255, 255, 0)
    ORANGE = (128, 255, 255) #*
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128) #?

# TODO: Make slope intervals parametric
def get_lines(frame, frame_shape: tuple, max_gap: int = 40):
    lines_right = []
    lines_left = []
    right_infos = []
    left_infos = []
    lines_side_left = []
    lines_side_right = []
    lines = cv2.HoughLinesP(image=frame, rho=1, theta=np.pi / 180, threshold=20,
                            minLineLength=40,
                            maxLineGap=max_gap, lines=np.array([]))
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # xmin, ymax, xmax, ymin
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            print(f"Slope is {slope}")
            if slope > tan(radians(20)) and slope < tan(radians(89.9999)):
                lines_right.append(Line((x1, y1), (x2, y2), slope))

            elif slope < -tan(radians(20)) and slope > -tan(radians(89.9999)):
                lines_left.append(Line((x1, y1), (x2, y2), slope))

            elif -tan(radians(10)) < slope and slope < tan(radians(10)):  # Right or left
                if int(x1 // 2) < frame_shape[0] // 2:  # self.preprocessed.shape[1]:
                    lines_side_left.append(Line((x1, y1), (x2, y2), slope))
                else:
                    lines_side_right.append(Line((x1, y1), (x2, y2), slope))


    return lines_right, lines_left, lines_side_left, lines_side_right


def ROI(img: np.array, bounds_x: tuple, bounds_y: tuple) -> np.array:

    mask = np.zeros_like(img, dtype=np.uint8)
    vertices = np.array([([bounds_x[0], bounds_y[0]], [bounds_x[0], bounds_y[1]],
                [bounds_x[1], bounds_y[1]], [bounds_x[1], bounds_y[0]])], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    img_roi = cv2.bitwise_and(img, mask)
    return img_roi

def selectROIRect(img:np.array, xmax: np.array, xmin:np.array, ymax: np.array, ymin:np.array):
    pass

def preprocess(img: np.array, blur_kernel: tuple, canny_thresh: tuple) -> np.array:

    if canny_thresh[0] >= canny_thresh[1]:
        raise Errors.BadParameter

    img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_processed = cv2.GaussianBlur(img_processed, blur_kernel, 0)
    img_processed = cv2.Canny(img_processed, canny_thresh[0], canny_thresh[1])
    return img_processed


def draw_lines(frame:np.array, color: tuple, *lines):
    line_plot = np.zeros_like(frame)
    for line in lines:
        print(f"Line {line} is not len == 0")
        if not len(lines) == 0:#if not any(line.point1 == np.nan) or not any(line.point2 == np.nan):
            cv2.line(line_plot, line.point1,
                    line.point2, color, 10)

    return line_plot