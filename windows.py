from copy import deepcopy
from typing import List

import cv2
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.ndimage.filters import gaussian_filter


class Window:
    def __init__(self, level, window_shape, img_shape, x_init):
        if window_shape[1] % 2 == 0:
            raise Exception("width must be odd")
        self.height = window_shape[0]
        self.width = window_shape[1]
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]
        self.level = level

        self.y_begin = self.img_h - (level + 1) * self.height  # top row of pixels for window
        self.y_end = self.y_begin + self.height  # one past the bottom row of pixels for window

        self.x = x_init
        self.y = self.y_begin + self.height / 2.0

        self.magnitude = None
        self.noise_magnitude = None
        self.detected = False

    def x_begin(self):
        return int(max(0, self.x - self.width // 2))

    def x_end(self):
        return int(min(self.x + self.width // 2, self.img_w))

    def signal_noise_ratio(self):
        if self.magnitude == 0:
            return 0.0
        return self.magnitude / (self.magnitude + self.noise_magnitude)


class WindowHandler:
    def __init__(self, mode, window_shape, meas_var, process_var, img_shape):
        # Set mode
        if mode == 'left':
            x_init = img_shape[1] / 4
        elif mode == 'right':
            x_init = (img_shape[1] / 4) * 3
        else:
            raise Exception('Not a valid mode. Should be `left` or `right`')
        self.__mode = mode

        # Create windows
        self.windows_raw = \
            [Window(lvl, window_shape, img_shape, x_init) for lvl in range(img_shape[0] // window_shape[0])]
        self.windows_filtered = \
            deepcopy(self.windows_raw)

        # Create filters
        self.filters = [Kalman1D(meas_var, process_var, pos_init=x_init) for i in
                        range(len(self.windows_raw))]

    def update(self, image):
        # Select image region
        if self.__mode == 'left':
            x_offset = 0
            image = image[:, x_offset: image.shape[1] // 2]
        elif self.__mode == 'right':
            x_offset = image.shape[1] // 2
            image = image[:, x_offset:]
        else:
            raise Exception('self.__mode is not valid. Was it changed?')

        # Go through each layer looking for max pixel locations
        last_x = self.windows_raw[0].x
        for i, raw_window in enumerate(self.windows_raw):
            search_img = image[raw_window.y_begin:raw_window.y_end, :]
            column_scores = self.gaussian_filter_across_columns(search_img, raw_window.width)

            # Find the best window and score it
            if max(column_scores) != 0:
                # Update raw window
                raw_window.x = np.argmax(column_scores) + x_offset
                raw_window.magnitude = np.sum(
                    column_scores[raw_window.x_begin() - x_offset: raw_window.x_end() - x_offset])
                raw_window.noise_magnitude = np.sum(column_scores) - raw_window.magnitude
                raw_window.detected = True

                # Update filtered windows
                self.filters[i].update(raw_window.x, confidence=raw_window.signal_noise_ratio())
                self.windows_filtered[i].x = self.filters[i].get_position()
                self.windows_filtered[i].detected = True

                last_x = self.windows_filtered[i].x
            else:
                self.windows_filtered[i].x = last_x
                raw_window.detected = False
                self.windows_filtered[i].detected = False

    def get_positions(self, mode, drop_undetected=False):
        if mode == 'raw':
            windows = self.windows_raw
        elif mode == 'filtered':
            windows = self.windows_filtered
        else:
            raise Exception('Not a valid mode. Should be `raw` or `filtered`')
        return [(window.x, window.y) for window in windows if window.detected or not drop_undetected]

    def gaussian_filter_across_columns(self, img, width):
        return gaussian_filter(np.sum(img, axis=0), sigma=width / 3, truncate=3.0)

    def img_windows_raw(self, color=(0, 255, 0), show_signal_noise_ratio=True):
        return get_window_img(self.windows_raw, color, show_signal_noise_ratio)

    def img_windows_filtered(self, color=(0, 0, 255)):
        return get_window_img(self.windows_filtered, color, show_signal_noise_ratio=False)


def get_window_img(windows: List[Window], color, show_signal_noise_ratio):
    intensity_func = None if not show_signal_noise_ratio else lambda window: window.signal_noise_ratio()
    mask = create_window_mask(windows, intensity_func)
    return np.array(cv2.merge((mask * color[0], mask * color[1], mask * color[2])), np.uint8)


def create_window_mask(windows: List[Window], intensity_func=None, drop_undetected=False):
    mask = np.zeros((windows[0].img_h, windows[0].img_w))
    for window in windows:
        if drop_undetected and not window.detected:
            continue
        single_window_mask = get_window_mask(window)
        scale = intensity_func(window) if intensity_func is not None else 1
        mask[single_window_mask > 0] = 1 * scale
    return mask


def mask_windows(img, windows: List[Window]):
    mask = create_window_mask(windows)
    masked_img = np.copy(img)
    masked_img[mask != 1] = 0
    return masked_img


def get_window_mask(window: Window):
    img_h, img_w = window.img_h, window.img_w
    mask = np.zeros((img_h, img_w))
    mask[window.y_begin:window.y_end, window.x_begin():window.x_end()] = 1
    return mask


class Kalman1D:
    def __init__(self, meas_var, process_var, pos_init=0.0, uncertainty_init=2 ** 30):
        """
        A one dimensional Kalman filter used to track the position of a single point along one axis.

        State variable:  x = [position,
                              velocity]
        Update function: F = [[1, 1],
                              [0, 1]
                         AKA a constant velocity model.
        """
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # Update function
        self.kf.F = np.array([[1., 1],
                              [0., 0.75]])

        # Measurement function
        self.kf.H = np.array([[1., 0.]])

        # Initial state estimate
        self.kf.x = np.array([pos_init, 0])

        # Initial Covariance matrix
        self.kf.P = np.eye(self.kf.dim_x) * uncertainty_init

        # Measurement noise
        self.base_meas_var = meas_var
        self.kf.R = np.array([[self.base_meas_var]])

        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=process_var)

    def update(self, pos, confidence):
        """
        Given an estimate x position, uses the kalman filter to estimate the most likely true position of the
        lane pixel.
        :param pos: measured x position of the pixel
        :return: best estimate of the true x position of the pixel
        """
        # TODO: Make noise model a passable parameter
        modified_meas_var = np.interp(confidence,
                                      [.4, 0.7, 1],
                                      [self.base_meas_var * 10, self.base_meas_var, self.base_meas_var])
        self.kf.R = np.array([[modified_meas_var]])
        self.kf.predict()
        self.kf.update(pos)

    def get_position(self):
        return self.kf.x[0]
