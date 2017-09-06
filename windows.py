from copy import deepcopy
from typing import List

import cv2
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from filterpy.stats import logpdf
from scipy.ndimage.filters import gaussian_filter


class Window:
    def __init__(self, level, width, height, img_shape):
        if width % 2 == 0:
            raise Exception("width must be odd")
        self.width = width
        self.height = height
        self.img_w = img_shape[1]
        self.img_h = img_shape[0]

        self.y_begin = self.img_h - (level + 1) * self.height
        self.y_end = self.y_begin + height
        self.y = self.y_begin + height / 2.0
        self.x = None

        self.level = level

        self.magnitude = None
        self.noise_magnitude = None

    def x_begin(self):
        return int(max(0, self.x - self.width // 2))

    def x_end(self):
        return int(min(self.x + self.width // 2, self.img_w))

    def signal_noise_ratio(self):
        if self.magnitude == 0:
            return 0.0
        return self.magnitude / (self.magnitude + self.noise_magnitude)


class WindowTracker:
    def __init__(self, mode, window_width, window_height, meas_variance, process_variance, img_shape):
        if not (mode == 'left' or mode == 'right'):
            raise Exception('Not a valid mode. Should be `left` or `right`')
        self.__mode = mode
        self.windows_raw = [Window(lvl, window_width, window_height, img_shape)
                            for lvl in range(img_shape[0] // window_height)]
        self.windows_filtered = deepcopy(self.windows_raw)
        self.filters = [Kalman1D(meas_variance, process_variance, log_likelihood_min=-20) for i in
                        range(len(self.windows_raw))]

    def update(self, image):
        if self.__mode == 'left':
            x_offset = 0
            image = image[:, x_offset: image.shape[1] // 2]
        elif self.__mode == 'right':
            x_offset = image.shape[1] // 2
            image = image[:, x_offset:]

        img_h, img_w = image.shape[:2]

        # Get a starting guess for the line by finding the center of intensity in the bottom section of the image
        bottom_column_scores = self.gaussian_filter_across_columns(image[2 * img_h // 3:, :], self.windows_raw[0].width)
        x_last = np.argmax(bottom_column_scores) + x_offset

        # Go through each layer looking for max pixel locations
        for window in self.windows_raw:
            search_img = image[window.y_begin:window.y_end, :]
            column_scores = self.gaussian_filter_across_columns(search_img, window.width)

            # Find the best window and score it
            x_max_score = np.argmax(column_scores) + x_offset
            window.x = x_max_score if max(column_scores) != 0 else x_last  # reuse x_last if no max is found (empty img)
            window.magnitude = np.sum(column_scores[window.x_begin() - x_offset: window.x_end() - x_offset])
            window.noise_magnitude = np.sum(column_scores) - window.magnitude

            x_last = window.x

        # Filter positions
        for i in range(len(self.windows_raw)):
            self.filters[i].update(self.windows_raw[i].x, self.windows_raw[i].signal_noise_ratio())
            self.windows_filtered[i].x = self.filters[i].get_position()

    def get_positions(self, mode):
        if mode == 'raw':
            windows = self.windows_raw
        elif mode == 'filtered':
            windows = self.windows_filtered
        else:
            raise Exception('Not a valid mode. Should be `raw` or `filtered`')
        return [(window.x, window.y) for window in windows]

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


def create_window_mask(windows: List[Window], intensity_func=None):
    mask = np.zeros((windows[0].img_h, windows[0].img_w))
    for window in windows:
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
    def __init__(self, meas_var, process_var, log_likelihood_min=None, pos_init=0.0, uncertainty_init=2 ** 30):
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
        self.log_likelihood_min = log_likelihood_min

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
        # Apply outlier rejection using log likelihood
        if self.log_likelihood_min is not None:
            pos_log_likelihood = logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)
            if pos_log_likelihood <= self.log_likelihood_min:
                # Log Likelihood is too low, most likely an outlier. Reject this measurement.
                return

        # TODO: Make noise model a passable parameter
        modified_meas_var = np.interp(confidence,
                                      [.4, 0.7, 1],
                                      [self.base_meas_var * 10, self.base_meas_var, self.base_meas_var])
        self.kf.R = np.array([[modified_meas_var]])
        self.kf.predict()
        self.kf.update(pos)

    def get_position(self):
        return self.kf.x[0]
