from typing import List

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, logpdf
from scipy.ndimage.filters import gaussian_filter


class Window:
    def __init__(self, level, window_shape, img_shape, x_init, x_search_range):
        if window_shape[1] % 2 == 0:
            raise Exception("width must be odd")
        # Image info
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]

        # Window shape
        self.height = window_shape[0]
        self.width = window_shape[1]
        self.y_begin = self.img_h - (level + 1) * self.height  # top row of pixels for window
        self.y_end = self.y_begin + self.height  # one past the bottom row of pixels for window

        # Window position
        self.x_filtered = x_init
        self.y = self.y_begin + self.height / 2.0
        self.level = level

        # Detection info
        self.filter = WindowFilter(pos_init=x_init)
        self.x_search_range = x_search_range
        self.x_measured = None
        self.dropped = True
        self.frozen = False
        self.frozen_dur = 0
        self.max_frozen_dur = 8

    def x_begin(self, param='x_filtered'):
        x = getattr(self, param)
        return int(max(0, x - self.width // 2))

    def x_end(self, param='x_filtered'):
        x = getattr(self, param)
        return int(min(x + self.width // 2, self.img_w))

    def area(self):
        return self.height * self.width

    def freeze(self):
        if self.frozen_dur > self.max_frozen_dur:
            self.drop()
        else:
            self.frozen = True
        self.frozen_dur += 1
        self.filter.grow_uncertainty(5)

    def unfreeze(self):
        self.frozen_dur = 0
        self.dropped = False
        self.frozen = False

    def drop(self):
        self.dropped = True
        # Reset filter
        self.filter.kf.P = np.eye(self.filter.kf.dim_x) * 2 ** 30

    def update(self, image):
        assert image.shape[0] == self.img_h and \
               image.shape[1] == self.img_w, 'Window not parameterized for this image size'

        # Apply a column-wise gaussian filter to score the x-positions in this window's search region
        x_offset = self.x_search_range[0]
        search_region = image[self.y_begin: self.y_end, x_offset: self.x_search_range[1]]
        column_scores = gaussian_filter(np.sum(search_region, axis=0), sigma=self.width / 3, truncate=3.0)

        if max(column_scores) != 0:
            # Update measurement
            self.x_measured = np.argmax(column_scores) + x_offset
            window_magnitude = \
                np.sum(column_scores[self.x_begin('x_measured') - x_offset: self.x_end('x_measured') - x_offset])
            noise_magnitude = np.sum(column_scores) - window_magnitude
            signal_noise_ratio = \
                window_magnitude / (window_magnitude + noise_magnitude) if window_magnitude is not 0 else 0

            # Filter measurement and set position
            if signal_noise_ratio < 0.6 or self.filter.loglikelihood(
                    self.x_measured) < -40:  # noise_magnitude < self.area() * 0.01 or
                # Bad measurement, don't update filter/position
                self.freeze()
                return
            self.unfreeze()
            self.filter.update(self.x_measured)
            self.x_filtered = self.filter.get_position()

        else:
            # No signal in search region
            self.freeze()

    def get_mask(self, param='x_filtered'):
        mask = np.zeros((self.img_h, self.img_w))
        mask[self.y_begin: self.y_end, self.x_begin(param): self.x_end(param)] = 1
        return mask


def window_batch_positions(windows: List[Window], param, include_frozen=True, include_dropped=False):
    positions = []
    for window in windows:
        if window.dropped and not include_dropped:
            continue
        elif window.frozen and not include_frozen:
            continue
        else:
            positions.append((getattr(window, param), window.y))
    return positions


def window_image(windows: List[Window], param='x_filtered',
                 color=(0, 255, 0), color_frozen=None, color_dropped=None):
    if color_frozen is None:
        color_frozen = [ch * 0.6 for ch in color]
    if color_dropped is None:
        color_dropped = [0, 0, 0]
    mask = np.zeros((windows[0].img_h, windows[0].img_w, 3))
    for window in windows:
        if window.dropped:
            color_curr = color_dropped
        elif window.frozen:
            color_curr = color_frozen
        else:
            color_curr = color
        mask[window.get_mask(param) > 0] = color_curr
    return mask.astype('uint8')


class WindowFilter:
    def __init__(self, pos_init=0.0, meas_variance=50, process_variance=1, uncertainty_init=2 ** 30):
        """
        A one dimensional Kalman filter tuned to track the position of a window.

        State variable:   = [position,
                             velocity]
        """
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition function
        self.kf.F = np.array([[1., 1],
                              [0., 0.5]])

        # Measurement function
        self.kf.H = np.array([[1., 0.]])

        # Initial state estimate
        self.kf.x = np.array([pos_init, 0])

        # Initial Covariance matrix
        self.kf.P = np.eye(self.kf.dim_x) * uncertainty_init

        # Measurement noise
        self.kf.R = np.array([[meas_variance]])

        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=process_variance)

    def update(self, pos):
        """
        Given an estimate x position, uses the kalman filter to estimate the most likely true position of the
        lane pixel.
        :param pos: measured x position of the pixel
        :return: best estimate of the true x position of the pixel
        """
        self.kf.predict()
        self.kf.update(pos)

    def grow_uncertainty(self, mag):
        """Grows state uncertainty."""
        # P = FPF' + Q
        self.kf.P = self.kf.Q * mag

    def loglikelihood(self, pos):
        return logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)

    def get_position(self):
        return self.kf.x[0]
