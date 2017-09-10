from typing import List

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, logpdf, dot3
from scipy.ndimage.filters import gaussian_filter


class Window:
    def __init__(self, level, window_shape, img_shape, x_init, max_frozen_dur):
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
        self.x_measured = None
        self.dropped = True
        self.frozen = False
        self.frozen_dur = 0
        self.max_frozen_dur = max_frozen_dur

    def x_begin(self, param='x_filtered'):
        x = getattr(self, param)
        return int(max(0, x - self.width // 2))

    def x_end(self, param='x_filtered'):
        x = getattr(self, param)
        return int(min(x + self.width // 2, self.img_w))

    def area(self):
        return self.height * self.width

    def freeze(self):
        self.frozen = True
        if self.frozen_dur > self.max_frozen_dur:
            self.drop()
        self.frozen_dur += 1
        self.filter.grow_uncertainty(5)

    def unfreeze(self):
        self.frozen_dur = 0
        self.dropped = False
        self.frozen = False

    def drop(self):
        self.dropped = True
        # Quickly grow uncertainty
        self.filter.grow_uncertainty(50)

    def update(self, image, x_search_range):
        assert image.shape[0] == self.img_h and \
               image.shape[1] == self.img_w, 'Window not parametrized for this image size'

        # Apply a column-wise gaussian filter to score the x-positions in this window's search region
        x_search_range = (max(0, int(x_search_range[0])), min(int(x_search_range[1]), self.img_w))
        x_offset = x_search_range[0]
        search_region = image[self.y_begin: self.y_end, x_offset: x_search_range[1]]
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
            if signal_noise_ratio < 0.6 or self.filter.loglikelihood(self.x_measured) < -40:
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

    def pos_xy(self, param='x_filtered'):
        assert param == 'x_filtered' or param == 'x_measured', 'Invalid position parameter'
        return getattr(self, param), self.y


def sliding_window_update(windows: List[Window], image, margin, mode):
    assert mode == 'left' or mode == 'right', "Mode not valid."
    img_h, img_w = image.shape[0:2]

    # Update the base window
    if mode == 'left':
        windows[0].update(image, (0, img_w // 2))
    elif mode == 'right':
        windows[0].update(image, (img_w // 2, img_w))

    # Find the starting point for our search
    if windows[0].dropped:
        # Starting window does not exist, find an approximation.
        search_region = image[2 * img_h // 3:, :]  # search bottom 1/3rd of image
        column_scores = gaussian_filter(np.sum(search_region, axis=0), sigma=windows[0].width / 3, truncate=3.0)
        if mode == 'left':
            search_center = argmax_between(column_scores, 0, img_w // 2)
        elif mode == 'right':
            search_center = argmax_between(column_scores, img_w // 2, img_w)
    else:
        # Already know the position of the base window
        search_center = windows[0].x_filtered

    # Continue searching nearby the last detected window
    for window in windows[1:]:
        x_search_range = (search_center - margin, search_center + margin)
        window.update(image, x_search_range)

        if not window.dropped:
            search_center = window.x_filtered


def argmax_between(arr: np.ndarray, begin: int, end: int) -> int:
    max_ndx = np.argmax(arr[begin:end]) + begin
    return max_ndx


def filter_window_list(windows: List[Window], include_frozen=True, include_dropped=False):
    ret_windows = []
    args = []
    for i, window in enumerate(windows):
        if window.dropped and not include_dropped:
            continue
        elif window.frozen and not include_frozen:
            continue
        else:
            ret_windows.append(window)
            args.append(i)
    return ret_windows, args


def window_image(windows: List[Window], param='x_filtered', color=(0, 255, 0), color_frozen=None, color_dropped=None):
    if color_frozen is None:
        color_frozen = [ch * 0.6 for ch in color]
    if color_dropped is None:
        color_dropped = [0, 0, 0]
    mask = np.zeros((windows[0].img_h, windows[0].img_w, 3))
    for window in windows:
        if getattr(window, param) is None:
            continue
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
        self.kf.P = self.kf._alpha_sq * dot3(self.kf.F, self.kf.P, self.kf.F.T) + self.kf.Q

    def loglikelihood(self, pos):
        self.kf.S = dot3(self.kf.H, self.kf.P, self.kf.H.T) + self.kf.R
        return logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)

    def get_position(self):
        return self.kf.x[0]
