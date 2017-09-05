import cv2
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from filterpy.stats import logpdf
from scipy.ndimage import convolve as center_convolve


class Window:
    def __init__(self, x_pos=None, internal_response=None, external_response=None):
        self.x_pos = x_pos
        self.internal_response = internal_response
        self.external_response = external_response

    def signal_noise_ratio(self):
        if self.internal_response == 0:
            return 0.0
        return self.internal_response / (self.internal_response + self.external_response)


class WindowSet:
    def __init__(self, size):
        self.width = size[0]
        self.height = size[1]


class WindowTracker:
    def __init__(self, n_windows, window_size, meas_var=100, process_var=1.0):
        self.window_filters = [Kalman1D(meas_var, process_var) for i in range(n_windows)]
        self.window_width, self.window_height = window_size

    def update(self, window_pos):
        if len(window_pos) != len(self.window_filters):
            raise Exception('window_pos and self.window_filters must have the same length')
        for i, pos in enumerate(window_pos):
            self.window_filters[i].update(pos)

    def get_estimate(self):
        window_positions = [point.get_position() for point in self.window_filters]
        return window_positions


def find_windows(image, window_w, window_h):
    img_h, img_w = image.shape[0:2]
    windows_left = []  # stored per level
    windows_right = []  # stored per level

    # Get a starting guess for the line
    img_strip = image[2 * img_h // 3:, :]  # search bottom 1/3rd of image
    column_scores = score_columns(img_strip, window_w)
    lw_center = argmax_between(column_scores, begin=0, end=img_w // 2)
    rw_center = argmax_between(column_scores, begin=img_w // 2, end=img_w)

    # Go through each layer looking for max pixel locations
    for level in range(0, image.shape[0] // window_h):
        img_strip = image[img_h - (level + 1) * window_h:img_h - level * window_h, :]
        column_scores = score_columns(img_strip, window_w)

        # Find the best left window and score it
        window_left = Window()
        l_max_ndx = argmax_between(column_scores, 0, img_w // 2)
        lw_center = l_max_ndx if column_scores[l_max_ndx] != 0 else lw_center
        window_left.x_pos = lw_center
        window_left.internal_response = np.sum(
            column_scores[max(0, lw_center - window_w // 2): min(lw_center + window_w // 2 + 1, img_w)])
        window_left.external_response = np.sum(column_scores[:img_w // 2]) - window_left.internal_response

        # Find the best right window and score it
        window_right = Window()
        r_max_ndx = argmax_between(column_scores, img_w // 2, img_w)
        rw_center = r_max_ndx if column_scores[r_max_ndx] != 0 else rw_center
        window_right.x_pos = rw_center
        window_right.internal_response = np.sum(
            column_scores[max(0, rw_center - window_w // 2): min(rw_center + window_w // 2 + 1, img_w)])
        window_right.external_response = np.sum(column_scores[img_w // 2:]) - window_right.internal_response

        windows_left.append(window_left)
        windows_right.append(window_right)

    return windows_left, windows_right


def overlay_windows(img, windows, width, height, color=(0, 255, 0), show_confidence=True):
    # Points used to draw all the left and right windows
    l_points = np.zeros((img.shape[0], img.shape[1]))
    r_points = np.zeros((img.shape[0], img.shape[1]))

    # Go through each level and draw the windows
    left_windows = windows[0]
    right_windows = windows[1]
    for level in range(0, len(left_windows)):
        # Window_mask is a function to draw window areas
        left_window = left_windows[level]
        right_window = right_windows[level]
        l_mask = single_window_mask(img, left_window, width, height, level)
        r_mask = single_window_mask(img, right_window, width, height, level)
        # Add graphic points from window mask here to total pixels found
        l_points[l_mask > 0] = 1 + show_confidence * left_window.signal_noise_ratio() ** 2
        r_points[r_mask > 0] = 1 + show_confidence * right_window.signal_noise_ratio() ** 2

    # Draw the results
    template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
    template = np.array(cv2.merge((template * color[0], template * color[1], template * color[2])),
                        np.uint8)  # make window pixels green
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.array(cv2.merge((img, img, img)), np.uint8)  # making 3 color channels
    return cv2.addWeighted(img, 1, template, 0.5, 0.0)  # overlay the original image with window results


def mask_windows(img, windows, window_width, window_height):
    if len(windows) <= 0:
        return
    # Create a mask of all window areas
    mask = np.zeros_like(img)
    for level in range(0, len(windows)):
        # Find the mask for this window
        this_windows_mask = single_window_mask(img, windows[level], window_width, window_height, level)
        # Add it to our overall mask
        mask[(mask == 1) | (this_windows_mask == 1)] = 1

    # Apply the mask
    masked_img = np.copy(img)
    masked_img[mask != 1] = 0
    return masked_img


def single_window_mask(img_ref, window: Window, width, height, level):
    output = np.zeros((img_ref.shape[0], img_ref.shape[1]))
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(window.x_pos - width / 2)):min(int(window.x_pos + width / 2), img_ref.shape[1])] = 1
    return output


def score_columns(image, window_width):
    assert window_width % 2 != 0, 'window_width must be odd'
    window = np.ones(window_width)
    col_sums = np.sum(image, axis=0)
    scores = center_convolve(col_sums, window, mode='constant')
    return scores


def argmax_between(arr: np.ndarray, begin: int, end: int):
    max_ndx = np.argmax(arr[begin:end]) + begin
    return max_ndx


class Kalman1D:
    def __init__(self, meas_var, process_var, log_likelihood_min=None, pos_init=0, uncertainty_init=10 ** 9):
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
        self.kf.F = np.array([[1., 1.],
                              [0., 1.]])
        self.log_likelihood_min = log_likelihood_min

        # Measurement function
        self.kf.H = np.array([[1., 0.]])

        # Initial state estimate
        self.kf.x = np.array([pos_init, 0])

        # Initial Covariance matrix
        self.kf.P = np.eye(self.kf.dim_x) * uncertainty_init

        # Measurement noise
        self.kf.R = np.array([[meas_var]])

        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=process_var)

    def update(self, pos):
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

        self.kf.predict()
        self.kf.update(pos)

    def get_position(self):
        return self.kf.x[0]
