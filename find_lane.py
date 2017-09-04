#!/usr/bin/env python
"""
Finds lane lines and their curvature from camera input_video.

Author: Peter Moran
Created: 8/1/2017
"""
import glob
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import symfit
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, logpdf
from imageio.core import NeedDownloadError
from scipy.ndimage.filters import convolve as center_convolve

from dynamic_subplot import DynamicSubplot
from udacity_tools import overlay_windows, single_window_mask

# Import moviepy and install ffmpeg if needed.
try:
    from moviepy.editor import VideoFileClip
except NeedDownloadError as download_err:
    if 'ffmpeg' in str(download_err):
        prompt = input('The dependency `ffmpeg` is missing, would you like to download it? [y]/n')
        if prompt == '' or prompt == 'y' or prompt == 'Y':
            from imageio.plugins import ffmpeg

            ffmpeg.download()
            from moviepy.editor import VideoFileClip
        else:
            raise download_err
    else:
        # Unknown download error
        raise download_err


def threshold_lanes(image, base_threshold=50, thresh_window=411):
    # Mask the image
    binary = cv2.adaptiveThreshold(
        image,
        maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=thresh_window,
        C=base_threshold * -1)

    return binary


def find_window_centers(image, window_width, window_height):
    img_h, img_w = image.shape[0:2]
    window_lr_centroids = []  # Store the (left,right) window centroid positions per level

    # Get a starting guess for the line
    img_strip = image[2 * img_h // 3:, :]  # search bottom 1/3rd of image
    column_scores = score_columns(img_strip, window_width)
    last_window_left = argmax_between(column_scores, begin=0, end=img_w // 2)
    last_window_right = argmax_between(column_scores, begin=img_w // 2, end=img_w)

    # Go through each layer looking for max pixel locations
    for level in range(0, image.shape[0] // window_height):
        img_strip = image[img_h - (level + 1) * window_height:img_h - level * window_height, :]
        column_scores = score_columns(img_strip, window_width)

        # Find the best left window
        l_max_ndx = argmax_between(column_scores, 0, img_w // 2 - 1)

        # Find the best right window
        r_max_ndx = argmax_between(column_scores, img_w // 2, img_w)

        # If there were no pixels in search region (ie max was zero), reuse the last window.
        last_window_left = l_max_ndx if column_scores[l_max_ndx] != 0 else last_window_left
        last_window_right = r_max_ndx if column_scores[r_max_ndx] != 0 else last_window_right

        window_lr_centroids.append((last_window_left, last_window_right))

    return window_lr_centroids


def score_columns(image, window_width):
    assert window_width % 2 != 0, 'window_width must be odd'
    window = np.ones(window_width)
    col_sums = np.sum(image, axis=0)
    scores = center_convolve(col_sums, window, mode='constant')
    return scores


def argmax_between(arr: np.ndarray, begin: int, end: int):
    max_ndx = np.argmax(arr[begin:end]) + begin
    return max_ndx


def get_nonzero_pixel_locations(binary_img):
    """
    Returns the x, y locations of all nonzero pixels.
    """
    points = np.argwhere(binary_img)
    if len(points) == 0:
        return None
    y, x = zip(*points)  # mapping r,c -> y,x format.
    return x, y


def fit_parallel_polynomials(points_left, points_right, n_rows):
    # TODO: Split function in half and make more generalizable.
    # Define global model to fit
    x_left, y_left, x_right, y_right = symfit.variables('x_left, y_left, x_right, y_right')
    a, b, x0_left, x0_right = symfit.parameters('a, b, x0_left, x0_right')

    model = symfit.Model({
        x_left: a * y_left ** 2 + b * y_left + x0_left,
        x_right: a * y_right ** 2 + b * y_right + x0_right
    })

    # Apply fit
    xl, yl = points_left
    xr, yr = points_right
    fit = symfit.Fit(model, x_left=xl, y_left=yl, x_right=xr, y_right=yr)
    fit = fit.execute()
    fit_vals = {'a': fit.value(a), 'b': fit.value(b), 'x0_left': fit.value(x0_left), 'x0_right': fit.value(x0_right)}

    # Determine the location of the polynomial fit line for each row of the image
    y_fit = np.linspace(0, n_rows - 1, num=n_rows).flatten()  # to cover y-range of image
    x_left_fit = fit_vals['a'] * y_fit ** 2 + fit_vals['b'] * y_fit + fit_vals['x0_left']
    x_right_fit = fit_vals['a'] * y_fit ** 2 + fit_vals['b'] * y_fit + fit_vals['x0_right']
    return y_fit, x_left_fit, x_right_fit, fit_vals


def mask_windows(img, centers, window_width, window_height):
    if len(centers) <= 0:
        return
    # Create a mask of all window areas
    mask = np.zeros_like(img)
    for level in range(0, len(centers)):
        # Find the mask for this window
        this_windows_mask = single_window_mask(img, centers[level], window_width, window_height, level)
        # Add it to our overall mask
        mask[(mask == 1) | (this_windows_mask == 1)] = 1

    # Apply the mask
    masked_img = np.copy(img)
    masked_img[mask != 1] = 0
    return masked_img


def draw_lane(undist_img, camera, left_fit_x, right_fit_x, fit_y):
    """
    Take an undistorted dashboard camera image and highlights the lane.
    :param undist_img: An undistorted dashboard view image.
    :param camera: The DashboardCamera object for the camera the image was taken on.
    :param left_fit_x: the x values for the left line polynomial at the given y values.
    :param right_fit_x: the x values for the right line polynomial at the given y values.
    :param fit_y: the y values the left and right line x values were calculated at.
    :return: The undistorted image with the lane overlaid on top of it.
    """
    # Create an undist_img to draw the lines on
    lane_poly_overhead = np.zeros_like(undist_img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([np.array(left_fit_x), fit_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, fit_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank undist_img
    cv2.fillPoly(lane_poly_overhead, np.int_([pts]), (0, 255, 0))

    # Warp back to original undist_img space
    lane_poly_dash = camera.warp_to_dashboard(lane_poly_overhead)

    # Combine the result with the original undist_img
    return cv2.addWeighted(undist_img, 1, lane_poly_dash, 0.3, 0)


class CurveTracker:
    def __init__(self, n_points, meas_var=100, process_var=1.0):
        self.curve_points = [Kalman1D(meas_var, process_var) for i in range(n_points)]

    def update(self, curve_points_pos):
        if len(curve_points_pos) != len(self.curve_points):
            raise Exception('curve_points_pos and self.curve_points must have the same length')
        for i, pos in enumerate(curve_points_pos):
            self.curve_points[i].update(pos)

    def get_estimate(self):
        point_positions = [point.get_position() for point in self.curve_points]
        return point_positions


class Kalman1D:
    def __init__(self, meas_var, process_var, log_likelihood_min=-100.0, pos_init=0, uncertainty_init=10 ** 9):
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
        pos_log_likelihood = logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)
        if pos_log_likelihood <= self.log_likelihood_min:
            # Log Likelihood is too low, most likely an outlier. Reject this measurement.
            return

        self.kf.predict()
        self.kf.update(pos)

    def get_position(self):
        return self.kf.x[0]


class DashboardCamera:
    def __init__(self, chessboard_img_fnames, chessboard_size, lane_shape):
        """
        Class for dashboard camera calibration, perspective warping, and mainlining various properties.
        :param chessboard_img_fnames: List of files name locations for the calibration images.
        :param chessboard_size: Size of the calibration chessboard.
        :param lane_shape: Pixel points for the trapezoidal profile of the lane (on straight road), clockwise starting
                           from the top left.
        """
        # Get image size
        example_img = cv2.imread(chessboard_img_fnames[0])
        self.img_size = example_img.shape[0:2]
        self.img_height = self.img_size[0]
        self.img_width = self.img_size[1]

        # Calibrate
        self.camera_matrix, self.distortion_coeffs = self.calibrate(chessboard_img_fnames, chessboard_size)

        # Define overhead transform and its inverse
        top_left, top_right, bottom_left, bottom_right = lane_shape
        source = np.float32([top_left, top_right, bottom_right, bottom_left])
        destination = np.float32([(bottom_left[0], 0), (bottom_right[0], 0),
                                  (bottom_right[0], self.img_height - 1), (bottom_left[0], self.img_height - 1)])
        self.overhead_transform = cv2.getPerspectiveTransform(source, destination)
        self.inverse_overhead_transform = cv2.getPerspectiveTransform(destination, source)

    def calibrate(self, chessboard_img_files, chessboard_size):
        """
        Calibrates the camera using chessboard calibration images.
        :param chessboard_img_files: List of files name locations for the calibration images.
        :param chessboard_size: Size of the calibration chessboard.
        :return: Two lists: objpoints, imgpoints
        """
        # Create placeholder lists
        chess_rows, chess_cols = chessboard_size
        chess_corners = np.zeros((chess_cols * chess_rows, 3), np.float32)
        chess_corners[:, :2] = np.mgrid[0:chess_rows, 0:chess_cols].T.reshape(-1, 2)
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Determine object point, image point pairs
        for fname in chessboard_img_files:
            # Load images
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            found, img_corners = cv2.findChessboardCorners(gray, (chess_rows, chess_cols), None)

            # If found, save object points, image points
            if found:
                objpoints.append(chess_corners)
                imgpoints.append(img_corners)

        # Perform calibration
        success, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.img_size,
                                                                                None, None)
        if not success:
            raise Exception("Camera calibration unsuccessful.")
        return camera_matrix, dist_coeffs

    def undistort(self, image):
        """
        Removes distortion this camera's raw images.
        """
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs, None, self.camera_matrix)

    def warp_to_overhead(self, dashboard_image):
        """
        Transforms this camera's images from the dashboard perspective to an overhead perspective.
        :param dashboard_image: an image taken from the dashboard of the car. Aka a raw image.
        """
        return cv2.warpPerspective(dashboard_image, self.overhead_transform, dsize=(self.img_width, self.img_height))

    def warp_to_dashboard(self, dashboard_image):
        """
        Transforms this camera's images from the dashboard perspective to an overhead perspective.
        :param dashboard_image: an image taken from the dashboard of the car. Aka a raw image.
        """
        return cv2.warpPerspective(dashboard_image, self.inverse_overhead_transform,
                                   dsize=(self.img_width, self.img_height))


class LaneFinder:
    def __init__(self, camera: DashboardCamera):
        self.camera = camera

        # Window parameters
        self.window_width = 81
        self.window_height = 100

        # Two curve trackers, one point for each window
        n_window_per_lane = camera.img_height // self.window_height
        self.line_trackers = [CurveTracker(n_points=n_window_per_lane) for i in range(2)]

        # Initialize visuals to empty images
        VIZ_OPTIONS = ('dash_undistorted', 'overhead', 'saturation', 'saturation_binary', 'lightness',
                       'lightness_binary', 'pixel_scores', 'windows', 'masked_pixel_scores', 'highlighted_lane')
        self.visuals = {name: None for name in VIZ_OPTIONS}
        self.__viz_options = None

    def find_lines(self, img_dashboard, visuals=None):
        # Account for visualization options
        if visuals is None:
            visuals = ['highlighted_lane']
        self.__viz_options = self.fix_viz_dependencies(visuals)

        # Undistort and transform to overhead view
        img_dash_undistorted = self.camera.undistort(img_dashboard)
        img_overhead = self.camera.warp_to_overhead(img_dash_undistorted)

        # Score pixels
        pixel_scores = self.score_pixels(img_overhead)

        # Select lane lines and mask them out
        windows_left, windows_right = self.select_windows(pixel_scores)
        masked_scores_left = mask_windows(pixel_scores, windows_left, self.window_width, self.window_height)
        masked_scores_right = mask_windows(pixel_scores, windows_right, self.window_width, self.window_height)

        # TODO: Do something if no line found

        # Fit lines to selected scores
        pixel_locs_left = get_nonzero_pixel_locations(masked_scores_left)
        pixel_locs_right = get_nonzero_pixel_locations(masked_scores_right)
        y_fit, x_fit_left, x_fit_right, fit = fit_parallel_polynomials(pixel_locs_left, pixel_locs_right,
                                                                       self.camera.img_height)

        # TODO: Calculate radius of curvature

        # Log visuals
        self.save_visual('dash_undistorted', img_dash_undistorted)
        self.save_visual('overhead', img_overhead)
        self.save_visual('pixel_scores', pixel_scores,
                         img_proc_func=lambda img: cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
        self.save_visual('windows', self.visuals['pixel_scores'],
                         img_proc_func=lambda img: overlay_windows(img, list(zip(windows_left, windows_right)),
                                                                   self.window_width, self.window_height))
        self.save_visual('masked_pixel_scores', masked_scores_left + masked_scores_right,
                         img_proc_func=lambda img: cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
        self.save_visual('highlighted_lane', img_dash_undistorted,
                         img_proc_func=lambda img: draw_lane(img, self.camera, x_fit_left, x_fit_right, y_fit))

        return y_fit, x_fit_left, x_fit_right

    def score_pixels(self, img) -> np.ndarray:
        # Change color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img_lightness = hls[:, :, 1]
        img_saturation = hls[:, :, 2]

        # Threshold for lanes
        img_binary_L = threshold_lanes(img_lightness)
        img_binary_S = threshold_lanes(img_saturation)

        # Stack binary images
        lightness_score = cv2.normalize(img_binary_L, None, 0, 1, cv2.NORM_MINMAX)
        saturation_score = cv2.normalize(img_binary_S, None, 0, 1, cv2.NORM_MINMAX)

        # Log visuals
        self.save_visual('saturation', img_lightness)
        self.save_visual('saturation_binary', img_binary_S)
        self.save_visual('lightness', img_saturation)
        self.save_visual('lightness_binary', img_binary_S)

        return lightness_score + saturation_score

    def select_windows(self, pixel_scores) -> (list, list):
        # Select lane lines
        window_centers = find_window_centers(pixel_scores, self.window_width, self.window_height)

        # Filter window selection
        windows_left, windows_right = zip(*window_centers)
        self.line_trackers[0].update(windows_left)
        self.line_trackers[1].update(windows_right)
        left_window_prediction = self.line_trackers[0].get_estimate()
        right_window_prediction = self.line_trackers[1].get_estimate()
        return left_window_prediction, right_window_prediction

    def save_visual(self, name, img, img_proc_func=None):
        if 'all' not in self.__viz_options and name not in self.__viz_options:
            return  # Don't save this image
        if img_proc_func is not None:
            img = img_proc_func(img)
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise Exception('Image is not 3 channels or could not be converted to 3 channels. Cannot use.')
        self.visuals[name] = img

    def fix_viz_dependencies(self, viz_options: list):
        if 'windows' in viz_options:
            viz_options.append('pixel_scores')
        return viz_options

    def process_and_return(self, img, visual='highlighted_lane'):
        self.find_lines(img, [visual])
        return self.visuals[visual]

    def callback_func(self, visual='highlighted_lane'):
        return lambda img: self.process_and_return(img, visual=visual)

    def plot_pipeline(self, img):
        y_fit, x_fit_left, x_fit_right = self.find_lines(img, ['all'])
        dynamic_subplot = DynamicSubplot(3, 4)
        dynamic_subplot.imshow(self.visuals['dash_undistorted'], "Undistorted Road")
        dynamic_subplot.imshow(self.visuals['overhead'], "Overhead", cmap='gray')
        dynamic_subplot.imshow(self.visuals['lightness'], "Lightness", cmap='gray')
        dynamic_subplot.imshow(self.visuals['lightness_binary'], "Binary Lightness", cmap='gray')
        dynamic_subplot.skip_plot()
        dynamic_subplot.skip_plot()
        dynamic_subplot.imshow(self.visuals['saturation'], "Saturation", cmap='gray')
        dynamic_subplot.imshow(self.visuals['saturation_binary'], "Binary Saturation", cmap='gray')
        dynamic_subplot.imshow(self.visuals['pixel_scores'], "Scores", cmap='gray')
        dynamic_subplot.imshow(self.visuals['windows'], "Selected Windows")
        dynamic_subplot.imshow(self.visuals['masked_pixel_scores'], "Masking + Fitted Lines", cmap='gray')
        dynamic_subplot.modify_plot('plot', x_fit_left, y_fit)
        dynamic_subplot.modify_plot('plot', x_fit_right, y_fit)
        dynamic_subplot.modify_plot('set_xlim', 0, camera.img_width)
        dynamic_subplot.modify_plot('set_ylim', camera.img_height, 0)
        dynamic_subplot.imshow(self.visuals['highlighted_lane'], "Highlighted Lane")


if __name__ == '__main__':
    # Calibrate using checkerboard
    calibration_img_files = glob.glob('./camera_cal/*.jpg')
    lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]
    camera = DashboardCamera(calibration_img_files, chessboard_size=(9, 6), lane_shape=lane_shape)

    # Create lane finder
    lane_finder = LaneFinder(camera)

    if str(sys.argv[1]) == 'test':
        # Run pipeline on test images
        test_imgs = glob.glob('./test_images/*.jpg')
        for img_file in test_imgs[:]:
            img = plt.imread(img_file)
            lane_finder.plot_pipeline(img)

        # Show all plots
        plt.show()

    else:
        # Create video
        input_vid_file = str(sys.argv[1])
        output_vid_file = 'output_' + input_vid_file
        input_video = VideoFileClip(input_vid_file)
        output_video = input_video.fl_image(lane_finder.callback_func('windows'))
        output_video.write_videofile(output_vid_file, audio=False)
