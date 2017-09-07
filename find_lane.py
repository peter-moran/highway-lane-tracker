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
from imageio.core import NeedDownloadError

from dynamic_subplot import DynamicSubplot
from windows import WindowHandler

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
    def __init__(self, camera: DashboardCamera, window_shape=(120, 61), meas_variance=100, process_variance=1):
        self.camera = camera

        # Window parameters
        self.left_windows = \
            WindowHandler('left', window_shape, meas_variance, process_variance, camera.img_size)
        self.right_windows = \
            WindowHandler('right', window_shape, meas_variance, process_variance, camera.img_size)

        # State
        self.last_fit_vals = None
        self.last_masked_pixel_scores = [np.zeros(camera.img_size), np.zeros(camera.img_size)]
        for i in range(camera.img_height):
            self.last_masked_pixel_scores[0][i, camera.img_width // 4] = 1
            self.last_masked_pixel_scores[1][i, (camera.img_width // 4) * 3] = 1

        # Initialize visuals to empty images
        VIZ_OPTIONS = ('dash_undistorted', 'overhead', 'saturation', 'saturation_binary', 'lightness',
                       'lightness_binary', 'pixel_scores', 'windows_raw', 'masked_pixel_scores', 'highlighted_lane')
        self.visuals = {name: None for name in VIZ_OPTIONS}
        self.__viz_options = None
        self.__viz_dependencies = {'windows_raw': ['pixel_scores'], 'windows_filtered': ['pixel_scores']}

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

        # Select windows_raw
        self.left_windows.update(pixel_scores)
        self.right_windows.update(pixel_scores)

        # Filter window positions
        fit_vals = self.fit_lanes(zip(*self.left_windows.get_positions('filtered')),
                                  zip(*self.right_windows.get_positions('filtered')))

        # Determine the location of the polynomial fit line for each row of the image
        y_fit = np.linspace(0, camera.img_height - 1, num=camera.img_height).flatten()  # to cover y-range of image
        x_fit_left = fit_vals['a'] * y_fit ** 2 + fit_vals['b'] * y_fit + fit_vals['x0_left']
        x_fit_right = fit_vals['a'] * y_fit ** 2 + fit_vals['b'] * y_fit + fit_vals['x0_right']

        # TODO: Calculate radius of curvature

        # Log visuals
        self.save_visual('dash_undistorted', img_dash_undistorted)
        self.save_visual('overhead', img_overhead)
        self.save_visual('pixel_scores', pixel_scores,
                         img_proc_func=lambda img: cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
        self.save_visual('windows_raw', self.visuals['pixel_scores'],
                         img_proc_func=lambda img: self.viz_windows(img, 'raw'))
        self.save_visual('windows_filtered', self.visuals['pixel_scores'],
                         img_proc_func=lambda img: self.viz_windows(img, 'filtered'))
        self.save_visual('highlighted_lane', img_dash_undistorted,
                         img_proc_func=lambda img: self.draw_lane(img, self.camera, x_fit_left, x_fit_right, y_fit))

        return y_fit, x_fit_left, x_fit_right

    def score_pixels(self, img) -> np.ndarray:
        # Change color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img_lightness = hls[:, :, 1]
        img_saturation = hls[:, :, 2]

        # Threshold for lanes
        img_binary_L = threshold_gaussian(img_lightness)
        img_binary_S = threshold_gaussian(img_saturation)

        # Stack binary images
        lightness_score = cv2.normalize(img_binary_L, None, 0, 1, cv2.NORM_MINMAX)
        saturation_score = cv2.normalize(img_binary_S, None, 0, 1, cv2.NORM_MINMAX)

        # Log visuals
        self.save_visual('saturation', img_lightness)
        self.save_visual('saturation_binary', img_binary_S)
        self.save_visual('lightness', img_saturation)
        self.save_visual('lightness_binary', img_binary_S)

        return lightness_score + saturation_score

    def fit_lanes(self, points_left, points_right):
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
        fit_vals = {'a': fit.value(a), 'b': fit.value(b), 'x0_left': fit.value(x0_left),
                    'x0_right': fit.value(x0_right)}

        return fit_vals

    def draw_lane(self, undist_img, camera, left_fit_x, right_fit_x, fit_y):
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
        for viz_opt in self.__viz_dependencies:
            if viz_opt in viz_options:
                for dependency in self.__viz_dependencies[viz_opt]:
                    viz_options.append(dependency)
        return viz_options

    def viz_windows(self, img, mode):
        if mode == 'filtered':
            lw_img = self.left_windows.img_windows_filtered()
            rw_img = self.right_windows.img_windows_filtered()
        elif mode == 'raw':
            lw_img = self.left_windows.img_windows_raw()
            rw_img = self.right_windows.img_windows_raw()
        else:
            raise Exception('mode is not valid')
        combined = lw_img + rw_img
        return cv2.addWeighted(img, 1, combined, 0.5, 0)

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
        dynamic_subplot.imshow(self.visuals['windows_raw'], "Selected Windows")
        dynamic_subplot.imshow(self.visuals['masked_pixel_scores'], "Masking + Fitted Lines", cmap='gray')
        dynamic_subplot.modify_plot('plot', x_fit_left, y_fit)
        dynamic_subplot.modify_plot('plot', x_fit_right, y_fit)
        dynamic_subplot.modify_plot('set_xlim', 0, camera.img_width)
        dynamic_subplot.modify_plot('set_ylim', camera.img_height, 0)
        dynamic_subplot.imshow(self.visuals['highlighted_lane'], "Highlighted Lane")


def threshold_gaussian(image, base_threshold=50, thresh_window=411):
    # Mask the image
    binary = cv2.adaptiveThreshold(
        image,
        maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=thresh_window,
        C=base_threshold * -1)

    return binary


if __name__ == '__main__':
    # Calibrate using checkerboard
    calibration_img_files = glob.glob('./camera_cal/*.jpg')
    lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]
    camera = DashboardCamera(calibration_img_files, chessboard_size=(9, 6), lane_shape=lane_shape)

    argc = len(sys.argv)
    if str(sys.argv[1]) == 'test':
        # Run pipeline on test images
        test_imgs = glob.glob('./test_images/*.jpg')
        for img_file in test_imgs[:]:
            lane_finder = LaneFinder(camera)  # need new instance per image to prevent smoothing
            img = plt.imread(img_file)
            lane_finder.plot_pipeline(img)

        # Show all plots
        plt.show()

    else:
        # Video options
        input_vid_file = str(sys.argv[1])
        visual = str(sys.argv[2]) if argc >= 3 else 'highlighted_lane'
        output_vid_file = str(sys.argv[3]) if argc >= 4 else 'output_' + input_vid_file

        # Create video
        lane_finder = LaneFinder(camera)
        input_video = VideoFileClip(input_vid_file)
        output_video = input_video.fl_image(lane_finder.callback_func(visual))
        output_video.write_videofile(output_vid_file, audio=False)
