#!/usr/bin/env python
"""
Finds and highlights lane lines in dashboard camera videos.
See README.md for more info.

Author: Peter Moran
Created: 8/1/2017
"""
import glob
import sys
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import symfit
from imageio.core import NeedDownloadError

from dynamic_subplot import DynamicSubplot
from windows import Window, filter_window_list, joint_sliding_window_update, window_image

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

REGULATION_LANE_WIDTH = 3.7


class DashboardCamera:
    def __init__(self, chessboard_img_fnames, chessboard_size, lane_shape, scale_correction=(30 / 720, 3.7 / 700)):
        """
        Handles camera calibration, distortion correction, perspective warping, and maintains various camera properties.

        :param chessboard_img_fnames: List of file names of the chessboard calibration images.
        :param chessboard_size: Size of the calibration chessboard.
        :param lane_shape: Pixel locations of the four corners describing the profile of the lane lines on a straight
        road. Should be ordered clockwise, from the top left.
        :param scale_correction: Constants y_m_per_pix and x_m_per_pix describing the number of meters per pixel in the
        overhead transformation of the road.
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
        self.y_m_per_pix = scale_correction[0]
        self.x_m_per_pix = scale_correction[1]

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

    def warp_to_overhead(self, undistorted_img):
        """
        Transforms this camera's images from the dashboard perspective to an overhead perspective.

        Note: Make sure to undistort first.
        """
        return cv2.warpPerspective(undistorted_img, self.overhead_transform, dsize=(self.img_width, self.img_height))

    def warp_to_dashboard(self, overhead_img):
        """
        Transforms this camera's images from an overhead perspective back to the dashboard perspective.
        """
        return cv2.warpPerspective(overhead_img, self.inverse_overhead_transform,
                                   dsize=(self.img_width, self.img_height))


class LaneFinder:
    def __init__(self, cam: DashboardCamera, window_shape=(80, 61), search_margin=200, max_frozen_dur=15):
        """
        The primary interface for fitting lane lines. Used to initialize lane finding with desired settings and provides
        extensive options for visualization.

        :param cam: A calibrated camera.
        :param window_shape: Desired (window height, window width) that the sliding window search will use.
        :param search_margin: The maximum pixels of movement allowed between each level of windows.
        :param max_frozen_dur: The maximum amount of frames a window can continue to be used when frozen (eg when not
        found or when measurements are uncertain).
        """
        self.camera = cam

        # Create windows
        self.windows_left = []
        self.windows_right = []
        for level in range(cam.img_height // window_shape[0]):
            x_init_l = cam.img_width / 4
            x_init_r = cam.img_width / 4 * 3
            self.windows_left.append(Window(level, window_shape, cam.img_size, x_init_l, max_frozen_dur))
            self.windows_right.append(Window(level, window_shape, cam.img_size, x_init_r, max_frozen_dur))
        self.search_margin = search_margin

        # Initialize visuals
        VIZ_OPTIONS = ('dash_undistorted', 'overhead', 'lab_b', 'lab_b_binary', 'lightness', 'lightness_binary',
                       'value', 'value_binary', 'pixel_scores', 'windows_raw', 'windows_filtered', 'highlighted_lane',
                       'presentation')
        self.visuals = {name: None for name in VIZ_OPTIONS}  # Storage location of visualization images
        self.__viz_desired = None  # The visuals we want to save
        self.__viz_dependencies = {'windows_raw': ['pixel_scores'],  # Dependencies of visuals on other visuals
                                   'windows_filtered': ['pixel_scores'],
                                   'presentation': ['highlighted_lane', 'overhead', 'windows_raw', 'windows_filtered',
                                                    'pixel_scores']}

    def find_lines(self, img_dashboard, visuals=None):
        """
        Primary function for fitting lane lines in an image.

        Visualization options include:
        'dash_undistorted', 'overhead', 'lab_b', 'lab_b_binary', 'lightness', 'lightness_binary', 'value',
        'value_binary', 'pixel_scores', 'windows_raw', 'highlighted_lane', 'presentation'

        :param img_dashboard: Raw dashboard camera image taken by the calibrated `self.camera`.
        :param visuals: A list of visuals you would like to be saved to `self.visuals`.
        :return: A set of points along the left and right lane line: y_fit, x_fit_left, x_fit_right.
        """
        # Account for visualization options
        if visuals is None:
            visuals = ['highlighted_lane']
        self.__viz_desired = self.viz_fix_dependencies(visuals)

        # Undistort and transform to overhead view
        img_dash_undistorted = self.camera.undistort(img_dashboard)
        img_overhead = self.camera.warp_to_overhead(img_dash_undistorted)

        # Score pixels
        pixel_scores = self.score_pixels(img_overhead)

        # Select windows
        joint_sliding_window_update(self.windows_left, self.windows_right, pixel_scores, margin=self.search_margin)

        # Filter window positions
        win_left_valid, argvalid_l = filter_window_list(self.windows_left, remove_frozen=False, remove_dropped=True)
        win_right_valid, argvalid_r = filter_window_list(self.windows_right, remove_frozen=False, remove_dropped=True)

        assert len(win_left_valid) >= 3 and len(win_right_valid) >= 3, 'Not enough valid windows to create a fit.'
        # TODO: Do something if not enough windows to fit. Most likely fall back on old measurements.

        # Apply fit
        fit_vals = self.fit_lanes(zip(*[window.pos_xy() for window in win_left_valid]),
                                  zip(*[window.pos_xy() for window in win_right_valid]))

        # Find a safe region to apply the polynomial fit over. We don't want to extrapolate the shorter lane's extent.
        short_line_max_ndx = min(argvalid_l[-1], argvalid_r[-1])

        # Determine the location of the polynomial fit line for each row of the image
        y_fit = np.array(range(self.windows_left[short_line_max_ndx].y_begin, self.windows_left[0].y_end))
        x_fit_left = fit_vals['al'] * y_fit ** 2 + fit_vals['bl'] * y_fit + fit_vals['x0l']
        x_fit_right = fit_vals['ar'] * y_fit ** 2 + fit_vals['br'] * y_fit + fit_vals['x0r']

        # Calculate radius of curvature
        curve_radius = self.calc_curvature(win_left_valid)

        # Calculate position in lane.
        img_center = camera.img_width / 2
        lane_position_prcnt = np.interp(img_center, [x_fit_left[-1], x_fit_right[-1]], [0, 1])
        lane_position = lane_position_prcnt * REGULATION_LANE_WIDTH

        # Log visuals
        self.viz_save('dash_undistorted', img_dash_undistorted)
        self.viz_save('overhead', img_overhead)
        self.viz_save('pixel_scores', pixel_scores)
        self.viz_save('windows_raw', self.visuals['pixel_scores'],
                      img_proc_func=lambda img: self.viz_windows(img, 'raw'))
        self.viz_save('windows_filtered', self.visuals['pixel_scores'],
                      img_proc_func=lambda img: self.viz_windows(img, 'filtered'))
        self.viz_save('highlighted_lane', img_dash_undistorted,
                      img_proc_func=lambda img: viz_lane(img, self.camera, x_fit_left, x_fit_right, y_fit))
        self.viz_save('presentation', self.visuals['highlighted_lane'],
                      img_proc_func=lambda img: self.viz_presentation(img, lane_position, curve_radius))

        return y_fit, x_fit_left, x_fit_right

    def score_pixels(self, img) -> np.ndarray:
        """
        Takes a road image and returns an image where pixel intensity maps to likelihood of it being part of the lane.

        Each pixel gets its own score, stored as pixel intensity. An intensity of zero means it is not from the lane,
        and a higher score means higher confidence of being from the lane.

        :param img: an image of a road, typically from an overhead perspective.
        :return: The score image.
        """
        # Settings to run thresholding operations on
        settings = [{'name': 'lab_b', 'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 150},
                    {'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 220},
                    {'name': 'lightness', 'cspace': 'HLS', 'channel': 1, 'clipLimit': 2.0, 'threshold': 210}]

        # Perform binary thresholding according to each setting and combine them into one image.
        scores = np.zeros(img.shape[0:2]).astype('uint8')
        for params in settings:
            # Change color space
            color_t = getattr(cv2, 'COLOR_RGB2{}'.format(params['cspace']))
            gray = cv2.cvtColor(img, color_t)[:, :, params['channel']]

            # Normalize regions of the image using CLAHE
            clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
            norm_img = clahe.apply(gray)

            # Threshold to binary
            ret, binary = cv2.threshold(norm_img, params['threshold'], 1, cv2.THRESH_BINARY)

            scores += binary

            # Save images
            self.viz_save(params['name'], gray)
            self.viz_save(params['name'] + '_binary', binary)

        return cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)

    def fit_lanes(self, points_left, points_right, fit_globally=False) -> dict:
        """
        Applies and returns a polynomial fit for given points along the left and right lane line.

        Both lanes are described by a second order polynomial x(y) = ay^2 + by + x0. In the `fit_globally` case,
        a and b are modeled as equal, making the lines perfectly parallel. Otherwise, each line is fit independent of
        the other. The parameters of the model are returned in a dictionary with keys 'al', 'bl', 'x0l' for the left
        lane parameters and 'ar', 'br', 'x0r' for the right lane.

        :param points_left: Two lists of the x and y positions along the left lane line.
        :param points_right: Two lists of the x and y positions along the right lane line.
        :param fit_globally: Set True to use the global, parallel line fit model. In practice this does not allays work.
        :return: fit_vals, a dictionary containing the fitting parameters for the left and right lane as above.
        """
        xl, yl = points_left
        xr, yr = points_right

        fit_vals = {}
        if fit_globally:
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
            fit_vals.update({'ar': fit.value(a), 'al': fit.value(a), 'bl': fit.value(b), 'br': fit.value(b),
                             'x0l': fit.value(x0_left), 'x0r': fit.value(x0_right)})

        else:
            # Fit lines independently
            x, y = symfit.variables('x, y')
            a, b, x0 = symfit.parameters('a, b, x0')

            model = symfit.Model({
                x: a * y ** 2 + b * y + x0,
            })

            # Apply fit on left
            fit = symfit.Fit(model, x=xl, y=yl)
            fit = fit.execute()
            fit_vals.update({'al': fit.value(a), 'bl': fit.value(b), 'x0l': fit.value(x0)})

            # Apply fit on right
            fit = symfit.Fit(model, x=xr, y=yr)
            fit = fit.execute()
            fit_vals.update({'ar': fit.value(a), 'br': fit.value(b), 'x0r': fit.value(x0)})

        return fit_vals

    def calc_curvature(self, windows: List[Window]):
        """
        Given a list of Windows along a lane, returns an estimated radius of curvature of the lane.

        Radius of curvature is found by transforming the x,y positions of the windows to the world space, applying
        a simple polynomial fit, and then using the fit values to find curvature.

        :param windows: A List of Windows along a single lane.
        :return: Radius of curvature, in meters.
        """
        x, y = zip(*[window.pos_xy() for window in windows])
        x = np.array(x)
        y = np.array(y)
        fit_cr = np.polyfit(y * camera.y_m_per_pix, x * camera.x_m_per_pix, 2)
        y_eval = np.max(y)
        return ((1 + (2 * fit_cr[0] * y_eval * camera.y_m_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

    def viz_save(self, name, image, img_proc_func=None):
        """
        Conditionally processes and saves the given image if this LaneFinder has been set to save it.

        Specifically: If `name` has been set to be visualized (ie is in `self.__viz_desired`), process the `image`
        with the `img_proc_func` and save it to `self.visuals`.

        :param name: Name of the visual. Should match a key from `self.visuals`.
        :param image: The image to process and save.
        :param img_proc_func: A single parameter function, which `image` will be passed to if it is to be saved.
        """
        if 'all' not in self.__viz_desired and name not in self.__viz_desired:
            return  # Don't save this image
        if img_proc_func is not None:
            image = img_proc_func(image)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise Exception('Image is not 3 channels or could not be converted to 3 channels. Cannot use.')
        self.visuals[name] = image

    def viz_fix_dependencies(self, viz_names: list):
        """
        Ensures that any dependencies of the visuals in `viz_names` are also saved.

        Each name in `viz_names` should match a key from `self.visuals`. Dependencies are defined in
        `self.__viz_dependencies`.
        """
        for viz_opt in self.__viz_dependencies:
            if viz_opt in viz_names:
                for dependency in self.__viz_dependencies[viz_opt]:
                    viz_names.append(dependency)
        return viz_names

    def viz_presentation(self, lane_img, lane_position, curve_radius, lane_width=REGULATION_LANE_WIDTH):
        """
        Processes the image for presentation purposes, with extra information displayed over the lane image.

        :param lane_img: Image with the lane highlighted.
        :param lane_position: The position of the car relative to the left lane, in meters.
        :param curve_radius: The radius of curvature of the lane, in meters.
        :param lane_width: The width of the lane, in meters.
        :return: The presentation_img visual.
        """
        presentation_img = np.copy(lane_img)
        lane_position_prcnt = lane_position / lane_width

        # Show overlays
        overhead_img = cv2.resize(self.visuals['overhead'], None, fx=1 / 3.0, fy=1 / 3.0)
        titled_overlay(presentation_img, overhead_img, 'Overhead (not to scale)', (0, 0))
        overhead_img = cv2.resize(self.visuals['windows_raw'], None, fx=1 / 3.0, fy=1 / 3.0)
        titled_overlay(presentation_img, overhead_img, 'Raw Lane Detection', (presentation_img.shape[1] // 3, 0))
        overhead_img = cv2.resize(self.visuals['windows_filtered'], None, fx=1 / 3.0, fy=1 / 3.0)
        titled_overlay(presentation_img, overhead_img, 'Filtered Lane Detection',
                       (presentation_img.shape[1] // 3 * 2, 0))

        # Show position
        x_text_start, y_text_start = (10, 350)
        line_start = (10 + x_text_start, 40 + y_text_start)
        line_len = 300
        cv2.putText(presentation_img, "Position", org=(x_text_start, y_text_start), fontScale=2, thickness=3,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA, color=(255, 255, 255))
        cv2.line(presentation_img, color=(255, 255, 255), thickness=2,
                 pt1=(line_start[0], line_start[1]),
                 pt2=(line_start[0] + line_len, line_start[1]))
        cv2.circle(presentation_img, center=(line_start[0] + int(lane_position_prcnt * line_len), line_start[1]),
                   radius=8,
                   color=(255, 255, 255))
        cv2.putText(presentation_img, '{:.2f} m'.format(lane_position), fontScale=1, thickness=1,
                    org=(line_start[0] + int(lane_position_prcnt * line_len) + 5, line_start[1] + 35),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), lineType=cv2.LINE_AA)

        # Show radius of curvature
        cv2.putText(presentation_img, "Curvature = {:>4.0f} m".format(curve_radius), fontScale=1, thickness=2,
                    org=(x_text_start, 130 + y_text_start), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255),
                    lineType=cv2.LINE_AA)

        return presentation_img

    def viz_windows(self, score_img, mode):
        """Displays the position of the windows over a score image."""
        if mode == 'filtered':
            lw_img = window_image(self.windows_left, 'x_filtered', color=(0, 255, 0))
            rw_img = window_image(self.windows_right, 'x_filtered', color=(0, 255, 0))
        elif mode == 'raw':
            color = (255, 0, 0)
            win_left_detected, arg = filter_window_list(self.windows_left, False, False, remove_undetected=True)
            win_right_detected, arg = filter_window_list(self.windows_right, False, False, remove_undetected=True)
            lw_img = window_image(win_left_detected, 'x_measured', color, color, color)
            rw_img = window_image(win_right_detected, 'x_measured', color, color, color)
        else:
            raise Exception('mode is not valid')
        combined = lw_img + rw_img
        return cv2.addWeighted(score_img, 1, combined, 0.5, 0)

    def viz_pipeline(self, img):
        """Displays most of the steps in the image processing pipeline for a single image."""
        y_fit, x_fit_left, x_fit_right = self.find_lines(img, ['all'])
        dynamic_subplot = DynamicSubplot(3, 4)
        dynamic_subplot.imshow(self.visuals['dash_undistorted'], "Undistorted Road")
        dynamic_subplot.imshow(self.visuals['overhead'], "Overhead", cmap='gray')
        dynamic_subplot.imshow(self.visuals['lightness'], "Lightness", cmap='gray')
        dynamic_subplot.imshow(self.visuals['lightness_binary'], "Binary Lightness", cmap='gray')
        dynamic_subplot.skip_plot()
        dynamic_subplot.skip_plot()
        dynamic_subplot.imshow(self.visuals['value'], "Value", cmap='gray')
        dynamic_subplot.imshow(self.visuals['value_binary'], "Binary Value", cmap='gray')
        dynamic_subplot.imshow(self.visuals['pixel_scores'], "Scores", cmap='gray')
        dynamic_subplot.imshow(self.visuals['windows_raw'], "Selected Windows")
        dynamic_subplot.imshow(self.visuals['windows_raw'], "Fitted Lines", cmap='gray')
        dynamic_subplot.modify_plot('plot', x_fit_left, y_fit)
        dynamic_subplot.modify_plot('plot', x_fit_right, y_fit)
        dynamic_subplot.modify_plot('set_xlim', 0, camera.img_width)
        dynamic_subplot.modify_plot('set_ylim', camera.img_height, 0)
        dynamic_subplot.imshow(self.visuals['highlighted_lane'], "Highlighted Lane")

    def viz_find_lines(self, img, visual='presentation'):
        """Runs `self.find_lines()` for a single visual, and returns it."""
        self.find_lines(img, [visual])
        return self.visuals[visual]

    def viz_callback(self, visual='presentation'):
        """
        Returns a callback function that takes an image, runs `self.find_lines()` and returns the requested visual.
        """
        return lambda img: self.viz_find_lines(img, visual=visual)


def viz_lane(undist_img, camera, left_fit_x, right_fit_x, fit_y):
    """
    Take an undistorted dashboard camera image and highlights the lane.

    Code from Udacity SDC-ND Term 1 course code.

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


def titled_overlay(image, overlay, title, org, border_thickness=2):
    """Puts a title above the overlay image and places it in image at the given origin."""
    # Place title
    title_img = np.ones((50, overlay.shape[1], 3)).astype('uint8') * 255
    cv2.putText(title_img, title, org=(10, 35), fontScale=1, thickness=2, color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA)

    # Add title to overlay
    overlay = np.concatenate((title_img, overlay), axis=0)

    # Add border to overlay
    overlay[:border_thickness, :, :] = 255
    overlay[-border_thickness:, :, :] = 255
    overlay[:, :border_thickness, :] = 255
    overlay[:, -border_thickness:, :] = 255

    # Place overlay onto image
    x_offset, y_offset = org
    image[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay

    # Add a white border


if __name__ == '__main__':
    argc = len(sys.argv)

    # Calibrate using checkerboard
    calibration_img_files = glob.glob('./data/camera_cal/*.jpg')
    lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]
    camera = DashboardCamera(calibration_img_files, chessboard_size=(9, 6), lane_shape=lane_shape)

    if str(sys.argv[1]) == 'test':
        # Run pipeline on test images
        test_imgs = glob.glob('./data/test_images/*.jpg')
        for img_file in test_imgs[:]:
            lane_finder = LaneFinder(camera)  # need new instance per image to prevent smoothing
            img = plt.imread(img_file)
            lane_finder.viz_pipeline(img)

        # Show all plots
        plt.show()

    else:
        # Video options
        input_vid_file = str(sys.argv[1])
        visual = str(sys.argv[2]) if argc >= 3 else 'presentation'
        if argc >= 4:
            output_vid_file = str(sys.argv[3])
        else:
            name, ext = input_vid_file.split('/')[-1].split('.')
            name += '_' + visual
            output_vid_file = './output/' + name + '.' + ext

        # Create video
        lane_finder = LaneFinder(camera)
        input_video = VideoFileClip(input_vid_file)
        output_video = input_video.fl_image(lane_finder.viz_callback(visual))
        output_video.write_videofile(output_vid_file, audio=False)
