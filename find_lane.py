#!/usr/bin/env python
"""
Finds lane lines and their curvature from camera input_video.

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
from windows import Window, filter_window_list, window_image, sliding_window_update

# Import moviepy and install ffmpeg if needed.
REGULATION_LANE_WIDTH = 3.7
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
    def __init__(self, chessboard_img_fnames, chessboard_size, lane_shape, scale_correction=(30 / 720, 3.7 / 700)):
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
        self.y_m_per_pix = scale_correction[0]
        self.x_m_per_pix = scale_correction[1]

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
    def __init__(self, cam: DashboardCamera, window_shape=(80, 61), search_width=300, max_frozen_dur=15):
        self.camera = cam

        # Create windows
        self.search_margin = search_width / 2
        self.windows_left = []
        self.windows_right = []
        for level in range(cam.img_height // window_shape[0]):
            x_init_l = cam.img_width / 4
            x_init_r = cam.img_width / 4 * 3
            self.windows_left.append(Window(level, window_shape, cam.img_size, x_init_l, max_frozen_dur))
            self.windows_right.append(Window(level, window_shape, cam.img_size, x_init_r, max_frozen_dur))

        # State
        self.last_fit_vals = None
        self.last_masked_pixel_scores = [np.zeros(cam.img_size), np.zeros(cam.img_size)]
        for i in range(cam.img_height):
            self.last_masked_pixel_scores[0][i, cam.img_width // 4] = 1
            self.last_masked_pixel_scores[1][i, (cam.img_width // 4) * 3] = 1

        # Initialize visuals to empty images
        VIZ_OPTIONS = ('dash_undistorted', 'overhead', 'saturation', 'saturation_binary', 'lightness',
                       'lightness_binary', 'pixel_scores', 'windows_raw', 'highlighted_lane')
        self.visuals = {name: None for name in VIZ_OPTIONS}
        self.__viz_options = None
        self.__viz_dependencies = {'windows_raw': ['pixel_scores'], 'windows_filtered': ['pixel_scores'],
                                   'presentation': ['highlighted_lane']}

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
        sliding_window_update(self.windows_left, pixel_scores, margin=self.search_margin, mode='left')
        sliding_window_update(self.windows_right, pixel_scores, margin=self.search_margin, mode='right')

        # TODO: Do something if not enough windows to fit

        # Filter window positions
        win_left_valid, argvalid_l = filter_window_list(self.windows_left, include_frozen=True, include_dropped=False)
        win_right_valid, argvalid_r = filter_window_list(self.windows_right, include_frozen=True, include_dropped=False)
        fit_vals = self.fit_lanes(zip(*[window.pos_xy() for window in win_left_valid]),
                                  zip(*[window.pos_xy() for window in win_right_valid]))

        # Find a safe region to apply the polynomial fit over. We don't want to extrapolate the shorter lane's extent.
        short_line_max_ndx = min(argvalid_l[-1], argvalid_r[-1])

        # Determine the location of the polynomial fit line for each row of the image
        y_fit = np.array(range(self.windows_left[short_line_max_ndx].y_begin, self.windows_left[0].y_end))
        x_fit_left = fit_vals['al'] * y_fit ** 2 + fit_vals['bl'] * y_fit + fit_vals['x0l']
        x_fit_right = fit_vals['ar'] * y_fit ** 2 + fit_vals['br'] * y_fit + fit_vals['x0r']

        # Calculate radius of curvature
        curve_rad = self.calc_curvature(win_left_valid)

        # Calculate position in lane.
        img_center = camera.img_width / 2
        position_prcnt = np.interp(img_center, [x_fit_left[-1], x_fit_right[-1]], [0, 1])

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
                         img_proc_func=lambda img: self.viz_lane(img, self.camera, x_fit_left, x_fit_right, y_fit))
        self.save_visual('presentation', self.visuals['highlighted_lane'],
                         img_proc_func=lambda img: self.viz_presentation(img, position_prcnt, curve_rad))

        return y_fit, x_fit_left, x_fit_right

    def score_pixels(self, img) -> np.ndarray:
        settings = [{'name': 'lab_b', 'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 150},
                    {'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 220},
                    {'name': 'lightness', 'cspace': 'HLS', 'channel': 1, 'clipLimit': 2.0, 'threshold': 210}]

        scores = np.zeros(img.shape[0:2])
        for params in settings:
            # Change color space
            color_t = getattr(cv2, 'COLOR_RGB2{}'.format(params['cspace']))
            gray = cv2.cvtColor(img, color_t)[:, :, params['channel']]

            # Threshold to binary
            clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
            norm_img = clahe.apply(gray)
            ret, binary = cv2.threshold(norm_img, params['threshold'], 255, cv2.THRESH_BINARY)
            scores += binary

            # Save images
            self.save_visual(params['name'], gray)
            self.save_visual(params['name'] + '_binary', binary)

        return scores.astype('uint8')

    def fit_lanes(self, points_left, points_right, fit_globally=False):
        xl, yl = points_left
        xr, yr = points_right

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
            fit_vals = {'ar': fit.value(a), 'al': fit.value(a), 'bl': fit.value(b), 'br': fit.value(b),
                        'x0l': fit.value(x0_left), 'x0r': fit.value(x0_right)}

        else:
            x, y = symfit.variables('x, y')
            a, b, x0 = symfit.parameters('a, b, x0')

            model = symfit.Model({
                x: a * y ** 2 + b * y + x0,
            })

            # Apply fit on left
            fit = symfit.Fit(model, x=xl, y=yl)
            fit = fit.execute()
            fit_vals = {'al': fit.value(a), 'bl': fit.value(b), 'x0l': fit.value(x0)}

            # Apply fit on right
            fit = symfit.Fit(model, x=xr, y=yr)
            fit = fit.execute()
            fit_vals.update({'ar': fit.value(a), 'br': fit.value(b), 'x0r': fit.value(x0)})

        return fit_vals

    def calc_curvature(self, windows: List[Window]):
        """From Udacity"""
        x, y = zip(*[window.pos_xy() for window in windows])
        x = np.array(x)
        y = np.array(y)
        fit_cr = np.polyfit(y * camera.y_m_per_pix, x * camera.x_m_per_pix, 2)
        y_eval = np.max(y)
        return ((1 + (2 * fit_cr[0] * y_eval * camera.y_m_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

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

    def viz_lane(self, undist_img, camera, left_fit_x, right_fit_x, fit_y):
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

    def viz_presentation(self, lane_img, position_prcnt, curve_rad):
        presentation_img = np.copy(lane_img)
        world_position = position_prcnt * REGULATION_LANE_WIDTH

        # Show position
        line_start = (10, 100)
        line_len = 300
        cv2.putText(presentation_img, "Position", org=(0, 50), fontScale=2, thickness=3,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA, color=(255, 255, 255))
        cv2.line(presentation_img, color=(255, 255, 255), thickness=2,
                 pt1=(line_start[0], line_start[1]),
                 pt2=(line_start[0] + line_len, line_start[1]))
        cv2.circle(presentation_img, center=(line_start[0] + int(position_prcnt * line_len), line_start[1]), radius=8,
                   color=(255, 255, 255))
        cv2.putText(presentation_img, '{:.2f} m'.format(world_position), fontScale=1, thickness=1,
                    org=(line_start[0] + int(position_prcnt * line_len) + 5, line_start[1] + 35),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), lineType=cv2.LINE_AA)

        # Show radius of curvature
        cv2.putText(presentation_img, "Curvature = {:>4.0f} m".format(curve_rad), org=(0, 200), fontScale=1,
                    thickness=2,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), lineType=cv2.LINE_AA)

        return presentation_img

    def viz_windows(self, img, mode):
        if mode == 'filtered':
            lw_img = window_image(self.windows_left, 'x_filtered', color=(0, 0, 255))
            rw_img = window_image(self.windows_right, 'x_filtered', color=(0, 0, 255))
        elif mode == 'raw':
            lw_img = window_image(self.windows_left, 'x_measured', color=(0, 255, 0))
            rw_img = window_image(self.windows_right, 'x_measured', color=(0, 255, 0))
        else:
            raise Exception('mode is not valid')
        combined = lw_img + rw_img
        return cv2.addWeighted(img, 1, combined, 0.5, 0)

    def viz_pipeline(self, img):
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
        self.find_lines(img, [visual])
        return self.visuals[visual]

    def viz_callback(self, visual='presentation'):
        return lambda img: self.viz_find_lines(img, visual=visual)


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
            lane_finder.viz_pipeline(img)

        # Show all plots
        plt.show()

    else:
        # Video options
        input_vid_file = str(sys.argv[1])
        visual = str(sys.argv[2]) if argc >= 3 else 'presentation'
        output_vid_file = str(sys.argv[3]) if argc >= 4 else 'output_' + input_vid_file

        # Create video
        lane_finder = LaneFinder(camera)
        input_video = VideoFileClip(input_vid_file)
        output_video = input_video.fl_image(lane_finder.viz_callback(visual))
        output_video.write_videofile(output_vid_file, audio=False)
