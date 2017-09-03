#!/usr/bin/env python
"""
Finds lane lines and their curvature from dashcam input_video.

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
from scipy.ndimage.filters import convolve as center_convolve

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

from udacity_tools import overlay_centroids, window_mask


class DynamicSubplot:
    def __init__(self, m, n):
        self.figure, self.plots = plt.subplots(m, n)
        self.plots = self.plots.flatten()
        self.__curr_plot = -1

    def imshow(self, img, title, cmap=None):
        """Shows the image in the next plot."""
        self.next_subplot()
        self.plots[self.__curr_plot].imshow(img, cmap=cmap)
        self.plots[self.__curr_plot].set_title(title)

    def skip_plot(self):
        """Sets the plot to empty and advances to the next plot."""
        self.next_subplot()
        self.plots[self.__curr_plot].axis('off')

    def call(self, func_name, *args, **kwargs):
        self.next_subplot()
        func = getattr(self.plots[self.__curr_plot], func_name)
        func(*args, **kwargs)

    def modify_plot(self, func_name, *args, **kwargs):
        """Allows you to call any function on the current plot."""
        if self.__curr_plot == -1:
            raise IndexError("There is no plot to modify.")
        func = getattr(self.plots[self.__curr_plot], func_name)
        func(*args, **kwargs)

    def next_subplot(self, n=1):
        """Increments to the next plot."""
        self.__curr_plot += n
        if self.__curr_plot > len(self.plots):
            raise IndexError("You've gone too far forward. There are no more subplots.")

    def last_subplot(self, n=1):
        """Increments to the next plot."""
        self.__curr_plot -= n
        if self.__curr_plot < 0:
            raise IndexError("You've gone too far back. There are no more subplots.")


def threshold_lanes(image, base_threshold=50, thresh_window=411):
    # Mask the image
    binary = cv2.adaptiveThreshold(
        image,
        maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=thresh_window,
        C=base_threshold * -1)

    return binary


def find_window_centroids(image, window_width, window_height, margin):
    img_h, img_w = image.shape[0:2]
    window_lr_centroids = []  # Store the (left,right) window centroid positions per level

    # Find the start of the line, along the bottom of the image
    search_strip = image[2 * img_h // 3:, :]  # search bottom 1/3rd of image
    strip_scores = score_columns(search_strip, window_width)
    l_line_center = argmax_between(strip_scores, begin=0, end=img_w // 2)
    r_line_center = argmax_between(strip_scores, begin=img_w // 2, end=img_w)

    # Add what we found for the first layer
    window_lr_centroids.append((l_line_center, r_line_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, image.shape[0] // window_height):
        search_strip = image[img_h - (level + 1) * window_height:img_h - level * window_height, :]
        strip_scores = score_columns(search_strip, window_width)

        # Find the best left centroid nearby the centroid from the row below
        l_search_min = max(l_line_center - margin, 0)
        l_search_max = min(l_line_center + margin, img_w)
        l_max_ndx = argmax_between(strip_scores, l_search_min, l_search_max)

        # Find the best right centroid nearby the centroid from the row below
        r_search_min = max(r_line_center - margin, 0)
        r_search_max = min(r_line_center + margin, img_w)
        r_max_ndx = argmax_between(strip_scores, r_search_min, r_search_max)

        # Update predicted line center, unless there were no pixels in search region (ie max was zero).
        l_line_center = l_max_ndx if strip_scores[l_max_ndx] != 0 else l_line_center
        r_line_center = r_max_ndx if strip_scores[r_max_ndx] != 0 else r_line_center

        window_lr_centroids.append((l_line_center, r_line_center))

    return window_lr_centroids


def score_columns(image, window_width):
    assert window_width % 2 != 0, 'window_width must be odd'
    window = np.ones(window_width)
    col_sums = np.sum(image, axis=0)
    scores = center_convolve(col_sums, window, mode='constant')
    return scores


def argmax_between(arr: np.ndarray, begin: int, end: int) -> int:
    max_ndx = np.argmax(arr[begin:end]) + begin
    return max_ndx


def get_nonzero_pixel_locations(binary_img):
    """
    Returns the x, y locations of all nonzero pixels.
    """
    points = np.argwhere(binary_img)
    y, x = zip(*points)  # mapping r,c -> y,x format.
    return x, y


def fit_pixels(binary_img):
    """
    Returns the polynomial that fits the pixels in a binary image, as well as the (x,y) pairs for the polynomial
    it forms on the image.
    """
    # Calculate fit
    x, y = get_nonzero_pixel_locations(binary_img)
    fit = np.polyfit(y, x, 2)  # we want x according to y

    # Determine the location of the polynomial fit line for each row of the image
    y = np.linspace(0, binary_img.shape[1] - 1, num=binary_img.shape[1])  # to cover y-range of image
    x = fit[0] * y ** 2 + fit[1] * y + fit[2]

    return fit, x, y


def fit_parallel_polynomials(points_left, points_right, n_rows):
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
    fit_y = np.linspace(0, n_rows - 1, num=n_rows)  # to cover y-range of image
    x_left_fit = fit_vals['a'] * fit_y ** 2 + fit_vals['b'] * fit_y + fit_vals['x0_left']
    x_right_fit = fit_vals['a'] * fit_y ** 2 + fit_vals['b'] * fit_y + fit_vals['x0_right']
    return fit_y, x_left_fit, x_right_fit, fit_vals


def mask_with_centroids(img, centroids, window_width, window_height):
    if len(centroids) <= 0:
        return
    # Create a mask of all window areas
    mask = np.zeros_like(img)
    for level in range(0, len(centroids)):
        # Find the mask for this window
        this_windows_mask = window_mask(window_width, window_height, img, centroids[level], level)
        # Add it to our overall mask
        mask[(mask == 1) | ((this_windows_mask == 1))] = 1

    # Apply the mask
    masked_img = np.copy(img)
    masked_img[mask != 1] = 0
    return masked_img


def find_curvature(binary_img, y_eval):
    """Returns radius of curvature in meters."""
    Y_M_PER_PIX = 30 / 720  # meters per pixel in y dimension
    X_M_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

    # Fit in world space
    x, y = get_nonzero_pixel_locations(binary_img)
    fit_cr = np.polyfit(np.array(y) * Y_M_PER_PIX, np.array(x) * X_M_PER_PIX, 2)

    # Calculate the new radii of curvature
    curvature_rad = ((1 + (2 * fit_cr[0] * y_eval * Y_M_PER_PIX + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * fit_cr[0])

    return curvature_rad


def draw_lane(undist_img, camera, left_fit_x, right_fit_x, fit_y):
    """
    Take an undistorted dashboard camera image and highlights the lane.
    :param undist_img: An undistorted dashboard view image.
    :param camera: The DashboardCamera object for the camera the image was taken on.
    :param left_fit_x: the x values for the left line polynomial at the given y values.
    :param right_fit_x: the x values for the right line polynomial at the given y values.
    :param fit_y: the y values the left and right line x values were calculated at.
    :return: The undistorted image with the lane overlayed on top of it.
    """
    # Create an undist_img to draw the lines on
    lane_poly_overhead = np.zeros_like(undist_img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, fit_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, fit_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank undist_img
    cv2.fillPoly(lane_poly_overhead, np.int_([pts]), (0, 255, 0))

    # Warp back to original undist_img space
    lane_poly_dash = camera.warp_to_dashboard(lane_poly_overhead)

    # Combine the result with the original undist_img
    return cv2.addWeighted(undist_img, 1, lane_poly_dash, 0.3, 0)


def find_lane_in_frame(dashcam_img, camera, dynamic_subplot=None):
    # Undistort
    undistorted_img = camera.undistort(dashcam_img)

    # Change color space
    hls = cv2.cvtColor(dashcam_img, cv2.COLOR_BGR2HLS)
    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]

    # Transform
    transformed_lightness = camera.warp_to_overhead(lightness)
    transformed_saturation = camera.warp_to_overhead(saturation)

    # Threshold for lanes
    lightness_binary = threshold_lanes(transformed_lightness)
    saturation_binary = threshold_lanes(transformed_saturation)

    # Stack binary images
    combo_binary = lightness_binary + saturation_binary

    # Select lane lines
    window_width = 41
    window_height = 100
    margin = 50
    window_centroids = find_window_centroids(combo_binary, window_width, window_height, margin)

    # Mask out lane lines according to centroid windows
    left_centroids, right_centroids = zip(*window_centroids)
    left_line_masked = mask_with_centroids(combo_binary, left_centroids, window_width, window_height)
    right_line_masked = mask_with_centroids(combo_binary, right_centroids, window_width, window_height)

    # Fit lines along the pixels
    fit_y, left_fit_x, right_fit_x, fit_vals = fit_parallel_polynomials(get_nonzero_pixel_locations(left_line_masked),
                                                                        get_nonzero_pixel_locations(right_line_masked),
                                                                        camera.img_height)

    # Calculate radius of curvature
    left_curvature = find_curvature(left_line_masked, y_eval=camera.img_height - 1)
    right_curvature = find_curvature(right_line_masked, y_eval=camera.img_height - 1)

    # Show the lane in world space
    lane_img = draw_lane(undistorted_img, camera, left_fit_x, right_fit_x, fit_y)
    font_size, font_thickness = 2, 4
    cv2.putText(lane_img, "a = {: 10e}".format(fit_vals['a']), org=(0, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_size, color=(255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)
    cv2.putText(lane_img, "b = {: 10e}".format(fit_vals['b']), org=(0, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_size, color=(255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)
    cv2.putText(lane_img, "x0_left = {}".format(fit_vals['x0_left']), org=(0, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_size, color=(255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)
    cv2.putText(lane_img, "x0_right = {}".format(fit_vals['x0_right']), org=(0, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_size, color=(255, 255, 255), thickness=font_thickness, lineType=cv2.LINE_AA)

    # Print out everything
    if dynamic_subplot is not None:
        dynamic_subplot.imshow(undistorted_img, "Undistorted Road")
        dynamic_subplot.imshow(lightness, "Lightness Image", cmap='gray')
        dynamic_subplot.imshow(transformed_lightness, "Overhead Lightness", cmap='gray')
        dynamic_subplot.imshow(lightness_binary, "Binary Lightness", cmap='gray')
        dynamic_subplot.skip_plot()
        dynamic_subplot.imshow(saturation, "Saturation Image", cmap='gray')
        dynamic_subplot.imshow(transformed_saturation, "Overhead Saturation", cmap='gray')
        dynamic_subplot.imshow(saturation_binary, "Binary Saturation", cmap='gray')
        dynamic_subplot.imshow(combo_binary, "Binary Combined", cmap='gray')
        centroids_img = overlay_centroids(combo_binary, window_centroids, window_height, window_width)
        dynamic_subplot.imshow(centroids_img, "Centroids")
        dynamic_subplot.imshow(left_line_masked + right_line_masked, "Masked & Fitted Lines", cmap='gray')
        dynamic_subplot.modify_plot('plot', left_fit_x, fit_y)
        dynamic_subplot.modify_plot('plot', right_fit_x, fit_y)
        dynamic_subplot.modify_plot('set_xlim', 0, camera.img_width)
        dynamic_subplot.modify_plot('set_ylim', camera.img_height, 0)
        dynamic_subplot.imshow(lane_img, "Highlighted Lane")

    return lane_img


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
                                  (bottom_right[0], self.img_height), (bottom_left[0], self.img_height)])
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
        return cv2.warpPerspective(dashboard_image, self.overhead_transform, (self.img_width, self.img_height))

    def warp_to_dashboard(self, dashboard_image):
        """
        Transforms this camera's images from the dashboard perspective to an overhead perspective.
        :param dashboard_image: an image taken from the dashboard of the car. Aka a raw image.
        """
        return cv2.warpPerspective(dashboard_image, self.inverse_overhead_transform, (self.img_width, self.img_height))


if __name__ == '__main__':
    # Calibrate using checkerboard
    calibration_img_files = glob.glob('./camera_cal/*.jpg')
    lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]
    dashcam = DashboardCamera(calibration_img_files, chessboard_size=(9, 6), lane_shape=lane_shape)

    if str(sys.argv[1]) == 'test':
        # Run pipeline on test images
        test_imgs = glob.glob('./test_images/*.jpg')
        for img_file in test_imgs[:]:
            subplots = DynamicSubplot(3, 4)
            img = plt.imread(img_file)
            find_lane_in_frame(img, dashcam, dynamic_subplot=subplots)

        # Show all plots
        plt.show()

    if str(sys.argv[1]) == 'video':
        # Create video
        input_vid_file = 'project_video.mp4'
        output_vid_file = 'output_' + input_vid_file
        input_video = VideoFileClip(input_vid_file)
        output_video = input_video.fl_image(lambda image: find_lane_in_frame(image, dashcam))
        output_video.write_videofile(output_vid_file, audio=False)
