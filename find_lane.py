#!/usr/bin/env python
"""
Finds lane lines and their curvature from dashcam video.

Author: Peter Moran
Created: 8/1/2017
"""
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import symfit
from scipy.ndimage.filters import convolve as center_convolve

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


def find_object_img_points(image_fnames, chess_rows, chess_cols):
    # Create object and image point pairings
    chess_corners = np.zeros((chess_cols * chess_rows, 3), np.float32)
    chess_corners[:, :2] = np.mgrid[0:chess_rows, 0:chess_cols].T.reshape(-1, 2)
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    for fname in image_fnames:
        # Load images
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found, img_corners = cv2.findChessboardCorners(gray, (chess_rows, chess_cols), None)

        # If found, save object points, image points
        if found:
            objpoints.append(chess_corners)
            imgpoints.append(img_corners)

    return objpoints, imgpoints


def calibrate(objpoints, imgpoints, img_size):
    """
    :return: The computed (camera matrix, distortion coefficients).
    """
    sucess, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    if not sucess:
        return None
    return camera_matrix, dist_coeffs


def threshold_lanes(image, base_threshold=50, thresh_window=411):
    # Mask the image
    binary = cv2.adaptiveThreshold(
        image,
        maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=thresh_window,
        C=base_threshold * -1)

    return binary


def get_overhead_transform(dx, dy):
    assert (dy, dx) == (720, 1280), "Unexpected image size."
    # Define points
    top_left = (584, 458)
    top_right = (701, 458)
    bottom_left = (295, 665)
    bottom_right = (1022, 665)
    source = np.float32([top_left, top_right, bottom_right, bottom_left])
    destination = np.float32([(bottom_left[0], 0), (bottom_right[0], 0),
                              (bottom_right[0], dy), (bottom_left[0], dy)])
    M_trans = cv2.getPerspectiveTransform(source, destination)
    return M_trans


def transform_to_overhead(image):
    # Transform to overhead image
    dy, dx = image.shape[0:2]
    M_trans = get_overhead_transform(dx, dy)
    overhead_img = cv2.warpPerspective(image, M_trans, (dx, dy))

    return overhead_img


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

    # Determine the location of the polynomial fit line for each row of the image
    y = np.linspace(0, n_rows - 1, num=n_rows)  # to cover y-range of image
    x_left_fit = fit.value(a) * y ** 2 + fit.value(b) * y + fit.value(x0_left)
    x_right_fit = fit.value(a) * y ** 2 + fit.value(b) * y + fit.value(x0_right)
    return y, x_left_fit, x_right_fit

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


def find_lane_in_frame(dashcam_img, cam_matrix, distortion_coeffs, dynamic_subplot=None):
    n_rows, n_cols = dashcam_img.shape[:2]
    # Undistort
    undistorted_img = cv2.undistort(dashcam_img, cam_matrix, distortion_coeffs, None, cam_matrix)

    # Change color space
    hls = cv2.cvtColor(dashcam_img, cv2.COLOR_BGR2HLS)
    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]

    # Transform
    transformed_lightness = transform_to_overhead(lightness)
    transformed_saturation = transform_to_overhead(saturation)

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
    y, left_fit_x, right_fit_x = fit_parallel_polynomials(get_nonzero_pixel_locations(left_line_masked),
                                                          get_nonzero_pixel_locations(right_line_masked),
                                                          n_rows)


    # Calculate radius of curvature
    left_curvature = find_curvature(left_line_masked, y_eval=n_rows - 1)
    right_curvature = find_curvature(right_line_masked, y_eval=n_rows - 1)
    print(left_curvature, 'm', right_curvature, 'm')

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
        dynamic_subplot.modify_plot('plot', left_fit_x, y)
        dynamic_subplot.modify_plot('plot', right_fit_x, y)
        dynamic_subplot.modify_plot('set_xlim', 0, n_cols)
        dynamic_subplot.modify_plot('set_ylim', n_rows, 0)


if __name__ == '__main__':
    # Calibrate using checkerboard
    calib_imgs = glob.glob('./camera_cal/*.jpg')
    example_img = cv2.imread(calib_imgs[0])
    img_size = (example_img.shape[1], example_img.shape[0])
    objpoints, imgpoints = find_object_img_points(calib_imgs, 9, 6)
    camera_matrix, dist_coeffs = calibrate(objpoints, imgpoints, img_size)

    # Run pipeline on test images
    test_imgs = glob.glob('./test_images/*.jpg')
    for imgf in test_imgs[:]:
        subplots = DynamicSubplot(3, 4)
        img = plt.imread(imgf)
        find_lane_in_frame(img, camera_matrix, dist_coeffs, dynamic_subplot=subplots)

    # Show all plots
    plt.show()
