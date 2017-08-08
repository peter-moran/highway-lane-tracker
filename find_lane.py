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


def threshold_lanes(overhead_img, base_threshold=50, thresh_window=411, subplots=None):
    # Mask the image
    mask = cv2.adaptiveThreshold(
        overhead_img,
        maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=thresh_window,
        C=base_threshold * -1)

    if subplots is not None:
        # Show binary image
        subplots.imshow(mask, "Masking Image", cmap='gray')

    overhead_img[mask == 0] = 0
    if subplots is not None:
        # Show binary image
        subplots.imshow(overhead_img, "Masked Image", cmap='gray')

    return overhead_img


def get_overhead_image(image, subplots=None):
    # Transform to overhead image
    dy, dx = image.shape[0:2]
    M_trans = get_overhead_transform(dx, dy)
    overhead_img = cv2.warpPerspective(image, M_trans, (dx, dy))
    if subplots is not None:
        # Show overhead image
        subplots.imshow(overhead_img, "Overhead Image", cmap='gray')
    return overhead_img


def find_lane(dashcam_img, cam_matrix, distortion_coeffs, subplots=None):
    # Undistort
    dashcam_img = cv2.undistort(dashcam_img, cam_matrix, distortion_coeffs, None, cam_matrix)
    if subplots is not None:
        # Show undistorted road
        subplots.imshow(dashcam_img, "Undistorted Road")

    # Change color space
    hls = cv2.cvtColor(dashcam_img, cv2.COLOR_BGR2HLS)
    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]

    # Mask lanes
    lightness_masked = threshold_lanes(lightness, subplots=subplots)
    saturation_masked = threshold_lanes(saturation, subplots=subplots)

    # Print out everything
    if subplots is not None:
        # Show lightness image
        subplots.imshow(lightness, "Lightness Image", cmap='gray')
        # Show saturation image
        subplots.imshow(saturation, "Saturation Image", cmap='gray')


class DynamicSubplot:
    def __init__(self, m, n):
        self.figure, self.plots = plt.subplots(m, n)
        self.plots = self.plots.flatten()
        self.curr_plot = 0

    def imshow(self, img, title, **kwargs):
        self.plots[self.curr_plot].imshow(img, **kwargs)
        self.plots[self.curr_plot].set_title(title)
        self.curr_plot += 1


if __name__ == '__main__':
    # Calibrate using checkerboard
    calib_imgs = glob.glob('./camera_cal/*.jpg')
    example_img = cv2.imread(calib_imgs[0])
    img_size = (example_img.shape[1], example_img.shape[0])

    objpoints, imgpoints = find_object_img_points(calib_imgs, 9, 6)
    camera_matrix, dist_coeffs = calibrate(objpoints, imgpoints, img_size)

    # Show example undistorted checkerboard
    example_undistorted = cv2.undistort(example_img, camera_matrix, dist_coeffs, None, camera_matrix)

    plt.subplot(1, 2, 1)
    plt.imshow(example_img)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(example_undistorted)
    plt.title("Undistorted Image")
    plt.savefig('output_images/test_undist.jpg', bbox_inches='tight')

    # Run pipeline on single image
    test_imgs = glob.glob('./test_images/*.jpg')
    for imgf in test_imgs[:]:
        img = plt.imread(imgf)
        subplots = DynamicSubplot(3, 3)
        find_lane(img, camera_matrix, dist_coeffs, subplots=subplots)
    plt.show()
