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


def calibrate(image_fnames, chess_rows=9, chess_cols=6):
    """
    :param calibration_folder: folder holding `.jpg` chessboard calibration images.
    :return: The computed (camera matrix, distortion coefficients).
    """
    example_img = cv2.imread(image_fnames[0])
    img_size = (example_img.shape[1], example_img.shape[0])

    # Create object and image point pairings
    chess_corners = np.zeros((chess_cols * chess_rows, 3), np.float32)
    chess_corners[:, :2] = np.mgrid[0:chess_rows, 0:chess_cols].T.reshape(-1, 2)

    chesspoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    for fname in image_fnames:
        # Load images
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found, img_corners = cv2.findChessboardCorners(gray, (chess_rows, chess_cols), None)

        # If found, save object points, image points
        if found:
            chesspoints.append(chess_corners)
            imgpoints.append(img_corners)

    # Calibrate and return
    sucess, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chesspoints, imgpoints, img_size, None, None)
    if sucess:
        return (camera_matrix, dist_coeffs)
    else:
        return None


if __name__ == '__main__':
    # Calibrate using checkerboard
    calib_imgs = glob.glob('./camera_cal/*.jpg')
    camera_matrix, dist_coeffs = calibrate(calib_imgs, 9, 6)

    # Show undistort example
    example_img = cv2.imread(calib_imgs[0])
    example_undistorted = cv2.undistort(example_img, camera_matrix, dist_coeffs, None, camera_matrix)

    plt.subplot(1, 2, 1)
    plt.imshow(example_img)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(example_undistorted)
    plt.title("Undistorted Image")
    plt.savefig('output_images/test_undist.jpg', bbox_inches='tight')
    plt.show()
