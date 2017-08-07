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


def find_obj_img_point_pairs(image_fnames, chess_rows, chess_cols):
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


if __name__ == '__main__':
    # Calibrate using checkerboard
    calib_imgs = glob.glob('./camera_cal/*.jpg')
    example_img = cv2.imread(calib_imgs[0])
    img_size = (example_img.shape[1], example_img.shape[0])

    objpoints, imgpoints = find_obj_img_point_pairs(calib_imgs, 9, 6)
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

    # Show example undistorted road
    test_imgs = glob.glob('./test_images/*.jpg')
    img = plt.imread(test_imgs[0])
    img = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
    plt.figure()
    plt.imshow(img)
    plt.title("Undistorted Road")

    # Get overhead transformation
    dy, dx = img.shape[0:2]
    assert (dy, dx) == (720, 1280), "Unexpected image size."
    SOURCE = np.float32([(278, 670), (616, 437), (662, 437), (1025, 670)])
    DESTIN = np.float32([(278, dy), (278, 0), (1025, 0), (1025, dy)])
    M_trans = cv2.getPerspectiveTransform(SOURCE, DESTIN)

    # Show example overhead image
    overhead = cv2.warpPerspective(img, M_trans, (dx, dy))
    plt.figure()
    plt.imshow(overhead)
    plt.title("Overhead Image")
    plt.show()