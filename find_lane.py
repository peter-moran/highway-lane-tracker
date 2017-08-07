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


def getOverheadTransform(dx, dy):
    assert (dy, dx) == (720, 1280), "Unexpected image size."
    # Define points
    top_left = (614, 435)
    top_right = (668, 435)
    bottom_left = (295, 665)
    bottom_right = (1022, 665)
    source = np.float32([top_left, top_right, bottom_right, bottom_left])
    destination = np.float32([(bottom_left[0], 0), (bottom_right[0], 0),
                              (bottom_right[0], dy), (bottom_left[0], dy)])
    M_trans = cv2.getPerspectiveTransform(source, destination)
    return M_trans


def find_lane(img, cam_matrix, distortion_coeffs, verbose=True):
    # Undistort
    img = cv2.undistort(img, cam_matrix, distortion_coeffs, None, cam_matrix)
    if verbose:
        # Show example undistorted road
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.title("Undistorted Road")

    # Get overhead image
    dy, dx = img.shape[0:2]
    M_trans = getOverheadTransform(dx, dy)
    overhead = cv2.warpPerspective(img, M_trans, (dx, dy))
    if verbose:
        # Show example overhead image
        plt.subplot(2, 1, 2)
        plt.imshow(overhead)
        plt.title("Overhead Image")


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

    # Run pipeline on single image
    test_imgs = glob.glob('./test_images/*.jpg')
    for imgf in test_imgs[:2]:
        img = plt.imread(imgf)
        find_lane(img, camera_matrix, dist_coeffs)
    plt.show()
