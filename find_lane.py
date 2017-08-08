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
    top_left = (584, 458)
    top_right = (701, 458)
    bottom_left = (295, 665)
    bottom_right = (1022, 665)
    source = np.float32([top_left, top_right, bottom_right, bottom_left])
    destination = np.float32([(bottom_left[0], 0), (bottom_right[0], 0),
                              (bottom_right[0], dy), (bottom_left[0], dy)])
    M_trans = cv2.getPerspectiveTransform(source, destination)
    return M_trans


def find_lane(img, cam_matrix, distortion_coeffs, verbose=True):
    ## Get an Overhead View
    # Undistort
    img = cv2.undistort(img, cam_matrix, distortion_coeffs, None, cam_matrix)
    if verbose:
        # Show undistorted road
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title("Undistorted Road")

    # Transform to overhead image
    dy, dx = img.shape[0:2]
    M_trans = getOverheadTransform(dx, dy)
    overhead_img = cv2.warpPerspective(img, M_trans, (dx, dy))
    if verbose:
        # Show overhead image
        plt.subplot(2, 3, 2)
        plt.imshow(overhead_img)
        plt.title("Overhead Image")

    ## Change color space
    hls = cv2.cvtColor(overhead_img, cv2.COLOR_BGR2HLS)
    lightness_img = hls[:, :, 2]
    if verbose:
        # Show lightness image
        plt.subplot(2, 3, 3)
        plt.imshow(lightness_img, cmap='gray')
        plt.title("Lightness Image")

    ## Get gradient
    gradmag, graddir = sobel_gradient(lightness_img, kernel_size=13)

    if verbose:
        # Show gradient image
        plt.subplot(2, 3, 4)
        plt.imshow(gradmag, cmap='gray')
        plt.title("Gradient Image")

    ## Thresholding
    # Set low pixels to zero
    threshold_low = 10
    clean_img = np.copy(gradmag)
    clean_img[gradmag < threshold_low] = 0.0

    # Set sparce pixel regions to zero
    clean_img = drop_sparce_regions(gradmag, kernel_size=(9, 9), min_average=5.0)

    if verbose:
        # Show cleaned image
        plt.subplot(2, 3, 5)
        plt.imshow(clean_img, cmap='gray')
        plt.title("Thresholding")

    ## Box filter
    box = box_filter(gradmag, kernel_size=(61, 61))

    if verbose:
        # Show box filter
        plt.subplot(2, 3, 6)
        plt.imshow(box, cmap='gray')
        plt.title("Normalized Box Filter")


def box_filter(img, kernel_size):
    box = np.float32(cv2.blur(img, ksize=kernel_size))
    cv2.normalize(box, box, 0, 255, cv2.NORM_MINMAX)
    return box


def drop_sparce_regions(img, kernel_size, min_average):
    """
    At each pixel, uses a kernel_size kernel to compute the average pixel value
    in that region. If the average is below min_average, the pixel is set to 0.
    """
    box = cv2.boxFilter(img, -1, kernel_size, normalize=True)
    box[box < min_average] = 0
    return box


def sobel_gradient(img, kernel_size):
    """
    :param img: image to return the gradient for.
    :param kernel_size: gradient kernel size.
    :return: magnitude image (normalized to 0-255) and direction image (-pi to pi).
    """
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = (sobel_x ** 2 + sobel_y ** 2) ** 0.5
    cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    direction = np.arctan2(sobel_y, sobel_x)
    assert isinstance(magnitude, np.ndarray)
    return magnitude, direction


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
    for imgf in test_imgs[:]:
        img = plt.imread(imgf)
        find_lane(img, camera_matrix, dist_coeffs)
    plt.show()
