import cv2
import numpy as np


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


def drop_sparce_regions(img, kernel_size, min_average):
    """
    At each pixel, uses a kernel_size kernel to compute the average pixel value
    in that region. If the average is below min_average, the pixel is set to 0.
    """
    box = cv2.boxFilter(img, -1, kernel_size, normalize=True)
    box[box < min_average] = 0
    return box


def box_filter(img, kernel_size):
    box = np.float32(cv2.blur(img, ksize=kernel_size))
    cv2.normalize(box, box, 0, 255, cv2.NORM_MINMAX)
    return box
