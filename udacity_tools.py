"""
Helper functions provided by Udacity or based on code from their 'Self Driving Car Term 1' Course, 
with some minor changes.
"""
import cv2
import numpy as np


def overlay_centroids(img, window_centroids, window_height, window_width):
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros((img.shape[0], img.shape[1]))
        r_points = np.zeros((img.shape[0], img.shape[1]))

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, img, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = np.array(cv2.merge((img, img, img)), np.uint8)  # making 3 color channels
        return cv2.addWeighted(img, 1, template, 0.5, 0.0)  # overlay the original image with window results


def window_mask(width, height, img_ref, center, level):
    """
    Creates a rectangular mask centered along x axis at `center` and y axis at `level`*`height` from the bottom of the
    image.
    """
    output = np.zeros((img_ref.shape[0], img_ref.shape[1]))
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output
