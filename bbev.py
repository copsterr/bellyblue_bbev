"""
Bellyblue Bird's Eye View (BBEV) Library
"""

import cv2 as cv
import numpy as np

# constant range for hsv ranging
HSV_RED_LOWER   = (0, 87, 219)
HSV_RED_UPPER   = (111, 255, 255)
HSV_GREEN_LOWER = (21, 43, 0)
HSV_GREEN_UPPER = (110, 255, 150)
HSV_BLUE_LOWER  = (3, 120, 190)
HSV_BLUE_UPPER  = (130, 255, 255)


def hsv_range(hsv_src, lower, upper):
    """
        Create an hsv segmentation correlated to its lowerbound and upperbound
        params
            hsv_src: hsv color channel image
            lower: tuple of color channel (h, s, v) to indicate the lowerbound
            lower: tuple of color channel (h, s, v) to indicate the upperbound
        returns
            Binary Masking Image
    """
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    return mask


def detect(dest, src, title="", color=(0, 0, 0), draw_contours=True, draw_center=True):
    """
        * This function will be edited for detecting multiple targets *

        Detect contours of a given src. This function is usually used to detect
        a circle. You can provide title for the detected object, setting draw
        contours and its center to turn on and off.
        params
            dest: destination image
            src: binary image to detect contours
            title: title of the detected object
            color: color of the contours, title and center
            draw_contours, draw_center: to draw contours and center respectively
        return
            destination image that's been drawn
            center of the detected circle
            radius of the detected circle
    """
    _img, contours, _hier = cv.findContours(src.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ((cx, cy), radius) = cv.minEnclosingCircle(contours[0])
    cx = round(cx)
    cy = round(cy)
#    M = cv.moments(contours[0])
#    cx = int(M['m10']/M['m00'])
#    cy = int(M['m01']/M['m00'])
    center = (round(cx), round(cy))
    
    if draw_contours:
        dest = cv.drawContours(dest, contours, 0, color, 2)
    if title != "":
        dest = cv.putText(dest, title, (cx + 10, cy - 10), cv.FONT_HERSHEY_SIMPLEX , 0.5, color)
    if draw_center:
        dest	= cv.circle(dest, center, 1, color, 3)
    
    return dest, center, round(radius)


def find_slope(src, center1, center2):
    # convert cartesian
    center1_car = (center1[0], src.shape[1] - center1[1])
    center2_car = (center2[0], src.shape[1] - center2[1])
    
    # find slope
    slope = (center1_car[1] - center2_car[1]) / (center1_car[0] - center2_car[0])
    
    return slope


