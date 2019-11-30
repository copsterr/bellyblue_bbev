import cv2 as cv
import numpy as np
from bbev import *

img = cv.imread("./srcimg/test_img0.jpg")
img = cv.GaussianBlur(img, (5, 5), 0) # apply blur to filter noise
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # convert BGR to HSV

# find head
mask = hsv_range(hsv, HSV_RED_LOWER, HSV_RED_UPPER)
out, center_head, radius_head = detect(img, mask, "Head")

# find tail
mask = hsv_range(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
out, center_tail, radius_tail = detect(img, mask, "Tail")

# find target
mask = hsv_range(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
out, center_target, radius_target = detect(img, mask, "Target")

# draw a line from head to tail
out	= cv.arrowedLine(img, center_tail, center_head, 0, 3)
  

cv.imshow("", out)
cv.waitKey(0)
cv.destroyAllWindows()
