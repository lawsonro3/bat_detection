# Import useful libraries
import cv2
import numpy as np
import matplotlib
import os

# Set up read path
file = '2016-07-30_014634'
frameno = '54'
extension = '.jpg'
readpath = '/Users/icunitz/Desktop/bat_detection/frames/' + file + '/frame' + frameno + extension

n = 20 # Half of length of square sides

# Open frame
img = cv2.imread(readpath)
cv2.imshow('Frame ' + frameno, img)

# Click event
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_location = [(x, y)]
        print (ref_location)

        #cv2.circle(np.float32(img), (x, y), (10, 10, 200), 10, 2)
        cv2.rectangle(np.float32(img), (x - n, y + n), (x + n, y - n), (0, 0, 0), 2)

cv2.setMouseCallback('Frame ' + frameno, click_event)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows


