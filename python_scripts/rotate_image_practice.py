# Import useful libraries
import cv2
import numpy as np
# from scipy import signal
# from skimage.measure import compare_ssim as ssim
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits import mplot3d
# import os # Not currently using - figure out how to effectively use os.path.join

img = cv2.imread('/Users/icunitz/Desktop/bat_detection/frames/2016-07-30_014634/frame45.jpg', cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape[:2]

a_deg = 110

a_rad = a_deg * (np.pi / 180)
if a_deg <= 90:
        r = int(rows*np.cos(a_rad) + cols*np.sin(a_rad))
        c = int(cols*np.cos(a_rad) + rows*np.sin(a_rad))
        print ('Positive')
else:
        r = int(cols*np.cos(a_rad - np.pi/2) + rows*np.sin(a_rad - np.pi/2))
        c = int(rows*np.cos(a_rad - np.pi/2) + cols*np.sin(a_rad - np.pi/2))
        print ('Negative')
M = cv2.getRotationMatrix2D((cols//2, rows//2), a_deg, 1.0)
M[0,2] += (c - cols) / 2
M[1,2] += (r - rows) / 2

# rows, cols = img.shape[:2]
# 
# a_deg = 90
# a_rad = a_deg * (np.pi / 180)
# r = int(rows*np.cos(a_rad) + cols*np.sin(a_rad))
# c = int(cols*np.cos(a_rad) + rows*np.sin(a_rad))
# 
# M = cv2.getRotationMatrix2D((cols/2, rows/2), a_deg, 1)
# M[0,2] += (c - cols) / 2
# M[1,2] += (r - rows) / 2
dst = cv2.warpAffine(img, M, (c, r))

cv2.imshow('Rotated', dst)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
