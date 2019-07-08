'''
Just a test script, not finished
'''

# Import useful libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up read path
file = '2016-07-30_014634'
readpath = '/Users/icunitz/Desktop/bat_detection/frames/' + file
directory = os.fsencode(readpath)

framenumbers_ = []

for filename in os.listdir(directory):
    framenumber = str(filename).lstrip('b\'frame')
    framenumber = int(framenumber.rstrip('.jpg\''))

    framenumbers_.apppend(framenumber)

    if str(filename).endswith('.jpg\''):
        print (framenumber)
        #img = cv2.imread(readpath + filename)
    else:
        print ("Tough luck")
