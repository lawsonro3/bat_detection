# Import useful libraries
import cv2
import os

# Set up read and write paths
category = 'insects'
parentpath_read = '/Users/icunitz/Desktop/analysis/vids/%s' % category
parentpath_write = '/Users/icunitz/Desktop/analysis/frames/%s/' % category
extension = '.avi'

for file in os.listdir(parentpath_read):
    if file.endswith(extension):
        readpath = os.path.join(parentpath_read, file)
        writepath = os.path.join(parentpath_write, file)
        
        vidcap = cv2.VideoCapture(readpath)
        
        success,image = vidcap.read()
        count = 0
        
        success = True

        if not os.path.exists(writepath):
            os.makedirs(writepath)
        while success:
            writefile = os.path.join(writepath, "%s_frame%d.jpg" % (file[:-4], count))
            cv2.imwrite(writefile, image)
            success, image = vidcap.read()
            # print ('Read a new frame: ', success)
            count += 1
        
        print ('Done: ' + writepath)
