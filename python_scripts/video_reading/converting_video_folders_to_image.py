import cv2
import os

# print(cv2.__version__)

# Set up read and write paths
parentpath_read = '/Users/icunitz/Desktop/clear_background/'
parentpath_write = '/Users/icunitz/Desktop/bat_detection/frames/clear_background/'
extension = '.avi'

for subdir, dirs, files in os.walk(parentpath_read):
    for file in files:
        # print(os.path.join(parentpath_read, subdir, file))
        if file.endswith(extension):
            readpath = os.path.join(parentpath_read, subdir, file)
            print (readpath)
            writepath = os.path.join(parentpath_write, readpath[40:-4])
            print (writepath)
            
            vidcap = cv2.VideoCapture(readpath)
            
            success,image = vidcap.read()
            count = 0

            if not os.path.exists(writepath):
                os.makedirs(writepath)
                success = True
            else:
                success = False

            while success:
                cv2.imwrite(os.path.join(writepath, "frame%d.jpg" % count), image)
                success, image = vidcap.read()
                print ('Read a new frame: ', success)
                count += 1

    #for distance in objecttype:
    #   for filename in distance:

    #       readpath = '/Users/icunitz/Desktop/clear_background/%s/%s/%s%s' % (objecttype, distance, filename, extension)
    #       writepath = '/Users/icunitz/Desktop/bat_detection/frames/clear_background/%s/%s/%s' % (objecttype, distance, filename)

    #       # Make new folder for the video's frames if one doesn't exist already
    #       

