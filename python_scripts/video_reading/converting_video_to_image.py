import cv2
import os

print(cv2.__version__)

# Set up read and write paths
objecttype = 'bats'
distance = 'close'
filename = 'bat_112309'
extension = '.avi'
readpath = '/Users/icunitz/Desktop/clear_background/%s/%s/%s%s' % (objecttype, distance, filename, extension)
vidcap = cv2.VideoCapture(readpath)
writepath = '/Users/icunitz/Desktop/bat_detection/frames/clear_background/%s/%s/%s' % (objecttype, distance, filename)

# Make new folder for the video's frames if one doesn't exist already
if not os.path.exists(writepath):
    os.makedirs(writepath)

success,image = vidcap.read()
count = 0
success = True

while success:
    cv2.imwrite(os.path.join(writepath, "frame%d.jpg" % count), image)
    success, image = vidcap.read()
    print ('Read a new frame: ', success)
    count += 1