import cv2
import os

print(cv2.__version__)

filename = '2016-10-02_234015'
extension = '.avi'
vidcap = cv2.VideoCapture('/Users/icunitz/Desktop/bat_detection/bat_vids/' + filename + extension)
writepath = '/Users/icunitz/Desktop/bat_detection/frames/' + filename

if not os.path.exists(writepath):
    os.mkdir(writepath)

success,image = vidcap.read()
count = 0
success = True

while success:
    cv2.imwrite(os.path.join(writepath, "frame%d.jpg" % count), image)
    success, image = vidcap.read()
    print ('Read a new frame: ', success)
    count += 1
