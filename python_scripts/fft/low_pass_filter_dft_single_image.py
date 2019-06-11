import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
#img = cv2.imread('Users/Desktop/frames/2016-07-25_221513/frame360.jpg',0)  ## Opens up the image from another directory
path = os.getcwd()
file_extension = '.jpg'

for file in os.listdir(path):
    if file.endswith(file_extension):
        filename, separator, extension = file.partition('.')


        img = cv2.imread(file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_float32 = np.float32(img_gray)

        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = img_gray.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        ### High pass filter

        mask = np.ones((rows, cols, 2), np.uint8)
        r = 200
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0

        # Band pass filter
        # mask = np.zeros((rows, cols, 2), np.uint8)
        # r_out = 150
        # r_in = 100
        # center = [crow, ccol]
        # x, y = np.ogrid[:rows, :cols]
        # mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
        #                            ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
        # mask[mask_area] = 1

        # apply mask and inverse DFT
        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back_2 = cv2.magnitude(img_back[:, :, 0],img_back[:, :, 1])

        plt.gray()
        plt.subplot(121)
        plt.imshow(img, cmap = 'gray')
        plt.title('Input Image')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(img_back_2, cmap = 'gray')
        plt.title('Magnitude Spectrum')
        plt.xticks([])
        plt.yticks([])
        #plt.imshow(img_back_2, cmap = 'gray')
        pix_val = list(img_back)

        plt.show()
        plt.savefig(filename + 'dft.png')

        print('Completed ' + filename)
