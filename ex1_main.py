from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    cumsumEq = np.cumsum(histEq)
    plt.gray()
    plt.plot(range(256), cumsum, 'r')
    plt.plot(range(256), cumsumEq, 'g')

    # Display the images
    plt.figure()
    plt.imshow(np.clip(img, 0, 1))

    plt.figure()
    plt.imshow(np.clip(imgeq,0,1))
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()

    img_lst, err_lst = quantizeImage(img, 3, 20)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %f" % err_lst[0])
    print("Error last:\t %f" % err_lst[-1])
    plt.gray()
    plt.imshow(np.clip(img_lst[0], 0, 1))
    plt.figure()
    plt.imshow(np.clip(img_lst[-1], 0, 1))
    plt.imshow(np.clip(img, 0, 1))

    # f, ax = plt.subplots(1, 3)
    # ax[0].imshow(np.clip(img, 0, 1))
    # ax[1].imshow(np.clip(img_lst[0], 0, 1))
    # ax[2].imshow(np.clip(img_lst[-1], 0, 1))
    # plt.show()

    plt.figure()
    plt.plot(err_lst, 'r')
    plt.show()


def main():
    print("ID:", myID())
    img_path = 'beach.jpg'

    # Basic read and display
    imDisplay(img_path, LOAD_GRAY_SCALE)
    imDisplay(img_path, LOAD_RGB)

    # Convert Color spaces
    img = imReadAndConvert(img_path, LOAD_RGB)
    yiq_img = transformRGB2YIQ(img)
    rgb_img = transformYIQ2RGB(yiq_img)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(np.clip(img, 0, 1))
    ax[1].imshow(np.clip(yiq_img, 0, 1))
    ax[2].imshow(np.clip(rgb_img, 0, 1))
    plt.show()

    # Image histEq
    histEqDemo(img_path, LOAD_GRAY_SCALE)
    histEqDemo(img_path, LOAD_RGB)

    # Image Quantization
    quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB)

    # Gamma
    gammaDisplay(img_path, LOAD_GRAY_SCALE)
    gammaDisplay(img_path, LOAD_RGB)

if __name__ == '__main__':
    main()
