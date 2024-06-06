"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import cv2
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE, imReadAndConvert


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global image_bgr  # Make image_bgr accessible to the trackbar callback
    image = imReadAndConvert(img_path, rep)
    if rep == 1:
        image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.namedWindow('Gamma Correction')
    # Create a trackbar for gamma adjustment
    cv2.createTrackbar('Gamma', 'Gamma Correction', 100, 200, gamma_correction)
    gamma_correction(100)  # Initialize with gamma=1.0

    # Display the image and wait for user interaction
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def gamma_correction(val):
    gamma = val / 100.0  # Trackbar value divided by 100 to get float value in range 0 to 2
    gamma_corrected = adjust_gamma(image_bgr, gamma)
    cv2.imshow('Gamma Correction', gamma_corrected)


def adjust_gamma(image, gamma):
    gamma = max(gamma, 0.01)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
