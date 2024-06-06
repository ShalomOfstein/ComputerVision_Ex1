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
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    return 318845500


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    if representation == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if representation == 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to float and normalize to range [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # Convert the image using the imReadAndConvert function
    image = imReadAndConvert(filename, representation)
    # Display the image using plt.imshow
    plt.figure()

    if representation == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    plt.axis('off')  # Hide axes
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]])
    # Reshape image to (height*width, 3)
    h, w, c = imgRGB.shape
    imRGB_reshaped = imgRGB.reshape((h * w, 3))

    # Convert image to YIQ
    imgYIQ = np.dot(imRGB_reshaped, RGB2YIQ.T)
    imgYIQ = imgYIQ.reshape((h, w, 3))
    return imgYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    YIQ2RGB = np.array([[1.0, 0.956, 0.621],
                        [1.0, -0.272, -0.647],
                        [1.0, -1.106, 1.703]])
    # Reshape image to (height*width, 3)
    h, w, c = imgYIQ.shape
    imYIQ_reshaped = imgYIQ.reshape((h * w, 3))

    # Convert image to RGB
    imgRGB = np.dot(imYIQ_reshaped, YIQ2RGB.T)
    imgRGB = imgRGB.reshape((h, w, 3))
    return imgRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # Check if the image is in RGB
    is_rgb = len(imgOrig.shape) == 3 and imgOrig.shape[2] == 3

    if is_rgb:
        # Convert RGB to YIQ
        imYIQ = transformRGB2YIQ(imgOrig)
        # Only work on the Y channel
        Y_channel = imYIQ[:, :, 0]

    # in the case where the image is in grayscale
    else:
        # Only work on the Y channel
        Y_channel = imgOrig


    # Normalize Y_channel to [0, 255]
    Y_channel_norm = (Y_channel * 255).astype(np.uint8)

    # Calculate histogram
    histOrg, bins = np.histogram(Y_channel_norm.flatten(), bins=256, range=[0, 256])

    # Calculate cumulative sum
    cumsum = np.cumsum(histOrg)

    # Normalize cumulative sum
    cumsum_normalized = cumsum / cumsum[-1]

    # Create Look Up Table (LUT)
    LUT = (cumsum_normalized * 255).astype(np.uint8)

    # Apply LUT to the original image
    Y_channel_eq = LUT[Y_channel_norm]

    # Normalize the equalized Y channel back to [0, 1]
    Y_channel_eq = Y_channel_eq.astype(np.float32) / 255.0

    if is_rgb:
        # Replace the Y channel with the equalized one
        imYIQ[:, :, 0] = Y_channel_eq
        # Convert YIQ to RGB
        imEq = transformYIQ2RGB(imYIQ)
    else:

        imEq = Y_channel_eq

    # Calculate histogram of the equalized image
    histEq, _ = np.histogram(Y_channel_eq.flatten() * 255, bins=256, range=[0, 256])

    return imEq, histOrg, histEq



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
