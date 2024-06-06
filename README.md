# Computer Vision Project: Image Processing and Enhancement

## Overview

This project demonstrates various image processing techniques including histogram equalization, image quantization, and gamma correction using Python and OpenCV.

## Requirements

- **Python Version:** 3.11
- **Platform:** Windows

## Installation

1. Install Python 3.11 from [python.org](https://www.python.org/downloads/).
2. Install the required libraries using pip:
   ```sh
   pip install numpy matplotlib opencv-python
   ```
## Files

### 1. `ex1_main.py`

This is the main script that runs various image processing demonstrations including histogram equalization, image quantization, and gamma correction.

#### Functions:

- **`histEqDemo(img_path: str, rep: int)`**: Demonstrates histogram equalization on an image.
- **`quantDemo(img_path: str, rep: int)`**: Demonstrates image quantization.
- **`main()`**: The main function that runs the demos for different image processing techniques.

### 2. `ex1_utils.py`

Contains utility functions for reading and converting images, displaying images, color space transformations, histogram equalization, and image quantization.

#### Functions:

- **`myID() -> int`**: Returns the student ID.
- **`imReadAndConvert(filename: str, representation: int) -> np.ndarray`**: Reads an image and converts it to the specified representation (grayscale or RGB).
- **`imDisplay(filename: str, representation: int)`**: Reads and displays an image in the specified representation.
- **`transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray`**: Converts an RGB image to YIQ color space.
- **`transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray`**: Converts a YIQ image to RGB color space.
- **`hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray)`**: Equalizes the histogram of an image.
- **`quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float])`**: Quantizes an image into a specified number of colors.

### 3. `gamma.py`

Contains functions for performing gamma correction on an image with a GUI slider to adjust the gamma value.

#### Functions:

- **`gammaDisplay(img_path: str, rep: int)`**: Displays a GUI for gamma correction with a slider to adjust gamma value.
- **`gamma_correction(val)`**: Callback function for the trackbar that applies gamma correction based on the slider value.
- **`adjust_gamma(image, gamma)`**: Adjusts the gamma of the image.

## Usage

1. **Running the Main Script**:
   ```sh
   python ex1_main.py
   ```
   This script will:
- Display the original image in grayscale and RGB.
- Convert the image between RGB and YIQ color spaces.
- Demonstrate histogram equalization on the image.
- Demonstrate image quantization on the image.
- Display a GUI for gamma correction with adjustable gamma value.

2. **Gamma Correction**:
   To run only the gamma correction demonstration:
   ```sh
   python gamma.py
   ```
## Notes

- Ensure the image file (`beach.jpg`) is in the same directory as the scripts.
- The gamma correction GUI uses OpenCV's `createTrackbar` for adjusting gamma values from 0 to 2 with a resolution of 0.01.
- The `ex1_utils.py` file contains core functions used by the main script for image processing tasks.

