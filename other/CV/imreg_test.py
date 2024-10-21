# -*- coding: utf-8 -*-
# imreg_examples.py

"""Imreg examples with detailed output and visualization."""

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import imreg_dft as ird
import cv2

# Read images
im1 = cv2.imread('00000.jpg', cv2.IMREAD_GRAYSCALE)
im0 = cv2.imread('11111.jpg', cv2.IMREAD_GRAYSCALE)

# Check and adjust shape if necessary
if im0.shape != im1.shape:
    im1 = cv2.resize(im1, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_LINEAR)

# Perform similarity transformation
result = ird.similarity(im0, im1, numiter=1)

# Ensure transformation was successful
if "timg" in result:
    # Get the transformed image
    transformed_image = result['timg']

    # Print detailed transformation parameters
    print("Transformation Parameters:")
    print(f"Scale: {result['scale']}")
    print(f"Angle (degrees): {result['angle']}")
    print(f"Translation Vector (Y, X): {result['tvec']}")
    print(f"Success: {result['success']}")

    # Display the transformed image using OpenCV
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Destroy all the windows opened by OpenCV
else:
    print("Transformation was not successful.")
