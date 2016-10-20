# -*- coding: utf-8 -*-
"""
acknowledgements for RGB <-> YCbCr convertion method:
http://www.equasys.de/colorconversion.html
https://en.wikipedia.org/wiki/YCbCr

"""
import sys
import scipy as sp
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from skimage import data, io, filters, util
from IPython.display import Image  # Image(filename='cb.png')
from scipy.ndimage import convolve  # kernel convolutions in one line

if(len(sys.argv) > 2):
    image_name = sys.argv[1]
    gamma_value = float(sys.argv[2])
else:
    image_name = "naive_linear.png"
    gamma_value = 2.2

# Loading a RAW image
img_raw = util.img_as_float(io.imread("./"+image_name))
###############################################################################
#               Gamma correction
###############################################################################
def rgb2ycbcr(rgb_pixel):
    conv_matrix = np.array([[0.299,  0.587, 0.114],
                            [-0.169, -0.331, 0.500],
                            [0.500, -0.419, 0.081]])
    range_correction = np.array([0,128,128])
    converted_pixel = conv_matrix.dot(rgb_pixel*255.0)
    converted_pixel = np.add(range_correction, converted_pixel)
    return converted_pixel/255.0

def ycbcr2rgb(ycbcr_pixel):
    conv_matrix = np.array([[0.299,  0.587, 0.114],
                            [-0.169, -0.331, 0.500],
                            [0.500, -0.419, 0.081]])
    range_correction = np.array([0,128,128])
    converted_pixel = conv_matrix.dot(rgb_pixel*255.0)
    converted_pixel = np.add(range_correction, converted_pixel)
    return converted_pixel/255.0

def gamma_correction(pixel, gamma_value):
    return ((pixel)**(1.0/gamma_value))

vec_gamma = np.vectorize(gamma_correction)
result = vec_gamma(img_raw, gamma_value)
result = np.clip(result, 0.0, 1.0)
###############################################################################
# Saving gamma-corrected image
io.imsave("./gamma_correctd_"+str(gamma_value)+"_"+image_name, util.img_as_ubyte(result))
###############################################################################
