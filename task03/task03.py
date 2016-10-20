# -*- coding: utf-8 -*-
"""
acknowledgements for gamma correction algorithm:
https://en.wikipedia.org/wiki/Gamma_correction
http://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-6-gamma-correction/


"""
import sys
import numpy as np
from skimage import io, util


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
def gamma_correction(pixel, gamma_value):
    return ((pixel)**(1.0/gamma_value))

vec_gamma = np.vectorize(gamma_correction)
result = vec_gamma(img_raw, gamma_value)
result = np.clip(result, 0.0, 1.0)
###############################################################################
# Saving gamma-corrected image
io.imsave("./gamma_correctd_"+str(gamma_value)+"_"+image_name, util.img_as_ubyte(result))
###############################################################################
