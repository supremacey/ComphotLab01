# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from skimage import io, util

# Loading a RAW image
img_raw = util.img_as_float(io.imread("./../lighthouse_RAW_noisy_sigma0.01.png"))

# Size of RAW image
(ydim, xdim) = img_raw.shape

# Creating array for each channel RGB
cr = np.zeros((ydim, xdim))
cg = np.zeros((ydim, xdim))
cb = np.zeros((ydim, xdim))

cr[0::2, 0::2] = img_raw[0::2, 0::2]
cg[1::2, 0::2] = img_raw[1::2, 0::2]
cg[0::2, 1::2] = img_raw[0::2, 1::2]
cb[1::2, 1::2] = img_raw[1::2, 1::2]

cr_demo = cr.copy()
cg_demo = cg.copy()
cb_demo = cb.copy()

###############################################################################
#                Demosaicking - linear interpolation
###############################################################################

# vertical red interpolation over upper and lower neighbour
for y in range(1,ydim,2):
    for x in range(0,xdim,2):
        cr_demo[y,x] += .5*cr_demo[max(y-1, 0), x]
        cr_demo[y,x] += .5*cr_demo[min(y+1, ydim-1), x]

# horizontal red interpolation over left and right neighbour
for y in range(0,ydim,2):
    for x in range(1,xdim,2):
        cr_demo[y,x] += .5*cr_demo[y, max(x-1,0)]
        cr_demo[y,x] += .5*cr_demo[y, min(x+1, xdim-1)]

# red channel middle interpolation
for y in range(1, ydim, 2):
    for x in range(1, xdim, 2):
        cr_demo[y,x] += .25*cr_demo[max(y-1, 0), max(x-1,0)] # left up
        cr_demo[y,x] += .25*cr_demo[max(y-1, 0), min(x+1,xdim-1)] # right up
        cr_demo[y,x] += .25*cr_demo[min(y+1,ydim-1), min(x+1,xdim-1)] # right down
        cr_demo[y,x] += .25*cr_demo[min(y+1,ydim-1), max(x-1,0)] # left down

# green channel
for y in range(0,ydim,2):
    for x in range(0,xdim,2):
            cg_demo[y,x] += .25*cg_demo[max(y-1,0),x]
            cg_demo[y,x] += .25*cg_demo[y,max(x-1,0)]
            cg_demo[y,x] += .25*cg_demo[y,min(x+1,xdim-1)]
            cg_demo[y,x] += .25*cg_demo[min(y+1,ydim-1),x]

for y in range(1,ydim,2):
    for x in range(1,xdim,2):
            cg_demo[y,x] += .25*cg_demo[max(y-1,0),x]
            cg_demo[y,x] += .25*cg_demo[y,max(x-1,0)]
            cg_demo[y,x] += .25*cg_demo[y,min(x+1,xdim-1)]
            cg_demo[y,x] += .25*cg_demo[min(y+1,ydim-1),x]

# blue channel middle interpolation
for y in range(0,ydim,2):
    for x in range(0,xdim,2):
        cb_demo[y,x] += .25*cb_demo[max(y-1, 0), max(x-1,0)] # left up
        cb_demo[y,x] += .25*cb_demo[max(y-1, 0), min(x+1,xdim-1)] # right up
        cb_demo[y,x] += .25*cb_demo[min(y+1,ydim-1), min(x+1,xdim-1)] # right down
        cb_demo[y,x] += .25*cb_demo[min(y+1,ydim-1), max(x-1,0)] # left down

# vertical blue interpolation over upper and lower neighbour
for y in range(0,ydim,2):
    for x in range(1,xdim,2):
        cb_demo[y,x] += .5*cb_demo[max(y-1, 0), x]
        cb_demo[y,x] += .5*cb_demo[min(y+1, ydim-1), x]

# horizontal blue interpolation over left and right neighbour
for y in range(1,ydim,2):
    for x in range(0,xdim,2):
        cb_demo[y,x] += .5*cb_demo[y, max(x-1,0)]
        cb_demo[y,x] += .5*cb_demo[y, min(x+1, xdim-1)]


# adding all 3 channels together
sum_demo = np.zeros((ydim,xdim,3))
for y in range(0,ydim):
    for x in range(0,xdim):
        sum_demo[y, x, 0] = cr_demo[y,x]  # red channel
        sum_demo[y, x, 1] = cg_demo[y,x]  # green channel
        sum_demo[y, x, 2] = cb_demo[y,x]  # blue channel
###############################################################################
# Saving images
io.imsave("./img/cr_demo.png", util.img_as_ubyte(cr_demo))
io.imsave("./img/cg_demo.png", util.img_as_ubyte(cg_demo))
io.imsave("./img/cb_demo.png", util.img_as_ubyte(cb_demo))
io.imsave("./img/sum_demo.png", util.img_as_ubyte(sum_demo))
io.imsave("./img/cr_splice.png", util.img_as_ubyte(cr))
io.imsave("./img/cg_splice.png", util.img_as_ubyte(cg))
io.imsave("./img/cb_splice.png", util.img_as_ubyte(cb))
###############################################################################
