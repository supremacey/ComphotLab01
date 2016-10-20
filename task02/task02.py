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
#                Demosaicking - gradient interpolation
###############################################################################
# green channel - edge-based
for y in range(0,ydim,2):
    for x in range(0,xdim,2):
        vertical_gradinet = abs(cg_demo[max(y-1,0),x] - cg_demo[min(y+1,ydim-1),x])/2.0
        horizontal_gradient = abs(cg_demo[y,max(x-1,0)] - cg_demo[y,min(x+1,xdim-1)])/2.0
        if (horizontal_gradient > vertical_gradinet):
            cg_demo[y,x] += .5*cg_demo[max(y-1,0),x]
            cg_demo[y,x] += .5*cg_demo[min(y+1,ydim-1),x]
        elif (vertical_gradinet > horizontal_gradient):
            cg_demo[y,x] += .5*cg_demo[y,max(x-1,0)]
            cg_demo[y,x] += .5*cg_demo[y,min(x+1,xdim-1)]
        else:
            cg_demo[y,x] += .25*cg_demo[max(y-1,0),x]
            cg_demo[y,x] += .25*cg_demo[y,max(x-1,0)]
            cg_demo[y,x] += .25*cg_demo[y,min(x+1,xdim-1)]
            cg_demo[y,x] += .25*cg_demo[min(y+1,ydim-1),x]

for y in range(1,ydim,2):
    for x in range(1,xdim,2):
        vertical_gradinet = abs(cg_demo[max(y-1,0),x] - cg_demo[min(y+1,ydim-1),x])/2.0
        horizontal_gradient = abs(cg_demo[y,max(x-1,0)] - cg_demo[y,min(x+1,xdim-1)])/2.0
        if (horizontal_gradient > vertical_gradinet):
            cg_demo[y,x] += .5*cg_demo[max(y-1,0),x]
            cg_demo[y,x] += .5*cg_demo[min(y+1,ydim-1),x]
        elif (vertical_gradinet > horizontal_gradient):
            cg_demo[y,x] += .5*cg_demo[y,max(x-1,0)]
            cg_demo[y,x] += .5*cg_demo[y,min(x+1,xdim-1)]
        else:
            cg_demo[y,x] += .25*cg_demo[max(y-1,0),x]
            cg_demo[y,x] += .25*cg_demo[y,max(x-1,0)]
            cg_demo[y,x] += .25*cg_demo[y,min(x+1,xdim-1)]
            cg_demo[y,x] += .25*cg_demo[min(y+1,ydim-1),x]

################################################################################
# red channel
# for recorded red value pixels: -> R-G
cr_demo[0::2, 0::2] = np.subtract(cr_demo[0::2, 0::2], cg_demo[0::2, 0::2])

# vertical red-green diff interpolation over upper and lower neighbour
for y in range(1,ydim,2):
    for x in range(0,xdim,2):
        cr_demo[y,x] += .5*cr_demo[max(y-1, 0), x]
        cr_demo[y,x] += .5*cr_demo[min(y+1, ydim-1), x]

# horizontal red-green diff interpolation over left and right neighbour
for y in range(0,ydim,2):
    for x in range(1,xdim,2):
        cr_demo[y,x] += .5*cr_demo[y, max(x-1,0)]
        cr_demo[y,x] += .5*cr_demo[y, min(x+1, xdim-1)]

# red channel-green diff middle interpolation
for y in range(1, ydim, 2):
    for x in range(1, xdim, 2):
        cr_demo[y,x] += .25*cr_demo[max(y-1, 0), max(x-1,0)] # left up
        cr_demo[y,x] += .25*cr_demo[max(y-1, 0), min(x+1,xdim-1)] # right up
        cr_demo[y,x] += .25*cr_demo[min(y+1,ydim-1), min(x+1,xdim-1)] # right down
        cr_demo[y,x] += .25*cr_demo[min(y+1,ydim-1), max(x-1,0)] # left down

# adding green to red
cr_demo = np.add(cr_demo, cg_demo)
# float [-1,1] range clip
cr_demo = np.clip(cr_demo, 0.0,1.0)

################################################################################
# blue channel

# for recorded blue value pixels: -> B-G
cb_demo[1::2, 1::2] = np.subtract(cb_demo[1::2, 1::2], cg_demo[1::2, 1::2])
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

# adding green to blue
cb_demo = np.add(cb_demo, cg_demo)
# float [-1,1] range clip
cb_demo = np.clip(cb_demo, 0.0,1.0)

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
