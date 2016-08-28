#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Original inpainting code by Davide Lasagna
# https://github.com/gasagna/openpiv-python/blob/master/openpiv/src/lib.pyx
# Cython removed and Gaussian kernel code added by opit (http://technarium.lt)
# Note that the Gaussian kernel has a default standard deviation equal to 3
# and is normalised to sum up to 1 to preserve flux, which means that for
# larger standard deviation you'd have to increase the kernel size to avoid
# artifacts.
# Matplotlib used for testing only.

from __future__ import division
import numpy as np
from scipy import ndimage
import inpaint
import matplotlib.pyplot as plt


def inpaint_array(inputArray, mask, **kwargs):
    maskedImg = np.ma.array(inputArray, mask=mask)
    NANMask = maskedImg.filled(np.NaN)
    badArrays, num_badArrays = ndimage.label(mask)
    # data_slices = ndimage.find_objects(badArrays)
    filled = inpaint.replace_nans(NANMask, **kwargs)
    return filled


def test():
    mask = np.zeros((41, 41))
    mask[16:21, 18:21] = 1
    inputArray = inpaint.makeGaussian(41, 25)/np.max(
        inpaint.makeGaussian(41, 8))

    fig1 = plt.figure(figsize=(8, 6))
    fig1.add_subplot(111)
    c = plt.imshow(inputArray * (1 - mask), interpolation='None',
                   cmap=plt.cm.cubehelix)
    plt.title("Bad Array")
    plt.colorbar(c)

    fig2 = plt.figure(figsize=(8, 6))
    fig2.add_subplot(111)
    d = inpaint_array(inputArray, mask, max_iter=20, tol=0.05,
                      kernel_radius=5, kernel_sigma=2,
                      method='idw')
    c = plt.imshow(d, interpolation="None", cmap=plt.cm.cubehelix)
    plt.title("Healed Array Using IDW")
    plt.colorbar(c)

    fig3 = plt.figure(figsize=(8, 6))
    fig3.add_subplot(111)
    d = inpaint_array(inputArray, mask, max_iter=20, tol=0.05,
                      kernel_radius=5, method='localmean')
    c = plt.imshow(d, interpolation="None", cmap=plt.cm.cubehelix)
    plt.title("Healed Array Using Localmean")
    plt.colorbar(c)

    plt.show()


if __name__ == '__main__':
    test()
