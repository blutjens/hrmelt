"""
Functions to blur images using opencv instead of torch.
These functions are used, for example, when running our 
code on Apple silicon chip for which torch.GaussianBlur
is very slow.
"""
import cv2
import numpy as np
from torchvision import transforms as T


def accept_tensor(func):
    def wrapper(x):
        # Perform some work on the input
        x = x.squeeze()
        x = np.asarray(x)
        # Call the original function with the modified input
        x = func(x)
        # Perform some work in the output
        x = T.ToTensor()(x)
        x = x.unsqueeze(0)

        return x
    return wrapper


def get_blur_func_cv(blur_type, kernel_size, sigma):
    @accept_tensor
    def gaus(x):
        return cv2.GaussianBlur(src=x, ksize=(kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

    @accept_tensor
    def biliteral(x):
        return cv2.bilateralFilter(x, d=kernel_size, sigmaColor=0, sigmaSpace=sigma)

    @accept_tensor
    def average(x):
        return cv2.boxFilter(x, ksize=(kernel_size, kernel_size), ddepth=-1)

    @accept_tensor
    def median(x):
        return cv2.medianBlur(x, ksize=3)

    # borderline insane i know
    def none(x):
        return x

    if blur_type is None:
        return none
    if blur_type == 'gaussian':
        return gaus
    if blur_type == 'biliteral':
        return biliteral
    if blur_type == 'average':
        return average
    if blur_type == 'median':
        return median
