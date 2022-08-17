# import alphashape
# import cv2
# import glob as glob
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import nibabel as nib
# import numpy as np
# import os
# import pydicom as dicom
# from shapely.geometry import Point
# from skimage import filters
# from skimage import measure
# from skimage import morphology
# from skimage.measure import label
# from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu

# from stl import mesh
# import time
from typing import Iterable, Union


def compute_otsu_thresh(array: Iterable) -> Union[float, int]:
    """Given a value numpy array. Will return the otsu threshold value."""
    thresh = threshold_otsu(array)
    return thresh

def apply_otsu_thresh(array: Iterable) -> Iterable:
	"""Given a value numpy array. Will return a boolean numpy array with an otsu threshold 
	applied."""
	thresh = compute_otsu_thresh(array)
	thresh_img = array > thresh
	return thresh_img

