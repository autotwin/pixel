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
from skimage import measure
# from skimage import morphology
# from skimage.measure import label
# from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu

# from stl import mesh
# import time
from typing import Iterable, Union

def re_scale_MRI_intensity(array: Iterable) -> Iterable:
	"""Given a 3D brain MRI array. Will re-scale intensity to enable white matter segment.
	TODO: Write this function!."""
	return array

def rotate_to_ix0_transverse_axis(array: Iterable) -> Iterable:
	"""Given a 3D brain MRI array. Will rotate it so the transverse axis is ix0.
	TODO: Write this function!."""
	return array 

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

def contour_points_slice(array_2D: Iterable,thresh: Union[float,int]) -> Iterable:
	"""Given a 2D image and a threshold. Will return all points that form contours at this
	threshold value. 
	NOTE: this function will work for slices in any direction, but for the next step to work
	we are expecting TRANSVERSE slices -- i.e., NOT sagital or coronal."""
	contours = measure.find_contours(array_2D, thresh)
	point_list = [] 
	contour_counter = 0
	for contour in contours:
		for kk in range(contour.shape[0]):
			point_list.append((contour[kk,0],contour[kk,1])) # tuples 
		contour_counter += 1
	return point_list 


