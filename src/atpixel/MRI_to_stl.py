import alphashape
# import cv2
# import glob as glob
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import nibabel as nib
import numpy as np
# import os
# import pydicom as dicom
from shapely.geometry import Point
# from skimage import filters
from skimage import measure
from skimage import morphology
from skimage.measure import label
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
from stl import mesh
# import time
from typing import Iterable, Union

# def re_scale_MRI_intensity(array: Iterable) -> Iterable:
# 	"""Given a 3D brain MRI array. Will re-scale intensity to enable white matter segment.
# 	TODO: Write this function!."""
# 	return array

# def rotate_to_ix0_transverse_axis(array: Iterable) -> Iterable:
# 	"""Given a 3D brain MRI array. Will rotate it so the transverse axis is ix0.
# 	TODO: Write this function!."""
# 	return array 

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

def alpha_shape_mask_slice(array_2D: Iterable,thresh: Union[float,int],alpha_shape_value:float= 0.0) -> Iterable:
	"""Given a 2D image, a threshold, and value for computing the alpha shape. 
	Will compute the alpha shape for each 2D image, turn that into a 2D mask.
	NOTE: this function will work for slices in any direction, but for best results 
	we are expecting TRANSVERSE slices -- i.e., NOT sagital or coronal. 
	For additional info: https://alphashape.readthedocs.io/en/latest/"""
	point_list = contour_points_slice(array_2D,thresh)
	alpha_shape = alphashape.alphashape(point_list, alpha_shape_value)
	mask_2D = np.zeros(array_2D.shape)
	for jj in range(mask_2D.shape[0]):
		for kk in range(mask_2D.shape[1]):
			mask_2D[jj,kk] = alpha_shape.contains(Point(jj,kk))
	return mask_2D

def alpha_shape_mask_all(array: Iterable,alpha_shape_value:float = 0.0) -> Iterable:
	"""Given a 3D image, and value for computing the alpha shape. Will get a mask for each 
	2D transverse slice based on the alpha shape, and then concatenate all of the 2D masks 
	into a 3D mask that will be used to define the outer surface of the scan.""" 
	mask_3D = np.zeros(array.shape)
	thresh = compute_otsu_thresh(array)
	for kk in range(0,mask_3D.shape[0]):
		array_2D = array[kk,:,:]
		mask_2D = alpha_shape_mask_slice(array_2D,thresh,alpha_shape_value)
		mask_3D[kk,:,:] = mask_2D
	return mask_3D

def threshold_lower_upper(array: Iterable, thresh_min: Union[float,int],thresh_max: Union[float,int]) -> Iterable:
	"""Given an image array and an upper and lower threshold will return a mask for all 
	pixels or voxels that are in the specified range. Designed for white matter."""
	check1 = (array > thresh_min).astype('uint8')
	check2 = (array < thresh_max).astype('uint8')
	check3 = check1 + check2 
	selected_array = (check3 == 2).astype('uint8')
	return selected_array

def select_largest_connected_volume(array: Iterable) -> Iterable:
	"""Given a thresholded array, will return the largest connected volume.
	This will help get rid of spurious features and ensure a meshable volume."""
	labels = label(array,background=0,connectivity=1)
	props = measure.regionprops(labels)
	areas = [] 
	for kk in range(0,len(props)):
		areas.append(props[kk].area)
	ix = np.argmax(areas)
	select_label = props[ix].label
	mask = labels == select_label
	return mask

def dilate_mask(array: Iterable, radius: int) -> Iterable:
	"""Given a 3D binary array, will dilate it to close holes and 
	extrude outwards with a spherical footprint."""
	footprint = morphology.ball(radius, dtype=bool)
	dilated_array = morphology.binary_dilation(array,footprint)
	return dilated_array

def close_mask(array: Iterable, radius: int) -> Iterable:
	"""Given a 3D binary array, will close holes with a spherical footprint."""
	footprint = morphology.ball(radius, dtype=bool)
	closed_array = morphology.binary_closing(array,footprint)
	return closed_array 
	
def mask_brain(array: Iterable, thresh_min: Union[float,int], thresh_max: Union[float,int], dilation_radius: int, close_radius: int) -> Iterable:
	"""Given a 3D image and specification parameters. Will return the filled mask that will
	be used to define the brain isosurface."""
	thresholded_array = threshold_lower_upper(array, thresh_min, thresh_max)
	mask = select_largest_connected_volume(thresholded_array)
	dilated_mask = dilate_mask(mask, dilation_radius)
	closed_mask = close_mask(dilated_mask, close_radius)
	return closed_mask

def pad_array(array: Iterable, pad_size: int) -> Iterable:
	"""Given an array. Add constant 0 padding on all sides."""
	padded_array = np.pad( array , pad_size, mode='constant', constant_values = 0)
	return padded_array

def padded_mask_to_verts_faces(array: Iterable, marching_step_size: int) -> Iterable:
	"""Given an array and marching cubes step size. Runs the marching cubes algorithm."""
	verts,faces, normals,_ = marching_cubes(array, step_size = marching_step_size)
	return verts, faces

def faces_verts_to_mesh_object(verts: Iterable, faces: Iterable) -> mesh.Mesh.dtype:
	"""Given vertices and faces arrays. Converts them into a mesh object.""" 
	mesh_for_stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
	for i, f in enumerate(faces):
		for j in range(0,3):
			mesh_for_stl.vectors[i][j] = verts[f[j],:]
	return mesh_for_stl

def mask_to_mesh_for_stl(mask: Iterable, marching_step_size: int, pad_size: int = 10) -> mesh.Mesh.dtype:
	"""Given a mask array. Converts the mask array into a stl mesh object."""
	padded_mask = pad_array(mask, pad_size)
	verts, faces = padded_mask_to_verts_faces(padded_mask, marching_step_size)
	mesh_for_stl = faces_verts_to_mesh_object(verts, faces)
	return mesh_for_stl


