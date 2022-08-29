import alphashape
import argparse
from atpixel import NIfTI_to_numpy as ntn

# import cv2
# import glob as glob
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import nibabel as nib
import numpy as np

# import os
from pathlib import Path

from scipy import ndimage

# import pydicom as dicom
from shapely.geometry import Point

# from skimage import filters
from skimage import measure
from skimage import morphology
from skimage.measure import label
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
from stl import mesh

# import sys
import time
from typing import Iterable, Union, List
import yaml

# def re_scale_MRI_intensity(array: Iterable) -> Iterable:
#     """Given a 3D brain MRI array. Will re-scale intensity to enable white matter segment.
#     TODO: Write this function!."""
#     return array

# def rotate_to_ix0_transverse_axis(array: Iterable) -> Iterable:
#     """Given a 3D brain MRI array. Will rotate it so the transverse axis is ix0.
#     TODO: Write this function!."""
#     return array


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


def contour_points_slice(array_2D: Iterable, thresh: Union[float, int]) -> Iterable:
    """Given a 2D image and a threshold. Will return all points that form contours at this
    threshold value.
    NOTE: this function will work for slices in any direction, but for the next step to work
    we are expecting TRANSVERSE slices -- i.e., NOT sagittal or coronal."""
    contours = measure.find_contours(array_2D, thresh)
    point_list = []
    contour_counter = 0
    for contour in contours:
        for kk in range(contour.shape[0]):
            point_list.append((contour[kk, 0], contour[kk, 1]))  # tuples
        contour_counter += 1
    return point_list


def alpha_shape_mask_slice(
    array_2D: Iterable, thresh: Union[float, int], alpha_shape_value: float = 0.0
) -> Iterable:
    """Given a 2D image, a threshold, and value for computing the alpha shape.
    Will compute the alpha shape for each 2D image, turn that into a 2D mask.
    NOTE: this function will work for slices in any direction, but for best results
    we are expecting TRANSVERSE slices -- i.e., NOT sagittal or coronal.
    For additional info: https://alphashape.readthedocs.io/en/latest/"""
    point_list = contour_points_slice(array_2D, thresh)
    alpha_shape = alphashape.alphashape(point_list, alpha_shape_value)
    mask_2D = np.zeros(array_2D.shape)
    for jj in range(mask_2D.shape[0]):
        for kk in range(mask_2D.shape[1]):
            mask_2D[jj, kk] = alpha_shape.contains(Point(jj, kk))
    return mask_2D


def resample_equal_voxel_mask(
    array: Iterable, scale_ax_0: Union[float,int], scale_ax_1: Union[float,int], scale_ax_2: Union[float,int]
) -> Iterable:
    """Given a 3D mask and specified scaling of each axis. Will return the re-scaled mask."""
    array_mult = array * 100.0
    array_rescale = ndimage.zoom(
        array_mult, (scale_ax_0, scale_ax_1, scale_ax_2), order=1
    )
    small_thresh = 0.01
    mask_rescale = array_rescale > small_thresh  # 0 gives errors with precision
    return mask_rescale


def alpha_shape_mask_all(
    array: Iterable, axis_slice_transverse: int, alpha_shape_value: float = 0.0
) -> Iterable:
    """Given a 3D image, and value for computing the alpha shape. Will get a mask for each
    2D transverse slice based on the alpha shape, and then concatenate all of the 2D masks
    into a 3D mask that will be used to define the outer surface of the scan."""
    mask_3D = np.zeros(array.shape)
    thresh = compute_otsu_thresh(array)
    for kk in range(0, mask_3D.shape[0]):
        if axis_slice_transverse == 0:
            array_2D = array[kk, :, :]
        elif axis_slice_transverse == 1:
            array_2D = array[:, kk, :]
        elif axis_slice_transverse == 2:
            array_2D = array[:, :, kk]

        mask_2D = alpha_shape_mask_slice(array_2D, thresh, alpha_shape_value)

        if axis_slice_transverse == 0:
            mask_3D[kk, :, :] = mask_2D
        elif axis_slice_transverse == 1:
            mask_3D[:, kk, :] = mask_2D
        elif axis_slice_transverse == 2:
            mask_3D[:, :, kk] = mask_2D

    return mask_3D


def threshold_lower_upper(
    array: Iterable, thresh_min: Union[float, int], thresh_max: Union[float, int]
) -> Iterable:
    """Given an image array and an upper and lower threshold will return a mask for all
    pixels or voxels that are in the specified range. Designed for white matter."""
    check1 = (array > thresh_min).astype("uint8")
    check2 = (array < thresh_max).astype("uint8")
    check3 = check1 + check2
    selected_array = (check3 == 2).astype("uint8")
    return selected_array


def select_largest_connected_volume(array: Iterable) -> Iterable:
    """Given a thresholded array, will return the largest connected volume.
    This will help get rid of spurious features and ensure a meshable volume."""
    labels = label(array, background=0, connectivity=1)
    props = measure.regionprops(labels)
    areas = []
    for kk in range(0, len(props)):
        areas.append(props[kk].area)
    ix = np.argmax(areas)
    select_label = props[ix].label
    mask = labels == select_label
    return mask


def dilate_mask(array: Iterable, radius: int) -> Iterable:
    """Given a 3D binary array, will dilate it to close holes and
    extrude outwards with a spherical footprint."""
    footprint = morphology.ball(radius, dtype=bool)
    dilated_array = morphology.binary_dilation(array, footprint)
    return dilated_array


def close_mask(array: Iterable, radius: int) -> Iterable:
    """Given a 3D binary array, will close holes with a spherical footprint."""
    footprint = morphology.ball(radius, dtype=bool)
    closed_array = morphology.binary_closing(array, footprint)
    return closed_array


def mask_brain(
    array: Iterable,
    thresh_min: Union[float, int],
    thresh_max: Union[float, int],
    dilation_radius: int,
    close_radius: int,
) -> Iterable:
    """Given a 3D image and specification parameters. Will return the filled mask that will
    be used to define the brain isosurface."""
    thresholded_array = threshold_lower_upper(array, thresh_min, thresh_max)
    mask = select_largest_connected_volume(thresholded_array)
    dilated_mask = dilate_mask(mask, dilation_radius)
    closed_mask = close_mask(dilated_mask, close_radius)
    return closed_mask


# def mask_outer(array:Iterable, close_radius: int) -> Iterable:
#     """Given a 3D image and specification parameters. Will return the filled mask that will
#     be used to define the outer skull isosurface."""
#     thresholded_array = apply_otsu_thresh(array)
#     mask = select_largest_connected_volume(thresholded_array)
#     closed_mask = close_mask(mask, close_radius)
#     return closed_mask


def pad_array(array: Iterable, pad_size: int) -> Iterable:
    """Given an array. Add constant 0 padding on all sides."""
    padded_array = np.pad(array, pad_size, mode="constant", constant_values=0)
    return padded_array


def padded_mask_to_verts_faces(array: Iterable, marching_step_size: int) -> Iterable:
    """Given an array and marching cubes step size. Runs the marching cubes algorithm."""
    verts, faces, normals, _ = marching_cubes(array, step_size=marching_step_size)
    return verts, faces


def faces_verts_to_mesh_object(verts: Iterable, faces: Iterable) -> mesh.Mesh.dtype:
    """Given vertices and faces arrays. Converts them into a mesh object."""
    mesh_for_stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(0, 3):
            mesh_for_stl.vectors[i][j] = verts[f[j], :]
    return mesh_for_stl


def mask_to_mesh_for_stl(
    mask: Iterable, marching_step_size: int, pad_size: int = 10
) -> mesh.Mesh.dtype:
    """Given a mask array. Converts the mask array into a stl mesh object."""
    padded_mask = pad_array(mask, pad_size)
    verts, faces = padded_mask_to_verts_faces(padded_mask, marching_step_size)
    mesh_for_stl = faces_verts_to_mesh_object(verts, faces)
    return mesh_for_stl


def _yml_to_dict(*, yml_path_file: Path) -> dict:
    """Given a valid Path to a yml input file, read it in and
    return the result as a dictionary."""

    # Compared to the lower() method, the casefold() method is stronger.
    # It will convert more characters into lower case, and will find more matches
    # on comparison of two strings that are both are converted
    # using the casefold() method.
    atpixel: str = "atpixel>"

    if not yml_path_file.is_file():
        raise FileNotFoundError(f"{atpixel} File not found: {str(yml_path_file)}")

    file_type = yml_path_file.suffix.casefold()

    supported_types = (".yaml", ".yml")

    if file_type not in supported_types:
        raise TypeError("Only file types .yaml, and .yml are supported.")

    try:
        with open(yml_path_file, "r") as stream:
            # See deprecation warning for plain yaml.load(input) at
            # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
            db = yaml.load(stream, Loader=yaml.SafeLoader)
    except yaml.YAMLError as error:
        print(f"Error with YAML file: {error}")
        # print(f"Could not open: {self.self.path_file_in}")
        print(f"Could not open or decode: {yml_path_file}")
        # raise yaml.YAMLError
        raise OSError

    version_specified = db.get("version")
    version_implemented = 1.0

    if version_specified != version_implemented:
        raise ValueError(
            f"Version mismatch: specified was {version_specified}, implemented is {version_implemented}"
        )
    else:
        # require that input file has at least the following keys:
        required_keys = (
            "version",
            "nii_path_file",
            "mask_path_file_outer",
            "stl_path_file_outer",
            "mask_path_file_brain",
            "stl_path_file_brain",
            "visualization_folder_name",
            "has_metadata",
            "process_outer",
            "process_brain",
            "alpha_shape_param",
            "white_matter_min",
            "white_matter_max",
            "dilation_radius",
            "close_radius",
            "padding_for_stl",
            "marching_step_size",
            "scale_ax_0",
            "scale_ax_1",
            "scale_ax_2",
            "axis_slice_transverse",
            "axis_slice_coronal",
            "axis_slice_sagittal",
        )

        # has_required_keys = all(tuple(map(lambda x: db.get(x) != None, required_keys)))
        # keys_tuple = tuple(map(lambda x: db.get(x), required_keys))
        # has_required_keys = all(tuple(map(lambda x: db.get(x), required_keys)))
        found_keys = tuple(db.keys())
        keys_exist = tuple(map(lambda x: x in found_keys, required_keys))
        has_required_keys = all(keys_exist)
        if not has_required_keys:
            raise KeyError(f"Input files must have these keys defined: {required_keys}")
    return db


def string_to_path(path_string: str) -> Path:
    """Given a string that specifies a path. Will turn into a Path type and expand ~."""
    path_file = Path(path_string)
    path_file_expanded = path_file.expanduser()
    return path_file_expanded


def path_to_string(path_file: Path) -> str:
    """Given a Path type. Will convert to a string."""
    path_string = str(path_file)
    return path_string


def string_to_boolean(val: str) -> bool:
    """Given a string from yaml that should be a boolean."""
    val_boolean = val == "True"
    return val_boolean


def save_mask(mask: Iterable, mask_path_file: Path) -> None:
    """Given a mask numpy array and file name will save mask in the mask folder."""
    mask_path_string = path_to_string(mask_path_file)
    np.save(mask_path_string, mask)
    return


def save_stl(mesh: mesh.Mesh.dtype, file_name: Path) -> None:
    """Given a stl formatted mesh will save information in the stl."""
    file_name_str = path_to_string(file_name)
    mesh.save(file_name_str)
    return


def run_and_time_all_code(input_file: Path) -> List[float]:
    """Runs every step of the pipeline and returns a list of timing for each step.
    The optional argument is only used during de-bugging b/c this is the slowest step."""
    time_all = []
    time_all.append(time.time())

    # read input file
    fin = Path(input_file).expanduser()
    if not fin.is_file():
        raise FileNotFoundError(f"File not found: {input_file}")

    # extract input information from yaml file
    user_input = _yml_to_dict(yml_path_file=input_file)

    nii_path_file = string_to_path(user_input["nii_path_file"])
    mask_path_file_outer = string_to_path(user_input["mask_path_file_outer"])
    stl_path_file_outer = string_to_path(user_input["stl_path_file_outer"])
    mask_path_file_brain = string_to_path(user_input["mask_path_file_brain"])
    stl_path_file_brain = string_to_path(user_input["stl_path_file_brain"])

    has_metadata = string_to_boolean(user_input["has_metadata"])
    process_outer = string_to_boolean(user_input["process_outer"])
    process_brain = string_to_boolean(user_input["process_brain"])

    alpha_shape_param = user_input["alpha_shape_param"]
    white_matter_min = user_input["white_matter_min"]
    white_matter_max = user_input["white_matter_max"]
    dilation_radius = user_input["dilation_radius"]
    close_radius = user_input["close_radius"]
    padding_for_stl = user_input["padding_for_stl"]
    marching_step_size = user_input["marching_step_size"]

    scale_ax_0 = user_input["scale_ax_0"]
    scale_ax_1 = user_input["scale_ax_1"]
    scale_ax_2 = user_input["scale_ax_2"]
    axis_slice_transverse = user_input["axis_slice_transverse"]
    #axis_slice_coronal = user_input["axis_slice_coronal"]
    #axis_slice_sagittal = user_input["axis_slice_sagittal"]

    # begin timing
    time_all.append(time.time())

    # import the NIfTI file as an array
    # if has_metadata == False:
    if has_metadata is False:
        img_array = ntn.NIfTI_to_numpy(nii_path_file)
    time_all.append(time.time())

    # create the mask that defines the outer surface
    if process_outer:
        outer_mask_unscaled = alpha_shape_mask_all(
            img_array, axis_slice_transverse, alpha_shape_param
        )
        # outer_mask = mask_outer(img_array,close_radius)
        outer_mask = resample_equal_voxel_mask(
            outer_mask_unscaled, scale_ax_0, scale_ax_1, scale_ax_2
        )
        save_mask(outer_mask, mask_path_file_outer)
    time_all.append(time.time())

    # create the mask that defines the brain surface
    if process_brain:
        brain_mask_unscaled = mask_brain(
            img_array, white_matter_min, white_matter_max, dilation_radius, close_radius
        )
        brain_mask = resample_equal_voxel_mask(
            brain_mask_unscaled, scale_ax_0, scale_ax_1, scale_ax_2
        )
        save_mask(brain_mask, mask_path_file_brain)
    time_all.append(time.time())

    # create the mesh that defines the outer surface
    if process_outer:
        outer_mesh = mask_to_mesh_for_stl(
            outer_mask, marching_step_size, padding_for_stl
        )
        save_stl(outer_mesh, stl_path_file_outer)
    time_all.append(time.time())

    # create the mesh that defines the brain surface
    if process_brain:
        brain_mesh = mask_to_mesh_for_stl(
            brain_mask, marching_step_size, padding_for_stl
        )
        save_stl(brain_mesh, stl_path_file_brain)
    time_all.append(time.time())

    return time_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="the .yml user input file")
    args = parser.parse_args()
    input_file_str = args.input_file
    input_file = Path(input_file_str)
    time_all = run_and_time_all_code(input_file)
    time_i = time_all[0]
    time_f = time_all[-1]
    minutes = (time_f - time_i) / 60.0
    print("code ran in ", minutes, " minutes")
