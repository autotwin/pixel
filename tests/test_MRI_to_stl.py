from atpixel import MRI_to_stl as mts
import numpy as np
import os
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.measure import marching_cubes
import platform
import pytest
from stl import mesh

# import sys # unused import

# def test_rotate_to_ix0_transverse_axis():
#     #TODO: write this test!
#     assert True

# def test_re_scale_MRI_intensity():
#     #TODO: write this test!
#     assert True


def test_compute_otsu_thresh():
    x1 = np.random.random((100, 100, 100))
    x2 = np.random.random((100, 100, 100)) * 500
    x = x1 + x2
    known = threshold_otsu(x)
    found = mts.compute_otsu_thresh(x)
    assert known == pytest.approx(found)


def test_compute_otsu_thresh_robust():
    dim = 100
    known_lower = 10
    known_upper = 100
    std_lower = 2
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower, std_lower, dim * dim * dim)
    x1 = np.reshape(x1, (dim, dim, dim))
    x2 = np.random.normal(known_upper, std_upper, dim * dim * dim)
    x2 = np.reshape(x2, (dim, dim, dim))
    choose = np.random.random((dim, dim, dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    found = mts.compute_otsu_thresh(x1)
    assert found > known_lower and found < (known_upper + known_lower)


def test_apply_otsu_thresh():
    size = 100
    array = np.random.random((size, size, size))
    thresh = threshold_otsu(array)
    known = array > thresh
    found = mts.apply_otsu_thresh(array)
    assert np.all(known == found)


def test_apply_otsu_thresh_robust():
    dim = 100
    known_lower = 10
    known_upper = 10000
    std_lower = 0.1
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower, std_lower, dim * dim * dim)
    x1 = np.reshape(x1, (dim, dim, dim))
    x2 = np.random.normal(known_upper, std_upper, dim * dim * dim)
    x2 = np.reshape(x2, (dim, dim, dim))
    choose = np.random.random((dim, dim, dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    known = x1 > np.mean(x1)
    found = mts.apply_otsu_thresh(x1)
    assert np.all(known == found)


def test_contour_points_slice():
    rad = 10
    fp = morphology.disk(rad, dtype=bool)
    value = 5
    array = np.zeros((rad * 4, rad * 4))
    array[rad : rad + fp.shape[0], rad : rad + fp.shape[1]] = fp
    array = array * value
    thresh = value / 2.0
    point_list = mts.contour_points_slice(array, thresh)
    is_border_list = []
    for pt in point_list:
        c0 = int(pt[0])
        c1 = int(pt[1])
        patch = array[c0 - 1 : c0 + 2, c1 - 1 : c1 + 2]
        if np.sum(patch) > 0 and np.sum(patch) < value * 9:
            is_border_list.append(True)
        else:
            is_border_list.append(False)
    assert np.all(is_border_list)


def test_alpha_shape_mask_slice():
    val = 10
    array_1 = np.zeros((val * 16, val * 16))
    array_2 = np.zeros((val * 16, val * 16))
    big_shape = np.ones((val * 4, val * 4))
    little_shape = np.zeros((val * 2, val * 2))
    array_1[val * 6 : val * 10, val * 6 : val * 10] = big_shape
    array_2[val * 6 : val * 10, val * 6 : val * 10] = big_shape
    array_1[val * 7 : val * 9, val * 7 : val * 9] = little_shape
    thresh = 0.5
    alpha_shape_value = 0.0
    known = array_2 > thresh
    found = mts.alpha_shape_mask_slice(array_1, thresh, alpha_shape_value)
    assert np.all(known == found)


@pytest.mark.skip("work in progress")
def test_alpha_shape_mask_slice_concave():
    aa = 44
    assert True


# @pytest.mark.skip("slow test")
def test_alpha_shape_mask_all():
    val = 3
    array_1 = np.zeros((val * 16, val * 16, val * 16))
    array_2 = np.zeros((val * 16, val * 16, val * 16))
    big_shape = np.ones((val * 4, val * 4, val * 4))
    little_shape = np.zeros((val * 2, val * 2, val * 2))
    array_1[val * 6 : val * 10, val * 6 : val * 10, val * 6 : val * 10] = big_shape
    array_2[val * 6 : val * 10, val * 6 : val * 10, val * 6 : val * 10] = big_shape
    array_1[val * 7 : val * 9, val * 7 : val * 9, val * 7 : val * 9] = little_shape
    thresh = 0.5
    alpha_shape_value = 0.0
    axis_slice_transverse = 0
    known = array_2 > thresh
    found = mts.alpha_shape_mask_all(array_1, axis_slice_transverse, alpha_shape_value)
    assert np.all(known == found)


@pytest.mark.skip("work in progress")
def test_alpha_shape_mask_all_concave():
    aa = 44
    assert True


def test_threshold_lower_upper():
    mag1 = 10
    mag2 = 20
    mag3 = 30
    val = 10
    array_1 = np.zeros((val * 3, val * 3, val * 3))
    array_1[0:val, :, :] = mag1
    array_1[val : val * 2, :, :] = mag2
    array_1[val * 2 : val * 3, :, :] = mag3
    array_2 = np.zeros((val * 3, val * 3, val * 3))
    array_2[val : val * 2, :, :] = 1
    known = array_2 > 0
    thresh_1 = (mag1 + mag2) / 2.0
    thresh_2 = (mag2 + mag3) / 2.0
    found = mts.threshold_lower_upper(array_1, thresh_1, thresh_2)
    assert np.all(known == found)


def test_select_largest_connected_volume():
    val = 10
    array_1 = np.zeros((val * 5, val * 5, val * 5))
    array_1[0:val, :, :] = 1
    array_1[val * 2 : val * 3, val * 2 : val * 3, :] = 1
    array_1[val * 4 : val * 5, val * 4 : val * 5, val * 4 : val * 5] = 1
    array_2 = np.zeros((val * 5, val * 5, val * 5))
    array_2[0:val, :, :] = 1
    known = array_2 > 0
    found = mts.select_largest_connected_volume(array_1)
    assert np.all(known == found)


def test_close_mask():
    val = 5
    array_1 = np.zeros((val * 16, val * 16, val * 16))
    array_2 = np.zeros((val * 16, val * 16, val * 16))
    big_shape = np.ones((val * 4, val * 4, val * 4))
    little_shape = np.zeros((val * 2, val * 2, val * 2))
    array_1[val * 6 : val * 10, val * 6 : val * 10, val * 6 : val * 10] = big_shape
    array_2[val * 6 : val * 10, val * 6 : val * 10, val * 6 : val * 10] = big_shape
    array_1[val * 7 : val * 9, val * 7 : val * 9, val * 7 : val * 9] = little_shape
    array_1 = array_1 > 0
    array_2 = array_2 > 0
    known = array_2
    radius = val
    found = mts.close_mask(array_1, radius)
    assert np.all(known == found)


def test_dilate_mask():
    array_1 = np.zeros((7, 7, 7))
    array_1[3, 3, 3] = 1
    radius = 2
    ball = morphology.ball(radius, dtype=bool)
    array_2 = np.zeros((7, 7, 7))
    array_2[1:6, 1:6, 1:6] = ball
    known = array_2 > 0
    found = mts.dilate_mask(array_1, radius)
    assert np.all(known == found)


def test_mask_brain():
    val = 5
    ball_1 = morphology.ball(val, dtype=bool)
    array = np.zeros((val * 6, val * 6, val * 6))
    array[val - 1 : val * 3, val - 1 : val * 3, val - 1 : val * 3] = ball_1
    array[2 * val - 1 : val * 4, 2 * val - 1 : val * 4, 2 * val - 1 : val * 4] = (
        ball_1 * 2.0
    )
    thresh_min = 0.5
    thresh_max = 1.5
    dilation_radius = 2
    close_radius = 2
    thresholded_array = mts.threshold_lower_upper(array, thresh_min, thresh_max)
    mask = mts.select_largest_connected_volume(thresholded_array)
    dilated_mask = mts.dilate_mask(mask, dilation_radius)
    closed_mask = mts.close_mask(dilated_mask, close_radius)
    known = closed_mask
    found = mts.mask_brain(array, thresh_min, thresh_max, dilation_radius, close_radius)
    assert np.all(known == found)


@pytest.mark.skip("work in progress")
def test_mask_brain_robust():
    # broke down mask_brain into smaller chunks where test was feasible
    # test for whole thing all at once = ?
    aa = 44
    assert True


def test_pad_array():
    val = 10
    array = np.ones((val, val, val))
    ix0, ix1, ix2 = array.shape
    pfs = 3
    array_padded = np.zeros((ix0 + pfs * 2, ix1 + pfs * 2, ix2 + pfs * 2))
    array_padded[pfs : ix0 + pfs, pfs : ix1 + pfs, pfs : ix2 + pfs] = array
    known = array_padded
    found = mts.pad_array(array, pfs)
    assert np.all(known == found)


def test_padded_mask_to_verts_faces():
    val = 10
    array = morphology.ball(val, dtype=bool)
    marching_step_size = 1
    pad_size = 3
    array_padded = mts.pad_array(array, pad_size)
    verts_known, faces_known, normals_known, _ = marching_cubes(
        array_padded, step_size=marching_step_size
    )
    verts_found, faces_found = mts.padded_mask_to_verts_faces(
        array_padded, marching_step_size
    )
    assert np.all(verts_known == verts_found) and np.all(faces_known == faces_found)


def test_faces_verts_to_mesh_object():
    val = 10
    array = morphology.ball(val, dtype=bool)
    marching_step_size = 1
    pad_size = 3
    array_padded = mts.pad_array(array, pad_size)
    verts_known, faces_known = mts.padded_mask_to_verts_faces(
        array_padded, marching_step_size
    )
    mesh_for_stl_known = mesh.Mesh(
        np.zeros(faces_known.shape[0], dtype=mesh.Mesh.dtype)
    )
    for i, f in enumerate(faces_known):
        for j in range(0, 3):
            mesh_for_stl_known.vectors[i][j] = verts_known[f[j], :]
    mesh_for_stl_found = mts.faces_verts_to_mesh_object(verts_known, faces_known)
    assert np.all(mesh_for_stl_known.points == mesh_for_stl_found.points) and np.all(
        mesh_for_stl_known.vectors == mesh_for_stl_found.vectors
    )


def test_mask_to_mesh_for_stl():
    val = 10
    array = morphology.ball(val, dtype=bool)
    marching_step_size = 1
    pad_size = 10
    array_padded = mts.pad_array(array, pad_size)
    verts_known, faces_known = mts.padded_mask_to_verts_faces(
        array_padded, marching_step_size
    )
    mesh_for_stl_known = mts.faces_verts_to_mesh_object(verts_known, faces_known)
    mesh_for_stl_found = mts.mask_to_mesh_for_stl(array, marching_step_size, pad_size)
    assert np.all(mesh_for_stl_known.points == mesh_for_stl_found.points) and np.all(
        mesh_for_stl_known.vectors == mesh_for_stl_found.vectors
    )


def test_yml_to_dict():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    input_file_path = data_path.joinpath("small_sphere_no_metadata.yaml")
    db = mts._yml_to_dict(yml_path_file=input_file_path)
    assert db["version"] == 1.0
    assert (
        db["nii_path_file"]
        == "~/autotwin/pixel/tests/files/small_sphere_no_metadata.nii"
    )
    assert db["has_metadata"] == "False"
    assert db["alpha_shape_param"] == 0.05
    assert db["dilation_radius"] == 5


def test_when_io_fails():
    """Given a file name or a path that does not exist, checks that the
    function raises a FileNotFoundError."""
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()

    # If the user tries to run with a file that does not exist
    # then check that a FileNotFoundError is raised
    with pytest.raises(FileNotFoundError) as error:
        input_file = data_path.joinpath("this_file_does_not_exist.yml")
        mts._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "FileNotFoundError"

    with pytest.raises(FileNotFoundError) as error:
        input_file = data_path.joinpath("this_file_does_not_exist.yml")
        mts.run_and_time_all_code(input_file)
    assert error.typename == "FileNotFoundError"

    # If the user tries to run with a file type that is not a .yml or .yaml,
    # then check that a TypeError is raised.
    with pytest.raises(TypeError) as error:
        input_file = data_path.joinpath("small_sphere_no_metadata.nii")
        mts._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "TypeError"

    # If the user tried to run the input yml version that is not the version
    # curently implemented, then check that a ValueError is raised.
    with pytest.raises(ValueError) as error:
        input_file = data_path.joinpath("small_sphere_no_metadata_bad_version.yaml")
        mts._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "ValueError"

    # If the user tried to run the input yml that
    # does not have the correct keys, then test that a KeyError is raised.
    with pytest.raises(KeyError) as error:
        input_file = data_path.joinpath("small_sphere_no_metadata_bad_keys.yaml")
        mts._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "KeyError"

    # If the yaml cannot be loaded, then test that an OSError is raised.
    with pytest.raises(OSError) as error:
        input_file = data_path.joinpath("small_sphere_no_metadata_bad_yaml_load.yaml")
        mts._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "OSError"


@pytest.mark.skipif(
    ("atlas" not in platform.uname().node)
    and ("bu.edu" not in platform.uname().node)
    and ("eml" not in platform.uname().node),
    reason="Run on Atlas, eml, and bu.edu machines only.",
)
def test_string_to_path():
    known = Path(__file__)
    path_string_1 = "~/autotwin/pixel/tests/test_MRI_to_stl.py"
    found = mts.string_to_path(path_string_1)
    assert known == found


def test_path_to_string():
    path_1 = Path(__file__)
    known = str(path_1)
    found = mts.path_to_string(path_1)
    assert known == found


def test_string_to_boolean():
    # assert False == mts.string_to_boolean("False")
    # assert True == mts.string_to_boolean("True")
    assert mts.string_to_boolean("False") is not True
    assert mts.string_to_boolean("True")


@pytest.mark.skipif(
    ("atlas" not in platform.uname().node)
    and ("bu.edu" not in platform.uname().node)
    and ("eml" not in platform.uname().node),
    reason="Run on Atlas, eml, and bu.edu machines only.",
)
def test_save_mask():
    path_string_1 = "~/autotwin/pixel/tests/test_save_mask_987311.npy"
    path = mts.string_to_path(path_string_1)
    mask = morphology.ball(10, dtype=bool)
    mts.save_mask(mask, path)
    file_exists = path.is_file()
    assert file_exists  # assert test file was written
    if file_exists:
        os.remove(path)  # clean up, remove test file


@pytest.mark.skipif(
    ("atlas" not in platform.uname().node)
    and ("bu.edu" not in platform.uname().node)
    and ("eml" not in platform.uname().node),
    reason="Run on Atlas, eml, and bu.edu machines only.",
)
def test_save_stl():
    path_string_1 = "~/autotwin/pixel/tests/test_save_stl_987311.stl"
    path = mts.string_to_path(path_string_1)
    mask = morphology.ball(10, dtype=bool)
    marching_step_size = 1
    pad_size = 10
    mesh = mts.mask_to_mesh_for_stl(mask, marching_step_size, pad_size)
    mts.save_stl(mesh, path)
    file_exists = path.is_file()
    assert file_exists  # assert test file was written
    if file_exists:
        os.remove(path)  # clean up, remove test file


def test_resample_equal_voxel_mask():
    val = 10
    array = morphology.ball(val, dtype=bool)
    scale_ax_0 = 2
    scale_ax_1 = 0.5
    scale_ax_2 = 1
    array_rescale = mts.resample_equal_voxel_mask(
        array, scale_ax_0, scale_ax_1, scale_ax_2
    )
    assert array_rescale.shape[0] == int(scale_ax_0 * array.shape[0])
    assert array_rescale.shape[1] == int(scale_ax_1 * array.shape[1])
    assert array_rescale.shape[2] == int(scale_ax_2 * array.shape[2])
    assert array_rescale.max() == 1
    assert array_rescale.min() == 0


# try to get this test running on CI with variations on io path
@pytest.mark.skipif(
    ("atlas" not in platform.uname().node) and ("bu.edu" not in platform.uname().node) and ("eml" not in platform.uname().node),
    reason="Run on Atlas, eml, and bu.edu machines only.",
)
def test_run_and_time_all_code():
    def path_setup_in_files(fname):
        self_path_file = Path(__file__)
        self_path = self_path_file.resolve().parent
        data_path = self_path.joinpath("files").resolve()
        path_constructed = data_path.joinpath(fname)
        return path_constructed

    input_file = path_setup_in_files("quad_sphere_no_metadata.yaml")
    path_1 = path_setup_in_files("quad_sphere_no_metadata_outer.npy")
    path_2 = path_setup_in_files("quad_sphere_no_metadata_outer.stl")
    path_3 = path_setup_in_files("quad_sphere_no_metadata_brain.npy")
    path_4 = path_setup_in_files("quad_sphere_no_metadata_brain.stl")

    output_path_list = [path_1, path_2, path_3, path_4]
    for op in output_path_list:
        if op.is_file():
            os.remove(op)

    time_all = mts.run_and_time_all_code(input_file)
    n_timing_steps = 7  # semantic clarity, avoid magic numbers
    assert len(time_all) == n_timing_steps
    for op in output_path_list:
        assert op.is_file()
