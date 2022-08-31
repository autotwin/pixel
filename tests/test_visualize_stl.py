from atpixel import MRI_to_stl as mts
from atpixel import NIfTI_to_numpy as ntn
from atpixel import visualize_stl as vstl
import os
from pathlib import Path
import platform
import pytest


@pytest.mark.skipif(
    ("atlas" not in platform.uname().node)
    and ("bu.edu" not in platform.uname().node)
    and ("eml" not in platform.uname().node),
    reason="Run on Atlas, eml, and bu.edu machines only.",
)
def test_create_folder():
    path_string_1 = "~/autotwin/pixel/tests/test_path_311768"
    path = mts.string_to_path(path_string_1)
    vstl.create_folder(path)
    dir_exists = path.is_dir()
    assert dir_exists  # assert test file was written
    # if dir_exists:
    #    os.remove(path)  # clean up, remove test path -- it seems that I do not currently have permission for this!


def path_to_test_files() -> Path:
    """Locates the autotwin/pixel/tests/files/ folder relative to this
    current file.
    """
    files_path = Path(__file__).parent.joinpath("files").resolve()
    return files_path


@pytest.mark.skipif(
    ("atlas" not in platform.uname().node)
    and ("bu.edu" not in platform.uname().node)
    and ("eml" not in platform.uname().node),
    reason="Run on Atlas, eml, and bu.edu machines only.",
)
def test_create_skull_still():
    # yml_path = Path(
    #     "~/autotwin/pixel/tests/files/quad_sphere_no_metadata.yaml"
    # ).expanduser()
    yml_path = path_to_test_files().joinpath("quad_sphere_no_metadata.yaml")
    db = mts._yml_to_dict(yml_path_file=yml_path)

    # stl_path_file_outer = Path(db["stl_path_file_outer"]).expanduser()
    # vis_path = Path(db["visualization_folder_name"]).expanduser()
    stl_path_file_outer = path_to_test_files().joinpath(db["stl_path_file_outer"])
    vis_path = path_to_test_files().joinpath(db["visualization_folder_name"])

    alpha_shape_param = db["alpha_shape_param"]
    axis_slice_transverse = db["axis_slice_transverse"]
    marching_step_size = db["marching_step_size"]

    # nii_path_file = Path(db["nii_path_file"]).expanduser()
    nii_path_file = path_to_test_files().joinpath(db["nii_path_file"])

    padding_for_stl = db["padding_for_stl"]

    img_array = ntn.NIfTI_to_numpy(nii_path_file)
    outer_mask = mts.alpha_shape_mask_all(
        img_array, axis_slice_transverse, alpha_shape_param
    )
    outer_mesh = mts.mask_to_mesh_for_stl(
        outer_mask, marching_step_size, padding_for_stl
    )
    mts.save_stl(outer_mesh, stl_path_file_outer)
    vstl.create_folder(vis_path)
    vstl.create_skull_still(stl_path_file_outer, vis_path)

    path = Path(str(vis_path) + "/skull_still_image_elev-90_azim90.png")
    file_exists = path.is_file()
    assert file_exists  # assert test file was written
    if file_exists:
        os.remove(path)


@pytest.mark.skipif(
    ("atlas" not in platform.uname().node)
    and ("bu.edu" not in platform.uname().node)
    and ("eml" not in platform.uname().node),
    reason="Run on Atlas, eml, and bu.edu machines only.",
)
def test_get_visualization_relevant_path_name():
    input_file_str = "~/autotwin/pixel/tests/files/quad_sphere_no_metadata.yaml"
    input_file_str = str(path_to_test_files().joinpath("quad_sphere_no_metadata.yaml"))

    vis_path, stl_path_file_outer = vstl.get_visualization_relevant_path_names(
        input_file_str
    )
    # assert str(vis_path) == str(
    #     Path(
    #         "~/autotwin/pixel/tests/files/quad_sphere_no_metadata_visualizations"
    #     ).expanduser()
    # )
    assert str(vis_path) == "quad_sphere_no_metadata_visualizations"

    # assert str(stl_path_file_outer) == str(
    #     mts.string_to_path(
    #         "~/autotwin/pixel/tests/files/quad_sphere_no_metadata_outer.stl"
    #     ).expanduser()
    # )
    assert str(stl_path_file_outer) == "quad_sphere_no_metadata_outer.stl"


# @pytest.mark.skipif(
#     ("atlas" not in platform.uname().node)
#     and ("bu.edu" not in platform.uname().node)
#     and ("eml" not in platform.uname().node),
#     reason="Run on Atlas, eml, and bu.edu machines only.",
# )
@pytest.mark.skip("WIP: CBH needs to rearchitect io.")
def test_run_visualization_code():
    input_file_str = "~/autotwin/pixel/tests/files/quad_sphere_no_metadata.yaml"
    input_file_str = str(path_to_test_files().joinpath("quad_sphere_no_metadata.yaml"))
    vstl.run_visualization_code(input_file_str)
    vis_path = Path(
        "~/autotwin/pixel/tests/files/quad_sphere_no_metadata_visualizations"
    ).expanduser()
    path = Path(str(vis_path) + "/skull_still_image_elev-90_azim90.png")
    file_exists = path.is_file()
    assert file_exists  # assert test file was written
    if file_exists:
        os.remove(path)
