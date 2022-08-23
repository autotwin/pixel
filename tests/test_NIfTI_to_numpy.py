from atpixel import NIfTI_to_numpy as ntn
from pathlib import Path 

def test_NIfTI_to_numpy():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    input_file = data_path.joinpath("small_sphere_no_metadata.nii")
    file_data = ntn.NIfTI_to_numpy(input_file)
    assert file_data.shape == (21,21,21)
    assert file_data.max() == 100.
    assert file_data.min() == 0.
    assert file_data.size == 9261