import nibabel as nib
import numpy as np
from pathlib import Path

# TODO: CH ask EL regarding return type of np.array versus numpy.memmap


# def NIfTI_to_numpy(input_file: Path) -> np.array:
def NIfTI_to_numpy(input_file: Path) -> np.ndarray:
    input_file_name = str(input_file)
    file = nib.load(input_file_name)
    file_data = file.get_fdata()
    return file_data
