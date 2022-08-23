import nibabel as nib
import numpy as np
from pathlib import Path

def NIfTI_to_numpy(input_file:Path) -> np.array:
    input_file_name = str(input_file)
    file = nib.load(input_file_name)
    file_data = file.get_fdata()
    return file_data 
