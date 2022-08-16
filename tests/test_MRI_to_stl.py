from atpixel import MRI_to_stl as mts
import numpy as np
from skimage.filters import threshold_otsu
import pytest


def test_compute_otsu_thresh():
    x1 = np.random.random((100, 100))
    x2 = np.random.random((100, 100)) * 500
    x = x1 + x2
    known = threshold_otsu(x)
    found = mts.compute_otsu_thresh(x)
    # print('known:', known,'found:',found)
    print(f"\nknown: {known}")
    print(f"found: {found}")
    # breakpoint()
    # assert known == found
    assert known == pytest.approx(found)


@pytest.mark.skip("work in progress")
def test_compute_otsu_thresh_robust():
    aa = 44
