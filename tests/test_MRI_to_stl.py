from atpixel import MRI_to_stl as mts
import numpy as np
from skimage.filters import threshold_otsu
from skimage import morphology
import pytest

def test_rotate_to_ix0_transverse_axis():
    #TODO: write this test! 
    assert True

def test_re_scale_MRI_intensity():
    #TODO: write this test!
    assert True

def test_compute_otsu_thresh():
    x1 = np.random.random((100, 100,100))
    x2 = np.random.random((100, 100,100)) * 500
    x = x1 + x2
    known = threshold_otsu(x)
    found = mts.compute_otsu_thresh(x)
    assert known == pytest.approx(found)

def test_compute_otsu_thresh_robust():
    dim = 100 
    dim = 100 
    known_lower = 10 
    known_upper = 100 
    std_lower = 2
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower,std_lower,dim*dim*dim)
    x1 = np.reshape(x1,(dim,dim,dim))
    x2 = np.random.normal(known_upper,std_upper,dim*dim*dim)
    x2 = np.reshape(x2,(dim,dim,dim))
    choose = np.random.random((dim,dim,dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    found = mts.compute_otsu_thresh(x1)
    assert found > known_lower and found < (known_upper + known_lower)

def test_apply_otsu_thresh():
    array = np.random.random((100,100,100))
    thresh = threshold_otsu(array)
    known = array > thresh    
    found = mts.apply_otsu_thresh(array)
    assert np.all(known == found)

def test_apply_otsu_thresh_robust():
    dim = 100 
    dim = 100 
    known_lower = 10 
    known_upper = 10000
    std_lower = .1
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower,std_lower,dim*dim*dim)
    x1 = np.reshape(x1,(dim,dim,dim))
    x2 = np.random.normal(known_upper,std_upper,dim*dim*dim)
    x2 = np.reshape(x2,(dim,dim,dim))
    choose = np.random.random((dim,dim,dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    known = x1 > np.mean(x1)
    found = mts.apply_otsu_thresh(x1)
    assert np.all(known == found)

def test_contour_points_slice():
    rad = 10
    fp = morphology.disk(rad, dtype=bool)
    value = 5 
    array = np.zeros((rad*4,rad*4))
    array[rad:rad+fp.shape[0],rad:rad+fp.shape[1]] = fp
    array = array*value
    thresh = value / 2.0
    point_list = mts.contour_points_slice(array,thresh)
    is_border_list = [] 
    for pt in point_list:
        c0 = int(pt[0])
        c1 = int(pt[1])
        patch = array[c0-1:c0+2,c1-1:c1+2]
        if np.sum(patch) > 0 and np.sum(patch) < value*9:
            is_border_list.append(True)
        else:
            is_border_list.append(False)
    assert np.all(is_border_list)


@pytest.mark.skip("work in progress")
def test_example_skip():
    aa = 44
    # print('known:', known,'found:',found)
    #print(f"\nknown: {known}")
    #print(f"found: {found}")
    # breakpoint()
    # assert known == found
