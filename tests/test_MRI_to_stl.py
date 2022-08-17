from atpixel import MRI_to_stl as mts
import numpy as np
from skimage.filters import threshold_otsu
import pytest


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

@pytest.mark.skip("work in progress")
def test_example_skip():
    aa = 44
    # print('known:', known,'found:',found)
    #print(f"\nknown: {known}")
    #print(f"found: {found}")
    # breakpoint()
    # assert known == found
