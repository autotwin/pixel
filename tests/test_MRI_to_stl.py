from atpixel import MRI_to_stl as mts 
import numpy as np
from skimage.filters import threshold_otsu

def test_compute_otsu_thresh():
	x1 = np.random.random((100,100))
	x2 = np.random.random((100,100))*500 
	x = x1 + x2 
	known = threshold_otsu(x)
	found = mts.compute_otsu_thresh(x)
	print('known:', known,'found:',found)
	assert known == found 
