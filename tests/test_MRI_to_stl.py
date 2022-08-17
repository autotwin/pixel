from atpixel import MRI_to_stl as mts
import numpy as np
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.measure import marching_cubes
import pytest
from stl import mesh

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

def test_alpha_shape_mask_slice():
    val = 10
    array_1 = np.zeros((val*16,val*16))
    array_2 = np.zeros((val*16,val*16))
    big_shape = np.ones((val*4,val*4))
    little_shape = np.zeros((val*2,val*2))
    array_1[val*6:val*10,val*6:val*10] = big_shape
    array_2[val*6:val*10,val*6:val*10] = big_shape
    array_1[val*7:val*9,val*7:val*9] = little_shape
    thresh = 0.5
    alpha_shape_value = 0.0 
    known = array_2 > thresh
    found = mts.alpha_shape_mask_slice(array_1,thresh,alpha_shape_value)
    assert np.all(known == found)

@pytest.mark.skip("work in progress")
def test_alpha_shape_mask_slice_concave():
    aa = 44
    assert True

#@pytest.mark.skip("slow test")
def test_alpha_shape_mask_all():
    val = 3
    array_1 = np.zeros((val*16,val*16,val*16))
    array_2 = np.zeros((val*16,val*16,val*16))
    big_shape = np.ones((val*4,val*4,val*4))
    little_shape = np.zeros((val*2,val*2,val*2))
    array_1[val*6:val*10,val*6:val*10,val*6:val*10] = big_shape
    array_2[val*6:val*10,val*6:val*10,val*6:val*10] = big_shape
    array_1[val*7:val*9,val*7:val*9,val*7:val*9] = little_shape
    thresh = 0.5
    alpha_shape_value = 0.0 
    known = array_2 > thresh
    found = mts.alpha_shape_mask_all(array_1,alpha_shape_value)
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
    array_1 = np.zeros((val*3,val*3,val*3))
    array_1[0:val,:,:] = mag1
    array_1[val:val*2,:,:] = mag2
    array_1[val*2:val*3,:,:] = mag3 
    array_2 = np.zeros((val*3,val*3,val*3))
    array_2[val:val*2,:,:] = 1
    known = array_2 > 0 
    thresh_1 = (mag1+mag2)/2.0
    thresh_2 = (mag2+mag3)/2.0
    found = mts.threshold_lower_upper(array_1, thresh_1, thresh_2)
    assert np.all(known == found)

def test_select_largest_connected_volume():
    val = 10 
    array_1 = np.zeros((val*5,val*5,val*5))
    array_1[0:val,:,:] = 1
    array_1[val*2:val*3,val*2:val*3,:] = 1 
    array_1[val*4:val*5,val*4:val*5,val*4:val*5] = 1 
    array_2 = np.zeros((val*5,val*5,val*5))
    array_2[0:val,:,:] = 1
    known = array_2 > 0 
    found = mts.select_largest_connected_volume(array_1)
    assert np.all(known == found)

def test_close_mask():
    val = 5
    array_1 = np.zeros((val*16,val*16,val*16))
    array_2 = np.zeros((val*16,val*16,val*16))
    big_shape = np.ones((val*4,val*4,val*4))
    little_shape = np.zeros((val*2,val*2,val*2))
    array_1[val*6:val*10,val*6:val*10,val*6:val*10] = big_shape
    array_2[val*6:val*10,val*6:val*10,val*6:val*10] = big_shape
    array_1[val*7:val*9,val*7:val*9,val*7:val*9] = little_shape
    array_1 = array_1 > 0
    array_2 = array_2 > 0 
    known = array_2
    radius = val
    found = mts.close_mask(array_1,radius)
    assert np.all(known == found)

def test_dilate_mask():
    array_1 = np.zeros((7,7,7))
    array_1[3,3,3] = 1
    radius = 2
    ball = morphology.ball(radius, dtype=bool)
    array_2 = np.zeros((7,7,7))
    array_2[1:6,1:6,1:6] = ball 
    known = array_2 > 0
    found = mts.dilate_mask(array_1,radius)
    assert np.all(known == found)

def test_mask_brain():
    val = 5
    ball_1 = morphology.ball(val, dtype=bool)
    array = np.zeros((val*6,val*6,val*6))
    array[val-1:val*3,val-1:val*3,val-1:val*3] = ball_1
    array[2*val-1:val*4,2*val-1:val*4,2*val-1:val*4] = ball_1*2.0
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
    array = np.ones((val,val,val))
    ix0,ix1,ix2 = array.shape
    pfs = 3
    array_padded = np.zeros((ix0+pfs*2,ix1+pfs*2,ix2+pfs*2))
    array_padded[pfs:ix0+pfs,pfs:ix1+pfs,pfs:ix2+pfs] = array
    known = array_padded
    found = mts.pad_array(array,pfs) 
    assert np.all(known == found)

def test_padded_mask_to_verts_faces():
    val = 10
    array = morphology.ball(val, dtype=bool)
    marching_step_size = 1
    pad_size = 3
    array_padded = mts.pad_array(array, pad_size)
    verts_known, faces_known, normals_known,_ = marching_cubes(array_padded, step_size = marching_step_size)
    verts_found, faces_found = mts.padded_mask_to_verts_faces(array_padded, marching_step_size) 
    assert np.all(verts_known == verts_found) and np.all(faces_known == faces_found)
    
def test_faces_verts_to_mesh_object():
    val = 10
    array = morphology.ball(val, dtype=bool)
    marching_step_size = 1
    pad_size = 3
    array_padded = mts.pad_array(array, pad_size)
    verts_known, faces_known = mts.padded_mask_to_verts_faces(array_padded, marching_step_size) 
    mesh_for_stl_known = mesh.Mesh(np.zeros(faces_known.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces_known):
        for j in range(0,3):
            mesh_for_stl_known.vectors[i][j] = verts_known[f[j],:]
    mesh_for_stl_found = mts.faces_verts_to_mesh_object(verts_known, faces_known) 
    assert np.all(mesh_for_stl_known.points == mesh_for_stl_found.points) and np.all(mesh_for_stl_known.vectors == mesh_for_stl_found.vectors)
    
def test_mask_to_mesh_for_stl():
    val = 10
    array = morphology.ball(val, dtype=bool)
    marching_step_size = 1
    pad_size = 10
    array_padded = mts.pad_array(array, pad_size)
    verts_known, faces_known = mts.padded_mask_to_verts_faces(array_padded, marching_step_size) 
    mesh_for_stl_known = mts.faces_verts_to_mesh_object(verts_known, faces_known) 
    mesh_for_stl_found = mts.mask_to_mesh_for_stl(array, marching_step_size, pad_size)
    assert np.all(mesh_for_stl_known.points == mesh_for_stl_found.points) and np.all(mesh_for_stl_known.vectors == mesh_for_stl_found.vectors)
    