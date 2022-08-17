# Spherical Baseline

*The Curvature Effect*

## Introduction

The workflow from MRI data to a mesh suitable for analysis contains two main
substeps:

* `Mask` $\mapsto$ `Isosurface`
  * from segmentation mask to isosurface tesselation, and
* `Isosurface` $\mapsto$ `Mesh`
  * from isosurface to finite element mesh

## Hypothesis

We hypothesize that numerical representation of the underlying physical object in terms of pixels, isosurface, and mesh is driven by `resolution`, which is ulitimately driven by local `curvature` of physical object.

We further hypothesize that the volume and curvature metrics of the numerical representation (in pixels, tesselation, or finite elements) *converge* to a constant value equal to the true underlying analog ground truth (the physical system being imaged by CT/MR).

## Definitions

* **Resolution** is defined as 
  * `dicom`: number of pixels per unit length
  * `stl`: number of triangles per isosurface tesselation, and 
  * `inp`: number of finite elements in a mesh.
* **Volume** is the $\real^3$ metric of space contained in the analytical domain $\Omega$.
  * For a domain $\Omega$ that is manifold.
  * For a domain $\Omega$ with watertight boundary $\partial \Omega$.
* **Curvature** is defined as the local second derivative of the boundary $\partial \Omega$.
* **Error** is the difference between the known ground truth value of volume and local curvature.
* **Error rate** is the slope of error versus resolution.

## Methods

We used an analytical shape of a sphere with radius = 10 cm.  A sphere is a useful baseline subject of study because it:

* can be easily approximated by a pixel stack at various resolutions (pixel densities),
* can easily be approximated by a finite element mesh,
* has a known analytical volume, and 
* has a known analytical local curvature

We selected this radius because it is the same order of magnitude as the human head.
