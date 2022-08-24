# Configuration

## Overview

* Configure the local machine with a virtual environment named `atpixelenv`.  
* Create a `pyproject.toml` to configure the `atpixel` package.
* Install the `atpixel` module in developer mode.
* Assess unit tests and code coverage of the `atpixel` module.

## Methods

* Reference: https://packaging.python.org/en/latest/tutorials/installing-packages/
* We are curing Python 3.8 that comes with a conda install.
* Prerequisites:
  * The `autotwin` directory is created within the home `~` directory, and 
  * The `atpixel` repo is cloned into that `autotwin` folder.

### Install the conda environment

List the current conda environments:

```bash
conda env list
```

If `atpixelenv` appears in the list, delete it (since we will rebuild it from scratch below):

```bash
conda env remove --name atpixelenv
```

Create the new virtual environment:

```bash
conda create --name atpixelenv python=3.8
```

Activate the new virtual environment:

```bash
conda activate atpixelenv
```

Verify the Python version:

```bash
python --version  # should return a 3.8 variant, e.g., 3.8.13
```

Update some base modules:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Verify the base modules installed so far:

```bash
pip list

Package    Version
---------- -------
pip        22.2.2
setuptools 65.0.2
wheel      0.37.1
```

### Install `atpixel` as a developer

Make sure you are in the `~/autotwin/pixel` directory.

Reference: https://packaging.python.org/en/latest/tutorials/packaging-projects/

```bash
packaging_tutorial/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── your_package_name_here/
│       ├── __init__.py
│       └── example.py
└── tests/
```

Create a `pyproject.toml`, e.g., example `.toml` reference: https://peps.python.org/pep-0621/#example and general setuptools documentation: https://setuptools.pypa.io/en/latest/index.html

Installing from a local source tree, reference:

* https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-a-local-src-tree, and
* development mode: https://setuptools.pypa.io/en/latest/userguide/development_mode.html

Now, create an editable install (aka development mode):

```bash
python -m pip install -e .  # note: `-e .` = `--editable .`
```

At the time of this writing, the current version of `atpixel` is shown below.  Your version may be newer.  Post-install package status:

```bash
  pip list

Package            Version   Editable project location
------------------ --------- -------------------------
alphashape         1.3.1
atpixel            0.0.3     /Users/cbh/autotwin/pixel
attrs              22.1.0
black              22.6.0
click              8.1.3
click-log          0.4.0
coverage           6.4.4
cycler             0.11.0
fonttools          4.34.4
imageio            2.21.1
importlib-metadata 4.12.0
iniconfig          1.1.1
kiwisolver         1.4.4
matplotlib         3.5.3
mypy-extensions    0.4.3
networkx           2.6.3
nibabel            4.0.1
numpy              1.21.6
numpy-stl          2.17.1
opencv-python      4.6.0.66
packaging          21.3
pathspec           0.9.0
Pillow             9.2.0
pip                22.2.2
platformdirs       2.5.2
pluggy             1.0.0
py                 1.11.0
pydicom            2.3.0
pyparsing          3.0.9
pytest             7.1.2
pytest-cov         3.0.0
python-dateutil    2.8.2
python-utils       3.3.3
PyWavelets         1.3.0
Rtree              1.0.0
scikit-image       0.19.3
scipy              1.7.3
setuptools         65.0.2
Shapely            1.8.2
six                1.16.0
tifffile           2021.11.2
tomli              2.0.1
trimesh            3.13.4
typed-ast          1.5.4
typing_extensions  4.3.0
wheel              0.37.1
zipp               3.8.1
```

Run from the REPL:

```bash
(atpixelenv) cbh@atlas/Users/cbh/autotwin/pixel> python
Python 3.8.13 | packaged by conda-forge | (default, Mar 25 2022, 06:05:16)
[Clang 12.0.1 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from atpixel import hello as hh
>>> hh.hello_pixel()
'Hello pixel!'
>>> quit()
```

Run the tests with `pytest`:

```bash
(atpixelenv) cbh@atlas/Users/cbh/autotwin/pixel> pytest --cov=atpixel --cov-report term-missing
================================================================ test session starts =================================================================
platform darwin -- Python 3.8.13, pytest-7.1.2, pluggy-1.0.0
rootdir: /Users/cbh/autotwin/pixel
plugins: cov-3.0.0
collected 5 items

tests/test_hello.py .....                                                                                                                      [100%]

---------- coverage: platform darwin, python 3.8.13-final-0 -----------
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
src/atpixel/__init__.py       0      0   100%
src/atpixel/hello.py         10      1    90%   10
-------------------------------------------------------
TOTAL                        10      1    90%


================================================================= 5 passed in 0.03s ==================================================================
```

Success!  The virtual environment `atpixelenv` has been created, 
and the `atpixel` module is now installed and tested.
