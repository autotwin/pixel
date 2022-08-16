"""
For coverage:
pytest geo/tests/test_hello.py -v --cov=geo/src/ptg --cov-report term-missing
"""

# import .hello as hh
# import hello as hh
# from ptg import hello as hh
# from atmesh import hello as hh
from atpixel import hello as hh


def test_one():
    assert True


def test_two():
    known = "Hello world!"
    found = hh.hello()

    assert found == known

def test_add_two():
	x = 3
	y = 7.0
	known = 10 
	found = hh.add_two(x,y)
	assert found == known

def test_adios():
    known = "Bye"
    found = hh.adios()

    assert found == known


def test_bubble_sort():
    assert hh.bubble_sort()


def test_pixel_repo():
    known = "Hello pixel!"
    found = hh.hello_pixel()

    assert found == known
    
