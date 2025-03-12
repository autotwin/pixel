"""This module provides unit tests of the command_line functionality."""

from atpixel import command_line as cl


def test_version():
    known = "0.0.7"
    found = cl.version()
    assert known == found
