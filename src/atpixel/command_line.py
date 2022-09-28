"""This module provides functions for command line
entry points.
"""

import pkg_resources  # part of setup tools


def version() -> str:
    ver = pkg_resources.require("atpixel")[0].version
    print("autotwin pixel module version:")
    return ver
