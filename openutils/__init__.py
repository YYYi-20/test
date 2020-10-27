#
# openutils - Python utils for the whole slide image
#
# Copyright (c) CityU HK
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 2.1 of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#


"""A  library of utils for reading and processing whole-slide images.
"""
from .version import __version__

from .normalize_staining import normalize_staining
from .slideset import Slideset
from .image_utli import *
