'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-02 19:16:11
'''
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

from .normalize_staining import *
from .wsi_util import *
from .image_utli import *
from .torch_util import *
from .classmap import *
from .signal_util import *
from .utils import *
