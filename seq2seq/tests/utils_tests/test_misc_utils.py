# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from utils import misc_utils

class TestMiscUtils(object):
    def test_print_out(self):
        misc_utils.print_out("This is a test.")

if __name__ == "__main__":
    pytest.main()
### EOF
