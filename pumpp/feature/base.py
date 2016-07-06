#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Feature extraction base class'''

import numpy as np

from ..core import Tensor

class FeatureExtractor(object):
    def __init__(self):
        raise NotImplementedError
