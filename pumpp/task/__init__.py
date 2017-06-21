#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Task transformations
====================
.. autosummary::
    :toctree: generated/

    BaseTaskTransformer
    BeatTransformer
    BeatPositionTransformer
    ChordTransformer
    SimpleChordTransformer
    ChordTagTransformer
    VectorTransformer
    DynamicLabelTransformer
    StaticLabelTransformer
    StructureTransformer
'''

from .base import *
from .chord import *
from .beat import *
from .regression import *
from .tags import *
from .structure import *
