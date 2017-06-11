#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Task transformations
====================
.. autosummary::
    :toctree: generated/

    BaseTaskTransformer
    BeatTransformer
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
from .event import *
from .regression import *
from .tags import *
from .structure import *
