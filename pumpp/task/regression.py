#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Regression task transformers'''

import numpy as np

from .base import BaseTaskTransformer

__all__ = ['VectorTransformer']


class VectorTransformer(BaseTaskTransformer):

    def __init__(self, namespace, dimension, name='vector', dtype=np.float32):

        super(VectorTransformer, self).__init__(namespace, name=name,
                                                sr=1, hop_length=1)

        self.dimension = dimension
        self.dtype = dtype

        self.register('vector', [None, self.dimension], self.dtype)

    def empty(self, duration):
        ann = super(VectorTransformer, self).empty(duration)

        ann.append(time=0, duration=duration, confidence=0,
                   value=np.zeros(self.dimension, dtype=np.float32))
        return ann

    def transform_annotation(self, ann, duration):

        vector = np.asarray(ann.data.value.iloc[0], dtype=self.dtype)
        if len(vector) != self.dimension:
            raise RuntimeError('vector dimension({:0}) '
                               '!= self.dimension({:1})'
                               .format(len(vector), self.dimension))

        return {'vector': vector}
