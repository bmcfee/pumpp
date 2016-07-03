#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Regression task transformers'''

import numpy as np

from .base import BaseTaskTransformer

__all__ = ['VectorTransformer']


class VectorTransformer(BaseTaskTransformer):

    def __init__(self, namespace, dimension, name='vector', sr=22050, hop_length=512):

        super(VectorTransformer, self).__init__(namespace, 0, sr, hop_length)

        self.dimension = dimension
        self.name = name

    def transform(self, jam):

        ann = self.find_annotation(jam)

        if ann:
            vector = np.asarray(ann.data.value.iloc[0])
            if len(vector) != self.dimension:
                raise RuntimeError('vector dimension({:0}) '
                                   '!= self.dimension({:1})'
                                   .format(len(vector), self.dimension))
            mask = True
        else:
            vector = np.zeros(self.dimension, dtype=np.float32)
            mask = False

        return {'output_{:s}'.format(self.name): vector,
                'mask_{:s}'.format(self.name): mask}
