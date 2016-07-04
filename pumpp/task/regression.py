#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Regression task transformers'''

import numpy as np

from .base import BaseTaskTransformer

__all__ = ['VectorTransformer']


class VectorTransformer(BaseTaskTransformer):

    def __init__(self, namespace, dimension, name='vector'):

        super(VectorTransformer, self).__init__(namespace, name=name,
                                                fill_na=0,
                                                sr=1, hop_length=1)

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

        return {'{:s}_vector'.format(self.name): vector,
                'mask_{:s}'.format(self.name): mask}
