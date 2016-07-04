#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Regression task transformers'''

import numpy as np
import jams

from .base import BaseTaskTransformer

__all__ = ['VectorTransformer']


class VectorTransformer(BaseTaskTransformer):

    def __init__(self, namespace, dimension, name='vector'):

        super(VectorTransformer, self).__init__(namespace, name=name,
                                                fill_na=0,
                                                sr=1, hop_length=1)

        self.dimension = dimension

    def empty(self, duration):
        vector = np.zeros(self.dimension, dtype=np.float32)
        ann = jams.Annotation(namespace=self.namespace)
        ann.append(time=0, duration=duration, value=vector, confidence=0)
        return ann

    def transform_annotation(self, ann, duration):

        vector = np.asarray(ann.data.value.iloc[0])
        if len(vector) != self.dimension:
            raise RuntimeError('vector dimension({:0}) '
                               '!= self.dimension({:1})'
                               .format(len(vector), self.dimension))

        return {'vector': vector}
