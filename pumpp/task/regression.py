#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Regression task transformers'''

import numpy as np

import jams

from .base import BaseTaskTransformer
from ..exceptions import DataError

__all__ = ['VectorTransformer']


class VectorTransformer(BaseTaskTransformer):
    '''Vector regression transformer.

    Attributes
    ----------
    name : str
        The name of this transformer

    namespace : str
        The target namespace of this transformer

    dimension : int > 0
        The dimension of the vector data

    dtype : np.dtype
        The desired data type of the output
    '''
    def __init__(self, name, namespace, dimension, dtype=np.float32):
        super(VectorTransformer, self).__init__(name=name,
                                                namespace=namespace,
                                                sr=1, hop_length=1)

        self.dimension = dimension
        self.dtype = dtype

        self.register('vector', [1, self.dimension], self.dtype)

    def empty(self, duration):
        '''Empty vector annotations.

        This returns an annotation with a single observation
        vector consisting of all-zeroes.

        Parameters
        ----------
        duration : number >0
            Length of the track

        Returns
        -------
        ann : jams.Annotation
            The empty annotation
        '''
        ann = super(VectorTransformer, self).empty(duration)

        ann.append(time=0, duration=duration, confidence=0,
                   value=np.zeros(self.dimension, dtype=np.float32))
        return ann

    def transform_annotation(self, ann, duration):
        '''Apply the vector transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The input annotation

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['vector'] : np.ndarray, shape=(dimension,)

        Raises
        ------
        DataError
            If the input dimension does not match
        '''
        _, values = ann.to_interval_values()
        vector = np.asarray(values[0], dtype=self.dtype)
        if len(vector) != self.dimension:
            raise DataError('vector dimension({:0}) '
                            '!= self.dimension({:1})'
                            .format(len(vector), self.dimension))

        return {'vector': vector}

    def inverse(self, vector, duration=None):
        '''Inverse vector transformer'''

        ann = jams.Annotation(namespace=self.namespace, duration=duration)

        if duration is None:
            duration = 0
        ann.append(time=0, duration=duration, value=vector)

        return ann
