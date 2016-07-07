#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Core class definitions'''

from collections import namedtuple
import numpy as np

__all__ = ['Tensor', 'Scope']

Tensor = namedtuple('Tensor', ['shape', 'dtype'])


class Scope(object):

    def __init__(self, name):

        self.name = name
        self.fields = dict()

    def scope(self, key):

        if self.name is None:
            return key
        return '{:s}/{:s}'.format(self.name, key)

    def register(self, field, shape, dtype):
        if not isinstance(dtype, type):
            raise TypeError('dtype={} must be a type'.format(dtype))

        if not all([s is None or isinstance(s, int) for s in shape]):
            raise ValueError('shape={} must be an iterable of integers'.format(shape))

        self.fields[self.scope(field)] = Tensor(tuple(shape), dtype)

    def merge(self, data):
        data_out = dict()

        # Iterate over all keys in data
        for key in set().union(*data):
            data_out[self.scope(key)] = np.stack([np.asarray(d[key]) for d in data],
                                                 axis=0)
        return data_out

