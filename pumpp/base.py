#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Base class definitions'''

from collections import namedtuple, Iterable
import numpy as np

from .exceptions import *
__all__ = ['Tensor', 'Scope']

# This type is used for storing shape information
Tensor = namedtuple('Tensor', ['shape', 'dtype'])
'''
Apparently you can document namedtuples here
'''


class Scope(object):
    '''
    A base class for managing named tensors

    Attributes
    ----------
    name : str or None
        The name of this object.  If not `None`,
        all field keys are prefixed by `name/`.

    fields : dict of str : Tensor
        A dictionary of fields produced by this object.
        Each value defines the shape and data type of the field.
    '''
    def __init__(self, name):
        self.name = name
        self.fields = dict()

    def scope(self, key):
        '''Apply the name scope to a key

        Parameters
        ----------
        key : string

        Returns
        -------
        `name/key` if `name` is not `None`;
        otherwise, `key`.
        '''
        if self.name is None:
            return key
        return '{:s}/{:s}'.format(self.name, key)

    def register(self, field, shape, dtype):
        '''Register a field as a tensor with specified shape and type.

        A `Tensor` of the given shape and type will be registered in this
        object's `fields` dict.

        Parameters
        ----------
        field : str
            The name of the field

        shape : iterable of `int` or `None`
            The shape of the output variable.
            This does not include a dimension for multiple outputs.

            `None` may be used to indicate variable-length outputs

        dtype : type
            The data type of the field

        Raises
        ------
        ParameterError
            If dtype or shape are improperly specified
        '''
        if not isinstance(dtype, type):
            raise ParameterError('dtype={} must be a type'.format(dtype))

        if not (isinstance(shape, Iterable) and
                all([s is None or isinstance(s, int) for s in shape])):
            raise ParameterError('shape={} must be an iterable of integers'.format(shape))

        self.fields[self.scope(field)] = Tensor(tuple(shape), dtype)

    def pop(self, field):
        return self.fields.pop(self.scope(field))

    def merge(self, data):
        '''Merge an array of output dictionaries into a single dictionary
        with properly scoped names.

        Parameters
        ----------
        data : list of dict
            Output dicts as produced by `pumpp.task.BaseTaskTransformer.transform`
            or `pumpp.feature.FeatureExtractor.transform`.

        Returns
        -------
        data_out : dict
            All elements of the input dicts are stacked along the 0 axis,
            and keys are re-mapped by `scope`.
        '''
        data_out = dict()

        # Iterate over all keys in data
        for key in set().union(*data):
            data_out[self.scope(key)] = np.stack([np.asarray(d[key]) for d in data],
                                                 axis=0)
        return data_out
