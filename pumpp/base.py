#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Base class definitions'''

from collections import namedtuple, Iterable
import numpy as np

from .exceptions import ParameterError
__all__ = ['Tensor', 'Scope', 'Slicer']

# This type is used for storing shape information
Tensor = namedtuple('Tensor', ['shape', 'dtype'])
'''
Multi-dimensional array descriptions: `shape` and `dtype`
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


class Slicer(object):
    '''Slicer can compute the duration of data with time-like fields,
    and slice down to the common time index.

    This class serves as a base for Sampler and Pump, and should not
    be used directly.

    Parameters
    ----------
    ops : one or more Scope (TaskTransformer or FeatureExtractor)
    '''
    def __init__(self, *ops):

        self._time = dict()

        for operator in ops:
            self.add(operator)

    def add(self, operator):
        '''Add an operator to the Slicer

        Parameters
        ----------
        operator : Scope (TaskTransformer or FeatureExtractor)
            The new operator to add
        '''
        if not isinstance(operator, Scope):
            raise ParameterError('Operator {} must be a TaskTransformer '
                                 'or FeatureExtractor'.format(operator))
        for key in operator.fields:
            self._time[key] = []
            # We add 1 to the dimension here to account for batching
            for tdim, idx in enumerate(operator.fields[key].shape, 1):
                if idx is None:
                    self._time[key].append(tdim)

    def data_duration(self, data):
        '''Compute the valid data duration of a dict

        Parameters
        ----------
        data : dict
            As produced by pumpp.transform

        Returns
        -------
        length : int
            The minimum temporal extent of a dynamic observation in data
        '''
        # Find all the time-like indices of the data
        lengths = []
        for key in self._time:
            for idx in self._time.get(key, []):
                lengths.append(data[key].shape[idx])

        return min(lengths)

    def crop(self, data):
        '''Crop a data dictionary down to its common time

        Parameters
        ----------
        data : dict
            As produced by pumpp.transform

        Returns
        -------
        data_cropped : dict
            Like `data` but with all time-like axes truncated to the
            minimum common duration
        '''

        duration = self.data_duration(data)
        data_out = dict()
        for key in data:
            idx = [slice(None)] * data[key].ndim
            for tdim in self._time.get(key, []):
                idx[tdim] = slice(duration)
            data_out[key] = data[key][idx]

        return data_out
