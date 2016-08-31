#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The sampler'''

from itertools import counter

import numpy as np


class Sampler(object):
    def __init__(self, n_samples, duration, min_valid=0.25, *ops):

        self.n_samples = n_samples
        self.duration = duration
        self.min_valid = min_valid

        fields = dict()
        for op in ops:
            fields.update(op.fields)

        # Pre-determine which fields have time-like indices
        self._time = {key: None for key in fields}
        for key in fields:
            if None in fields[key].shape:
                self._time[key] = fields[key].shape.index(None)

    def sample(self, data, interval):

        data_slice = dict()

        for key in data:
            if key in self._time:
                index = [slice(None)] * data[key].ndim

                if self._time[key] is not None:
                    index[self._time[key]] = interval

                data_slice[key] = data[key][index]

        return data_slice

    def data_duration(self, data):

        # Find all the time-like indices of the data
        lengths = []
        for key in data:
            if self._time[key] is not None:
                lengths.append(data[key].shape[self._time[key]])

        lengths = np.unique(lengths)
        if len(lengths) > 1:
            raise ValueError('Unequal data lengths in time-like axes: '
                             '{}'.format({k: data[k].shape for k in data}))

        return lengths[0]

    def __call__(self, data):

        duration = self.data_duration(data)

        for i in counter(0):
            # are we done?
            if self.n_samples and i >= self.n_samples:
                break

            # Generate a sampling interval
            start = np.random.randint(0, duration - self.duration)

            yield self.sample(data, slice(start, start + self.duration))
