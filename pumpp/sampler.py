#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The sampler'''

from itertools import count

import numpy as np


class Sampler(object):
    def __init__(self, n_samples, duration, *ops):

        self.n_samples = n_samples
        self.duration = duration

        fields = dict()
        for op in ops:
            fields.update(op.fields)

        # Pre-determine which fields have time-like indices
        self._time = {key: None for key in fields}
        for key in fields:
            if None in fields[key].shape:
                # Add one for the batching index
                self._time[key] = 1 + fields[key].shape.index(None)

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
        for key in self._time:
            if self._time[key] is not None:
                lengths.append(data[key].shape[self._time[key]])

        return min(lengths)

    def __call__(self, data):

        duration = self.data_duration(data)

        for i in count(0):
            # are we done?
            if self.n_samples and i >= self.n_samples:
                break

            # Generate a sampling interval
            start = np.random.randint(0, duration - self.duration)

            yield self.sample(data, slice(start, start + self.duration))
