#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Data subsampling
================
.. autosummary::
    :toctree: generated/

    Sampler
'''

from itertools import count

import numpy as np

__all__ = ['Sampler']


class Sampler(object):
    '''Generate samples from a pumpp data dict.

    Attributes
    ----------
    n_samples : int or None
        the number of samples to generate.
        If `None`, generate indefinitely.

    duration : int > 0
        the duration (in frames) of each sample

    ops : one or more pumpp.feature.FeatureExtractor or pumpp.task.BaseTaskTransformer
        The operators to include when sampling data.


    Examples
    --------

    >>> # Set up the parameters
    >>> sr, n_fft, hop_length = 22050, 512, 2048
    >>> # Instantiate some transformers
    >>> p_stft = pumpp.feature.STFTMag('stft', sr=sr, n_fft=n_fft,
    ...                                hop_length=hop_length)
    >>> p_beat = pumpp.task.BeatTransformer('beat', sr=sr,
    ...                                     hop_length=hop_length)
    >>> # Apply the transformers to the data
    >>> data = pumpp.transform('test.ogg', 'test.jams', p_stft, p_beat)
    >>> # We'll sample 10 patches of duration = 32 frames
    >>> stream = pumpp.Sampler(10, 32, p_stft, p_beat)
    >>> # Apply the streamer to the data dict
    >>> for example in stream(data):
    ...     process(data)
    '''
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
        '''Sample a patch from the data object

        Parameters
        ----------
        data : dict
            A data dict as produced by pumpp.transform

        interval : slice
            The time interval to sample

        Returns
        -------
        data_slice : dict
            `data` restricted to `interval`.
        '''
        data_slice = dict()

        for key in data:
            if key in self._time:
                index = [slice(None)] * data[key].ndim

                if self._time[key] is not None:
                    index[self._time[key]] = interval

                data_slice[key] = data[key][index]

        return data_slice

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
            if self._time[key] is not None:
                lengths.append(data[key].shape[self._time[key]])

        return min(lengths)

    def __call__(self, data):
        '''Generate samples from a data dict.

        Parameters
        ----------
        data : dict
            As produced by pumpp.transform

        Yields
        ------
        data_sample : dict
            A sequence of patch samples from `data`,
            as parameterized by the sampler object.
        '''
        duration = self.data_duration(data)

        for i in count(0):
            # are we done?
            if self.n_samples and i >= self.n_samples:
                break

            # Generate a sampling interval
            start = np.random.randint(0, duration - self.duration)

            yield self.sample(data, slice(start, start + self.duration))
