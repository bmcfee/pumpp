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

from .exceptions import ParameterError

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

    random_state : None, int, or np.random.RandomState
        If int, random_state is the seed used by the random number
        generator;

        If RandomState instance, random_state is the random number
        generator;

        If None, the random number generator is the RandomState instance
        used by np.random.

    ops : array of pumpp.feature.FeatureExtractor or pumpp.task.BaseTaskTransformer
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
    def __init__(self, n_samples, duration, *ops, random_state=None):

        self.n_samples = n_samples
        self.duration = duration

        if random_state is None:
            self.rng = np.random
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            raise ParameterError('Invalid random_state={}'.format(random_state))

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
            if '_valid' in key:
                continue

            index = [slice(None)] * data[key].ndim

            # if we have multiple observations for this key, pick one
            index[0] = self.rng.randint(0, data[key].shape[0])
            index[0] = slice(index[0], index[0] + 1)

            if self._time.get(key, None) is not None:
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

    def indices(self, data):
        '''Generate patch indices

        Parameters
        ----------
        data : dict of np.ndarray
            As produced by pumpp.transform

        Yields
        ------
        start : int >= 0
            The start index of a sample patch
        '''
        duration = self.data_duration(data)

        while True:
            # Generate a sampling interval
            yield self.rng.randint(0, duration - self.duration)

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
        if self.n_samples:
            counter = range(self.n_samples)
        else:
            counter = count(0)

        for i, start in zip(counter, self.indices(data)):
            yield self.sample(data, slice(start, start + self.duration))
