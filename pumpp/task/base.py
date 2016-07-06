#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The base class for task transformer objects'''

import numpy as np
import librosa
import jams

from ..core import Tensor

__all__ = ['BaseTaskTransformer']


def fill_value(dtype):
    '''Get a fill-value for a given dtype

    Parameters
    ----------
    dtype : type

    Returns
    -------
    `np.nan` if `dtype` is real or complex

    0 otherwise
    '''
    if np.issubdtype(dtype, np.float) or np.issubdtype(dtype, np.complex):
        return dtype(np.nan)

    return dtype(0)


class BaseTaskTransformer(object):
    '''Base class for task transformer objects'''

    def __init__(self, namespace, name, sr, hop_length):
        self.namespace = namespace

        self.sr = sr
        self.hop_length = hop_length
        self.name = name
        if name is not None:
            self._prefix = '{:s}/'.format(self.name)
        else:
            self._prefix = ''
        self.fields = dict()


    def empty(self, duration):
        return jams.Annotation(namespace=self.namespace, time=0, duration=0)

    def transform(self, jam, query=None):

        anns = []
        if query:
            results = jam.search(**query)
        else:
            results = jam.annotations

        # Find annotations that can be coerced to our target namespace
        for ann in results:
            try:
                anns.append(jams.nsconvert.convert(ann, self.namespace))
            except jams.NamespaceError:
                pass

        duration = jam.file_metadata.duration

        # If none, make a fake one
        if not anns:
            anns = [self.empty(duration)]

        # Apply transformations
        results = []
        for ann in anns:

            results.append(self.transform_annotation(ann, duration))
            # If the annotation range is None, it spans the entire track
            if ann.time is None or ann.duration is None:
                valid = [0, duration]
            else:
                valid = [ann.time, ann.time + ann.duration]

            results[-1]['_valid'] = librosa.time_to_frames(valid,
                                                           sr=self.sr,
                                                           hop_length=self.hop_length)

        # Prefix and collect
        return self.merge(results)

    def merge(self, results):
        output = dict()

        for key in results[0]:
            pkey = '{:s}{:s}'.format(self._prefix, key)
            output[pkey] = np.stack([np.asarray(r[key]) for r in results], axis=0)
        return output

    def register(self, field, shape, dtype):
        # TODO: validate shape and dtype here
        self.fields['{:s}{:s}'.format(self._prefix, field)] = Tensor(tuple(shape), dtype)

    def encode_events(self, duration, events, values, dtype=np.bool):
        '''Encode labeled events as a time-series matrix.

        Parameters
        ----------
        duration : number
            The duration of the track

        events : ndarray, shape=(n,)
            Time index of the events

        values : ndarray, shape=(n, m)
            Values array.  Must have the same first index as `events`.

        dtype : numpy data type

        Returns
        -------
        target : ndarray, shape=(n_frames, n_values)
        '''

        # FIXME: support sparse encoding
        frames = librosa.time_to_frames(events,
                                        sr=self.sr,
                                        hop_length=self.hop_length)

        n_total = int(librosa.time_to_frames(duration, sr=self.sr,
                                             hop_length=self.hop_length))

        target = np.empty((n_total, values.shape[1]), dtype=dtype)

        target.fill(fill_value(dtype))
        values = values.astype(dtype)
        for column, event in zip(values, frames):
            target[event] += column

        return target

    def encode_intervals(self, duration, intervals, values, dtype=np.bool):

        frames = librosa.time_to_frames(intervals,
                                        sr=self.sr,
                                        hop_length=self.hop_length)

        n_total = int(librosa.time_to_frames(duration, sr=self.sr,
                                             hop_length=self.hop_length))

        values = values.astype(dtype)

        target = np.empty((n_total, values.shape[1]), dtype=dtype)

        target.fill(fill_value(dtype))

        for column, interval in zip(values, frames):
            target[interval[0]:interval[1]] += column

        return target
