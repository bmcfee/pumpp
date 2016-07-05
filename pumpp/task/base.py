#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The base class for task transformer objects'''

import numpy as np
import librosa
import jams

__all__ = ['BaseTaskTransformer']

class BaseTaskTransformer(object):
    '''Base class for task transformer objects'''

    def __init__(self, namespace, name, fill_na, sr, hop_length):
        self.namespace = namespace

        if fill_na is None:
            fill_na = np.nan

        self.fill_na = fill_na
        self.sr = sr
        self.hop_length = hop_length
        self.name = name
        if name is not None:
            self._prefix = '{:s}/'.format(self.name)
        else:
            self._prefix = ''

    def transform(self, jam):

        # FIXME: this would be the part to insert filtering logic
        # Find annotations that can be coerced to our target namespace
        anns = []
        for ann in jam.annotations:
            try:
                anns.append(jams.nsconvert.convert(ann, self.namespace))
            except jams.NamespaceError:
                pass

        mask = True

        duration = jam.file_metadata.duration

        # If none, make a fake one
        if not anns:
            anns = [self.empty(duration)]
            mask = False

        # Apply transformations
        results = []
        for ann in anns:
            results.append(self.transform_annotation(ann, duration))
            results[-1]['mask'] = mask

        # Prefix and collect
        return self.merge(results)

    def merge(self, results):
        output = dict()

        for key in results[0]:
            pkey = '{:s}{:s}'.format(self._prefix, key)
            output[pkey] = np.stack([np.asarray(r[key]) for r in results], axis=0)
        return output

    def encode_events(self, duration, events, values):
        '''Encode labeled events as a time-series matrix.

        Parameters
        ----------
        duration : number
            The duration of the track

        events : ndarray, shape=(n,)
            Time index of the events

        values : ndarray, shape=(n, m)
            Values array.  Must have the same first index as `events`.

        Returns
        -------
        target : ndarray, shape=(n_frames, n_values)
        '''

        # FIXME: support sparse encoding
        # FIXME: support non-bool dtypes
        # FIXME: it's not clear what type `values` is here
        frames = librosa.time_to_frames(events,
                                        sr=self.sr,
                                        hop_length=self.hop_length)

        n_total = int(librosa.time_to_frames(duration, sr=self.sr,
                                             hop_length=self.hop_length))

        target = np.empty((n_total, values.shape[1]), dtype=values.dtype)

        target.fill(self.fill_na)

        for column, event in zip(values, frames):
            target[event] = column

        return target.astype(np.bool)

    def encode_intervals(self, duration, intervals, values):

        frames = librosa.time_to_frames(intervals,
                                        sr=self.sr,
                                        hop_length=self.hop_length)

        n_total = int(librosa.time_to_frames(duration, sr=self.sr,
                                             hop_length=self.hop_length))

        target = np.empty((n_total, values.shape[1]), dtype=values.dtype)

        target.fill(self.fill_na)

        for column, interval in zip(values, frames):
            target[interval[0]:interval[1]] += column

        return target.astype(np.bool)
