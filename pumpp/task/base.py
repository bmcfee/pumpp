#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The base class for task transformer objects'''

import numpy as np
from librosa import time_to_frames, times_like
from librosa.sequence import viterbi_binary, viterbi_discriminative
import jams

from ..base import Scope

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
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
        return dtype(np.nan)

    return dtype(0)


class BaseTaskTransformer(Scope):
    '''Base class for task transformer objects

    Attributes
    ----------
    name : str
        The name prefix for this transformer object

    namespace : str
        The JAMS namespace for annotations in this task

    sr : number > 0
        The sampling rate for audio

    hop_length : int > 0
        The number of samples between frames
    '''

    def __init__(self, name, namespace, sr, hop_length):
        super(BaseTaskTransformer, self).__init__(name)

        # This will trigger an exception if the namespace is not found
        jams.schema.is_dense(namespace)

        self.namespace = namespace
        self.sr = sr
        self.hop_length = hop_length

    def empty(self, duration):
        '''Create an empty jams.Annotation for this task.

        This method should be overridden by derived classes.

        Parameters
        ----------
        duration : int >= 0
            Duration of the annotation
        '''
        return jams.Annotation(namespace=self.namespace, time=0, duration=0)

    def transform(self, jam, query=None):
        '''Transform jam object to make data for this task

        Parameters
        ----------
        jam : jams.JAMS
            The jams container object

        query : string, dict, or callable [optional]
            An optional query to narrow the elements of `jam.annotations`
            to be considered.

            If not provided, all annotations are considered.

        Returns
        -------
        data : dict
            A dictionary of transformed annotations.
            All annotations which can be converted to the target namespace
            will be converted.
        '''
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

            results[-1]['_valid'] = time_to_frames(valid, sr=self.sr,
                                                   hop_length=self.hop_length)

        # Prefix and collect
        return self.merge(results)

    def transform_annotation(self, ann, duration):
        '''Transform jams.Annotation to make data for a given task.
        
        Parameters
        ----------
        ann : jams.Annotation
            The jams annotation containing the data

        duration : number > 0 
           time in seconds of the output duration

        Returns
        -------
        data : dict
            A dictionary of transformed annotation.
        '''
        raise NotImplementedError

    def encode_events(self, duration, events, values, dtype=bool):
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

        frames = time_to_frames(events, sr=self.sr,
                                hop_length=self.hop_length)

        n_total = int(time_to_frames(duration, sr=self.sr,
                                     hop_length=self.hop_length))

        n_alloc = n_total
        if np.any(frames):
            n_alloc = max(n_total, 1 + int(frames.max()))

        target = np.empty((n_alloc, values.shape[1]),
                          dtype=dtype)

        target.fill(fill_value(dtype))
        values = values.astype(dtype)
        for column, event in zip(values, frames):
            target[event] += column

        return target[:n_total]

    def encode_intervals(self, duration, intervals, values, dtype=bool,
                         multi=True, fill=None):
        '''Encode labeled intervals as a time-series matrix.

        Parameters
        ----------
        duration : number
            The duration (in frames) of the track

        intervals : np.ndarray, shape=(n, 2)
            The list of intervals

        values : np.ndarray, shape=(n, m)
            The (encoded) values corresponding to each interval

        dtype : np.dtype
            The desired output type

        multi : bool
            If `True`, allow multiple labels per interval.

        fill : dtype (optional)
            Optional default fill value for missing data.

            If not provided, the default is inferred from `dtype`.

        Returns
        -------
        target : np.ndarray, shape=(duration * sr / hop_length, m)
            The labeled interval encoding, sampled at the desired frame rate
        '''
        if fill is None:
            fill = fill_value(dtype)

        frames = time_to_frames(intervals, sr=self.sr,
                                hop_length=self.hop_length)

        n_total = int(time_to_frames(duration, sr=self.sr,
                                     hop_length=self.hop_length))

        values = values.astype(dtype)

        n_alloc = n_total
        if np.any(frames):
            n_alloc = max(n_total, 1 + int(frames.max()))

        target = np.empty((n_alloc, values.shape[1]),

                          dtype=dtype)

        target.fill(fill)

        for column, interval in zip(values, frames):
            if multi:
                target[interval[0]:interval[1]] += column
            else:
                target[interval[0]:interval[1]] = column

        return target[:n_total]

    def decode_events(self, encoded, transition=None, p_state=None, p_init=None):
        '''Decode labeled events into (time, value) pairs

        Real-valued inputs are thresholded at 0.5.

        Optionally, viterbi decoding can be applied to each event class.

        Parameters
        ----------
        encoded : np.ndarray, shape=(n_frames, m)
            Frame-level annotation encodings as produced by ``encode_events``.

        transition : None or np.ndarray [shape=(2, 2) or (m, 2, 2)]
            Optional transition matrix for each event, used for Viterbi

        p_state : None or np.ndarray [shape=(m,)]
            Optional marginal probability for each event

        p_init : None or np.ndarray [shape=(m,)]
            Optional marginal probability for each event

        Returns
        -------
        [(time, value)] : iterable of tuples
            where `time` is the event time and `value` is an
            np.ndarray, shape=(m,) of the encoded value at that time

        See Also
        --------
        librosa.sequence.viterbi_binary
        '''
        if np.isrealobj(encoded):
            if transition is None:
                encoded = (encoded >= 0.5)
            else:
                encoded = viterbi_binary(encoded.T, transition,
                                         p_state=p_state,
                                         p_init=p_init).T

        times = times_like(encoded,
                           sr=self.sr,
                           hop_length=self.hop_length,
                           axis=0)

        return zip(times, encoded)

    def decode_intervals(self, encoded, duration=None, multi=True, sparse=False,
                         transition=None, p_state=None, p_init=None):
        '''Decode labeled intervals into (start, end, value) triples

        Parameters
        ----------
        encoded : np.ndarray, shape=(n_frames, m)
            Frame-level annotation encodings as produced by
            ``encode_intervals``

        duration : None or float > 0
            The max duration of the annotation (in seconds)
            Must be greater than the length of encoded array.

        multi : bool
            If true, allow multiple labels per input frame.
            If false, take the most likely label per input frame.

        sparse : bool
            If true, values are returned as indices, not one-hot.
            If false, values are returned as one-hot encodings.

            Only applies when `multi=False`.

        transition : None or np.ndarray [shape=(m, m) or (2, 2) or (m, 2, 2)]
            Optional transition matrix for each interval, used for Viterbi
            decoding.  If `multi=True`, then transition should be `(2, 2)` or
            `(m, 2, 2)`-shaped.  If `multi=False`, then transition should be
            `(m, m)`-shaped.

        p_state : None or np.ndarray [shape=(m,)]
            Optional marginal probability for each label.

        p_init : None or np.ndarray [shape=(m,)]
            Optional marginal probability for each label.

        Returns
        -------
        [(start, end, value)] : iterable of tuples
            where `start` and `end` are the interval boundaries (in seconds)
            and `value` is an np.ndarray, shape=(m,) of the encoded value
            for this interval.
        '''
        if np.isrealobj(encoded):
            if multi:
                if transition is None:
                    encoded = encoded >= 0.5
                else:
                    encoded = viterbi_binary(encoded.T, transition,
                                             p_init=p_init, p_state=p_state).T
            elif sparse and encoded.shape[1] > 1:
                # map to argmax if it's densely encoded (logits)
                if transition is None:
                    encoded = np.argmax(encoded, axis=1)[:, np.newaxis]
                else:
                    encoded = viterbi_discriminative(encoded.T, transition,
                                                     p_init=p_init,
                                                     p_state=p_state)[:, np.newaxis]
            elif not sparse:
                # if dense and multi, map to one-hot encoding
                if transition is None:
                    encoded = (encoded == np.max(encoded, axis=1, keepdims=True))
                else:
                    encoded_ = viterbi_discriminative(encoded.T, transition,
                                                      p_init=p_init,
                                                      p_state=p_state)
                    # Map to one-hot encoding
                    encoded = np.zeros(encoded.shape, dtype=bool)
                    encoded[np.arange(len(encoded_)), encoded_] = True

        if duration is None:
            # 1+ is fair here, because encode_intervals already pads
            duration = 1 + encoded.shape[0]
        else:
            duration = 1 + time_to_frames(duration,
                                          sr=self.sr,
                                          hop_length=self.hop_length)

        # [0, duration] inclusive
        times = times_like(duration + 1,
                           sr=self.sr, hop_length=self.hop_length)

        # Find the change-points of the rows
        if sparse:
            idx = np.where(encoded[1:] != encoded[:-1])[0]
        else:
            idx = np.where(np.max(encoded[1:] != encoded[:-1], axis=-1))[0]

        idx = np.unique(np.append(idx, encoded.shape[0]))
        delta = np.diff(np.append(-1, idx))

        # Starting positions can be integrated from changes
        position = np.cumsum(np.append(0, delta))

        return [(times[p], times[p + d], encoded[p])
                for (p, d) in zip(position, delta)]
