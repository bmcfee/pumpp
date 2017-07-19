#!/usr/bin/env python
# -*- enconding: utf-8 -*-
'''Instantaneous event coding'''

import numpy as np

import jams
from mir_eval.util import boundaries_to_intervals, adjust_intervals
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from .base import BaseTaskTransformer
from ..exceptions import ParameterError

__all__ = ['BeatTransformer', 'BeatPositionTransformer']


class BeatTransformer(BaseTaskTransformer):
    '''Task transformation for beat tracking

    Attributes
    ----------
    name : str
        The name of this transformer

    sr : number > 0
        The audio sampling rate

    hop_length : int > 0
        The hop length for annotation frames
    '''
    def __init__(self, name='beat', sr=22050, hop_length=512):
        super(BeatTransformer, self).__init__(name=name,
                                              namespace='beat',
                                              sr=sr, hop_length=hop_length)

        self.register('beat', [None], np.bool)
        self.register('downbeat', [None], np.bool)
        self.register('mask_downbeat', [1], np.bool)

    def transform_annotation(self, ann, duration):
        '''Apply the beat transformer

        Parameters
        ----------
        ann : jams.Annotation
            The input annotation

        duration : number > 0
            The duration of the audio

        Returns
        -------
        data : dict
            data['beat'] : np.ndarray, shape=(n, 1)
                Binary indicator of beat/non-beat

            data['downbeat'] : np.ndarray, shape=(n, 1)
                Binary indicator of downbeat/non-downbeat

            mask_downbeat : bool
                True if downbeat annotations are present
        '''

        mask_downbeat = False

        intervals, values = ann.to_interval_values()
        values = np.asarray(values)

        beat_events = intervals[:, 0]
        beat_labels = np.ones((len(beat_events), 1))

        idx = (values == 1)
        if np.any(idx):
            downbeat_events = beat_events[idx]
            downbeat_labels = np.ones((len(downbeat_events), 1))
            mask_downbeat = True
        else:
            downbeat_events = np.zeros(0)
            downbeat_labels = np.zeros((0, 1))

        target_beat = self.encode_events(duration,
                                         beat_events,
                                         beat_labels)

        target_downbeat = self.encode_events(duration,
                                             downbeat_events,
                                             downbeat_labels)

        return {'beat': target_beat,
                'downbeat': target_downbeat,
                'mask_downbeat': mask_downbeat}

    def inverse(self, encoded, downbeat=None, duration=None):
        '''Inverse transformation for beats and optional downbeats'''

        ann = jams.Annotation(namespace=self.namespace, duration=duration)

        beat_times = [t for t, _ in self.decode_events(encoded) if _]
        if downbeat is not None:
            downbeat_times = set([t for t, _ in self.decode_events(downbeat)
                                  if _])
            pickup_beats = len([t for t in beat_times
                                if t < min(downbeat_times)])
        else:
            downbeat_times = set()
            pickup_beats = 0

        value = - pickup_beats - 1
        for beat in beat_times:
            if beat in downbeat_times:
                value = 1
            else:
                value += 1
            ann.append(time=beat, duration=0, value=value)

        return ann


class BeatPositionTransformer(BaseTaskTransformer):
    '''Encode beat- and downbeat-annotations as labeled intervals.

    This transformer assumes that the `value` field of a beat annotation
    encodes its metrical position (1, 2, 3, 4, ...).

    A `value` of 0 indicates that the beat does not belong to a bar,
    and should be used to indicate pickup beats.

    Beat position strings are coded as SUBDIVISION/POSITION

    For example, in 4/4 time, the 2 beat would be coded as "04/02".
    '''
    def __init__(self, name, max_divisions=12,
                 sr=22050, hop_length=512, sparse=False):

        super(BeatPositionTransformer, self).__init__(name=name,
                                                      namespace='beat',
                                                      sr=sr,
                                                      hop_length=hop_length)

        # Make the vocab set
        if not isinstance(max_divisions, int) or max_divisions < 1:
            raise ParameterError('Invalid max_divisions={}'.format(max_divisions))

        self.max_divisions = max_divisions
        labels = self.vocabulary()
        self.sparse = sparse

        if self.sparse:
            self.encoder = LabelEncoder()
        else:
            self.encoder = LabelBinarizer()
        self.encoder.fit(labels)
        self._classes = set(self.encoder.classes_)

        if self.sparse:
            self.register('position', [None, 1], np.int)
        else:
            self.register('position', [None, len(self._classes)], np.bool)

    def vocabulary(self):
        states = ['X']
        for d in range(1, self.max_divisions + 1):
            for n in range(1, d + 1):
                states.append('{:02d}/{:02d}'.format(d, n))
        return states

    def transform_annotation(self, ann, duration):
        '''Transform an annotation to the beat-position encoding

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['position'] : np.ndarray, shape=(n, n_labels) or (n, 1)
                A time-varying label encoding of beat position
        '''

        # 1. get all the events
        # 2. find all the downbeats
        # 3. map each downbeat to a subdivision counter
        #       number of beats until the next downbeat
        # 4. pad out events to intervals
        # 5. encode each beat interval to its position

        boundaries, values = ann.to_interval_values()
        # Convert to intervals and span the duration
        # padding at the end of track does not propagate the right label
        # this is an artifact of inferring end-of-track from boundaries though
        boundaries = list(boundaries[:, 0])
        if boundaries and boundaries[-1] < duration:
            boundaries.append(duration)
        intervals = boundaries_to_intervals(boundaries)
        intervals, values = adjust_intervals(intervals, values,
                                             t_min=0,
                                             t_max=duration,
                                             start_label=0,
                                             end_label=0)

        values = np.asarray(values, dtype=int)
        downbeats = np.flatnonzero(values == 1)

        position = []
        for i, v in enumerate(values):
            # If the value is a 0, mark it as X and move on
            if v == 0:
                position.extend(self.encoder.transform(['X']))
                continue

            # Otherwise, let's try to find the surrounding downbeats
            prev_idx = np.searchsorted(downbeats, i, side='right') - 1
            next_idx = 1 + prev_idx

            if prev_idx >= 0 and next_idx < len(downbeats):
                # In this case, the subdivision is well-defined
                subdivision = downbeats[next_idx] - downbeats[prev_idx]
            elif prev_idx < 0 and next_idx < len(downbeats):
                subdivision = np.max(values[:downbeats[0]+1])
            elif next_idx >= len(downbeats):
                subdivision = len(values) - downbeats[prev_idx]

            if subdivision > self.max_divisions or subdivision < 1:
                position.extend(self.encoder.transform(['X']))
            else:
                position.extend(self.encoder.transform(['{:02d}/{:02d}'.format(subdivision, v)]))

        dtype = self.fields[self.scope('position')].dtype

        position = np.asarray(position)
        if self.sparse:
            position = position[:, np.newaxis]

        target = self.encode_intervals(duration, intervals, position,
                                       multi=False, dtype=dtype)
        return {'position': target}

    def inverse(self, encoded, duration=None):
        '''Inverse transformation'''

        raise NotImplementedError
