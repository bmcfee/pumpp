#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Chord recognition task transformer'''

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import mir_eval
import jams

from .base import BaseTaskTransformer

__all__ = ['ChordTransformer', 'SimpleChordTransformer']


def _pad_nochord(target, axis=-1):

    ncmask = 1 - np.sum(target, axis=axis, keepdims=True)

    return np.concatenate([target, ncmask], axis=axis)


class ChordTransformer(BaseTaskTransformer):

    def __init__(self, name='chord', sr=22050, hop_length=512):
        '''Initialize a chord task transformer'''

        super(ChordTransformer, self).__init__('chord|chord_harte',
                                               name=name,
                                               fill_na=0,
                                               sr=sr,
                                               hop_length=hop_length)

        pitches = list(range(12))
        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([pitches])
        self._classes = set(self.encoder.classes_)

    def empty(self, duration):
        ann = jams.Annotation(namespace='chord')
        ann.append(time=0,
                   duration=duration,
                   value='N', confidence=0)
        return ann

    def transform_annotation(self, ann, duration):

        # Construct a blank annotation with mask = 0
        intervals, chords = ann.data.to_interval_values()

        # Suppress all intervals not in the encoder
        pitch = []
        root = []
        bass = []

        for c in chords:
            # Encode the pitches
            r, s, b = mir_eval.chord.encode(c)
            s = np.roll(s, r)

            pitch.append(s)

            if r in self._classes:
                root.extend(self.encoder.transform([[r]]))
                bass.extend(self.encoder.transform([[(r+b) % 12]]))
            else:
                root.extend(self.encoder.transform([[]]))
                bass.extend(self.encoder.transform([[]]))

        pitch = np.asarray(pitch)
        root = np.asarray(root)
        bass = np.asarray(bass)

        target_pitch = self.encode_intervals(duration, intervals, pitch)
        target_root = self.encode_intervals(duration, intervals, root)
        target_bass = self.encode_intervals(duration, intervals, bass)

        return {'pitches': target_pitch,
                'root': _pad_nochord(target_root),
                'bass': _pad_nochord(target_bass)}


class SimpleChordTransformer(ChordTransformer):

    def __init__(self, name='chord_simple', sr=22050, hop_length=512):
        '''Initialize a chord task transformer.

        This version of the task includes only pitch classes, but not root or bass.
        '''

        super(SimpleChordTransformer, self).__init__(name=name,
                                                     sr=sr,
                                                     hop_length=hop_length)

    def transform_annotation(self, ann, duration):

        data = super(SimpleChordTransformer, self).transform_annotation(ann, duration)

        data.pop('root', None)
        data.pop('bass', None)
        return data
