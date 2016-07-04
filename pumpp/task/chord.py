#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Chord recognition task transformer'''

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import mir_eval

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
        self.name = name

    def transform(self, jam):

        ann = self.find_annotation(jam)

        # Construct a blank annotation with mask = 0
        intervals = np.asarray([[0.0, jam.file_metadata.duration]])
        chords = ['N']
        mask = False
        if ann:
            ann_ints, ann_chords = ann.data.to_interval_values()
            intervals = np.vstack([intervals, ann_ints])
            chords.extend(ann_chords)
            mask = True

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

        target_pitch = self.encode_intervals(jam.file_metadata.duration,
                                             intervals, pitch)
        target_root = self.encode_intervals(jam.file_metadata.duration,
                                            intervals, root)
        target_bass = self.encode_intervals(jam.file_metadata.duration,
                                            intervals, bass)

        return {'{:s}_pitches'.format(self.name): target_pitch,
                '{:s}_root'.format(self.name): _pad_nochord(target_root),
                '{:s}_bass'.format(self.name): _pad_nochord(target_bass),
                'mask_{:s}'.format(self.name): mask}


class SimpleChordTransformer(BaseTaskTransformer):

    def __init__(self, name='chord_simple', sr=22050, hop_length=512):
        '''Initialize a chord task transformer.

        This version of the task includes only pitch classes, but not root or bass.
        '''

        super(SimpleChordTransformer, self).__init__('chord|chord_harte',
                                                     name=name,
                                                     fill_na=0,
                                                     sr=sr,
                                                     hop_length=hop_length)


        pitches = list(range(12))
        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([pitches])
        self._classes = set(self.encoder.classes_)
        self.name = name

    def transform(self, jam):

        ann = self.find_annotation(jam)

        # Construct a blank annotation with mask = 0
        intervals = np.asarray([[0.0, jam.file_metadata.duration]])
        chords = ['N']
        mask = False
        if ann:
            ann_ints, ann_chords = ann.data.to_interval_values()
            intervals = np.vstack([intervals, ann_ints])
            chords.extend(ann_chords)
            mask = True

        # Suppress all intervals not in the encoder
        pitch = []

        for c in chords:
            # Encode the pitches
            r, s, b = mir_eval.chord.encode(c)
            s = np.roll(s, r)

            pitch.append(s)

        pitch = np.asarray(pitch)

        target_pitch = self.encode_intervals(jam.file_metadata.duration,
                                             intervals, pitch)

        return {'{:s}_pitches'.format(self.name): target_pitch,
                'mask_{:s}'.format(self.name): mask}

