#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Chord recognition task transformer'''

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import mir_eval

from .base import BaseTaskTransformer

__all__ = ['ChordTransformer', 'SimpleChordTransformer']


def _pad_nochord(target, axis=-1):

    ncmask = ~np.max(target, axis=axis, keepdims=True)

    return np.concatenate([target, ncmask], axis=axis)


class ChordTransformer(BaseTaskTransformer):

    def __init__(self, name='chord', sr=22050, hop_length=512):
        '''Initialize a chord task transformer'''

        super(ChordTransformer, self).__init__(name=name,
                                               namespace='chord',
                                               sr=sr, hop_length=hop_length)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([list(range(12))])
        self._classes = set(self.encoder.classes_)
        self.register('pitch', [None, 12], np.bool)
        self.register('root', [None, 13], np.bool)
        self.register('bass', [None, 13], np.bool)

    def empty(self, duration):
        ann = super(ChordTransformer, self).empty(duration)

        ann.append(time=0,
                   duration=duration,
                   value='N', confidence=0)

        return ann

    def transform_annotation(self, ann, duration):

        # Construct a blank annotation with mask = 0
        intervals, chords = ann.data.to_interval_values()
        
        # If we don't have any labeled intervals, fill in a no-chord
        if not chords:
            intervals = np.asarray([[0, duration]])
            chords = ['N']

        # Suppress all intervals not in the encoder
        pitches = []
        roots = []
        basses = []

        for chord in chords:
            # Encode the pitches
            root, semi, bass = mir_eval.chord.encode(chord)
            pitches.append(np.roll(semi, root))

            if root in self._classes:
                roots.extend(self.encoder.transform([[root]]))
                basses.extend(self.encoder.transform([[(root + bass) % 12]]))
            else:
                roots.extend(self.encoder.transform([[]]))
                basses.extend(self.encoder.transform([[]]))

        pitches = np.asarray(pitches, dtype=np.bool)
        roots = np.asarray(roots, dtype=np.bool)
        basses = np.asarray(basses, dtype=np.bool)

        target_pitch = self.encode_intervals(duration, intervals, pitches)
        target_root = self.encode_intervals(duration, intervals, roots)
        target_bass = self.encode_intervals(duration, intervals, basses)

        return {'pitch': target_pitch,
                'root': _pad_nochord(target_root),
                'bass': _pad_nochord(target_bass)}


class SimpleChordTransformer(ChordTransformer):

    def __init__(self, name='chord', sr=22050, hop_length=512):
        super(SimpleChordTransformer, self).__init__(name=name,
                                                     sr=sr,
                                                     hop_length=hop_length)
        # Remove the extraneous fields
        self.pop('root')
        self.pop('bass')

    def transform_annotation(self, ann, duration):

        data = super(SimpleChordTransformer, self).transform_annotation(ann, duration)

        data.pop('root', None)
        data.pop('bass', None)
        return data
