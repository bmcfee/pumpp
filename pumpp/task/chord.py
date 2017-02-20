#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Chord recognition task transformer'''

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import mir_eval

from .base import BaseTaskTransformer

__all__ = ['ChordTransformer', 'SimpleChordTransformer']


def _pad_nochord(target, axis=-1):
    '''Pad a chord annotation with no-chord flags.

    Parameters
    ----------
    target : np.ndarray
        the input data

    axis : int
        the axis along which to pad

    Returns
    -------
    target_pad
        `target` expanded by 1 along the specified `axis`.
        The expanded dimension will be 0 when `target` is non-zero
        before padding, and 1 otherwise.
    '''
    ncmask = ~np.max(target, axis=axis, keepdims=True)

    return np.concatenate([target, ncmask], axis=axis)


class ChordTransformer(BaseTaskTransformer):
    '''Chord annotation transformers.

    This transformer uses a (pitch, root, bass) decomposition of
    chord annotations.

    Attributes
    ----------
    name : str
        The name of the chord transformer

    sr : number > 0
        The sampling rate of audio

    hop_length : int > 0
        The number of samples between each annotation frame

    See Also
    --------
    SimpleTransformer
    '''
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
        '''Empty chord annotations

        Parameters
        ----------
        duration : number
            The length (in seconds) of the empty annotation

        Returns
        -------
        ann : jams.Annotation
            A chord annotation consisting of a single `no-chord` observation.
        '''
        ann = super(ChordTransformer, self).empty(duration)

        ann.append(time=0,
                   duration=duration,
                   value='N', confidence=0)

        return ann

    def transform_annotation(self, ann, duration):
        '''Apply the chord transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The chord annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['pitch'] : np.ndarray, shape=(n, 12)
            data['root'] : np.ndarray, shape=(n, 13)
            data['bass'] : np.ndarray, shape=(n, 13)

            `pitch` is a binary matrix indicating pitch class
            activation at each frame.

            `root` is a one-hot matrix indicating the chord
            root's pitch class at each frame.

            `bass` is a one-hot matrix indicating the chord
            bass (lowest note) pitch class at each frame.

            `root` and `bass` have an extra final dimension
            which is active when there is no chord sounding.
        '''
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

    def inverse(self, pitch, root, bass, duration=None):

        raise NotImplementedError('Chord cannot be inverted')


class SimpleChordTransformer(ChordTransformer):
    '''Simplified chord transformations.  Only pitch class activity is encoded.

    Attributes
    ----------
    name : str
        name of the transformer

    sr : number > 0
        Sampling rate of audio

    hop_length : int > 0
        Hop length for annotation frames

    See Also
    --------
    ChordTransformer
    '''
    def __init__(self, name='chord', sr=22050, hop_length=512):
        super(SimpleChordTransformer, self).__init__(name=name,
                                                     sr=sr,
                                                     hop_length=hop_length)
        # Remove the extraneous fields
        self.pop('root')
        self.pop('bass')

    def transform_annotation(self, ann, duration):
        '''Apply the chord transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The chord annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['pitch'] : np.ndarray, shape=(n, 12)

            `pitch` is a binary matrix indicating pitch class
            activation at each frame.
        '''
        data = super(SimpleChordTransformer,
                     self).transform_annotation(ann, duration)

        data.pop('root', None)
        data.pop('bass', None)
        return data

    def inverse(self, *args, **kwargs):
        raise NotImplementedError('SimpleChord cannot be inverted')
