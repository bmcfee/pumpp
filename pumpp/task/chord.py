#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Chord recognition task transformer'''

import re
from itertools import product

import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

import mir_eval
import jams

from .base import BaseTaskTransformer
from ..exceptions import ParameterError

__all__ = ['ChordTransformer', 'SimpleChordTransformer', 'ChordTagTransformer']


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

    sparse : bool
        If True, root and bass values are sparsely encoded as integers in [0, 12].
        If False, root and bass values are densely encoded as 13-dimensional booleans.

    See Also
    --------
    SimpleChordTransformer
    '''
    def __init__(self, name='chord', sr=22050, hop_length=512, sparse=False):
        '''Initialize a chord task transformer'''

        super(ChordTransformer, self).__init__(name=name,
                                               namespace='chord',
                                               sr=sr, hop_length=hop_length)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([list(range(12))])
        self._classes = set(self.encoder.classes_)
        self.sparse = sparse

        self.register('pitch', [None, 12], np.bool)
        if self.sparse:
            self.register('root', [None, 1], np.int)
            self.register('bass', [None, 1], np.int)
        else:
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
            data['root'] : np.ndarray, shape=(n, 13) or (n, 1)
            data['bass'] : np.ndarray, shape=(n, 13) or (n, 1)

            `pitch` is a binary matrix indicating pitch class
            activation at each frame.

            `root` is a one-hot matrix indicating the chord
            root's pitch class at each frame.

            `bass` is a one-hot matrix indicating the chord
            bass (lowest note) pitch class at each frame.

            If sparsely encoded, `root` and `bass` are integers
            in the range [0, 12] where 12 indicates no chord.

            If densely encoded, `root` and `bass` have an extra
            final dimension which is active when there is no chord
            sounding.
        '''
        # Construct a blank annotation with mask = 0
        intervals, chords = ann.to_interval_values()

        # Get the dtype for root/bass
        if self.sparse:
            dtype = np.int
        else:
            dtype = np.bool

        # If we don't have any labeled intervals, fill in a no-chord
        if not chords:
            intervals = np.asarray([[0, duration]])
            chords = ['N']

        # Suppress all intervals not in the encoder
        pitches = []
        roots = []
        basses = []

        # default value when data is missing
        if self.sparse:
            fill = 12
        else:
            fill = False

        for chord in chords:
            # Encode the pitches
            root, semi, bass = mir_eval.chord.encode(chord)
            pitches.append(np.roll(semi, root))

            if self.sparse:
                if root in self._classes:
                    roots.append([root])
                    basses.append([(root + bass) % 12])
                else:
                    roots.append([fill])
                    basses.append([fill])
            else:
                if root in self._classes:
                    roots.extend(self.encoder.transform([[root]]))
                    basses.extend(self.encoder.transform([[(root + bass) % 12]]))
                else:
                    roots.extend(self.encoder.transform([[]]))
                    basses.extend(self.encoder.transform([[]]))

        pitches = np.asarray(pitches, dtype=np.bool)
        roots = np.asarray(roots, dtype=dtype)
        basses = np.asarray(basses, dtype=dtype)

        target_pitch = self.encode_intervals(duration, intervals, pitches)

        target_root = self.encode_intervals(duration, intervals, roots,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)
        target_bass = self.encode_intervals(duration, intervals, basses,
                                            multi=False,
                                            dtype=dtype,
                                            fill=fill)

        if not self.sparse:
            target_root = _pad_nochord(target_root)
            target_bass = _pad_nochord(target_bass)

        return {'pitch': target_pitch,
                'root': target_root,
                'bass': target_bass}

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


'''A list of normalized pitch class names'''
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


'''A mapping of chord quality encodings to their names'''
QUALITIES = {
    0b000100000000: 'min',
    0b000010000000: 'maj',
    0b000100010000: 'min',
    0b000010010000: 'maj',
    0b000100100000: 'dim',
    0b000010001000: 'aug',
    0b000100010010: 'min7',
    0b000010010001: 'maj7',
    0b000010010010: '7',
    0b000100100100: 'dim7',
    0b000100100010: 'hdim7',
    0b000100010001: 'minmaj7',
    0b000100010100: 'min6',
    0b000010010100: 'maj6',
    0b001000010000: 'sus2',
    0b000001010000: 'sus4'
}


class ChordTagTransformer(BaseTaskTransformer):
    '''Chord transformer that uses a tag-space encoding for chord labels.

    Attributes
    ----------
    name : str
        name of the transformer

    vocab : str

        A string of chord quality indicators to include:

            - '3': maj/min
            - '5': '3' + aug/dim
            - '6': '3' + '5' + maj6/min6
            - '7': '3' + '5' + '6' + 7/min7/maj7/dim7/hdim7/minmaj7
            - 's': sus2/sus4

        Note: 5 requires 3, 6 requires 5, 7 requires 6.

    sr : number > 0
        Sampling rate of audio

    hop_length : int > 0
        Hop length for annotation frames

    See Also
    --------
    ChordTransformer
    SimpleChordTransformer
    '''
    def __init__(self, name='chord', vocab='3567s',
                 sr=22050, hop_length=512, sparse=False):

        super(ChordTagTransformer, self).__init__(name=name,
                                                  namespace='chord',
                                                  sr=sr,
                                                  hop_length=hop_length)

        # Stringify and lowercase
        if set(vocab) - set('3567s'):
            raise ParameterError('Invalid vocabulary string: {}'.format(vocab))

        if '5' in vocab and '3' not in vocab:
            raise ParameterError('Invalid vocabulary string: {}'.format(vocab))

        if '6' in vocab and '5' not in vocab:
            raise ParameterError('Invalid vocabulary string: {}'.format(vocab))

        if '7' in vocab and '6' not in vocab:
            raise ParameterError('Invalid vocabulary string: {}'.format(vocab))

        self.vocab = vocab.lower()
        labels = self.vocabulary()
        self.sparse = sparse

        if self.sparse:
            self.encoder = LabelEncoder()
        else:
            self.encoder = LabelBinarizer()
        self.encoder.fit(labels)
        self._classes = set(self.encoder.classes_)

        # Construct the quality mask for chord encoding
        self.mask_ = 0b000000000000
        if '3' in self.vocab:
            self.mask_ |= 0b000110000000
        if '5' in self.vocab:
            self.mask_ |= 0b000110111000
        if '6' in self.vocab:
            self.mask_ |= 0b000110010100
        if '7' in self.vocab:
            self.mask_ |= 0b000110110111
        if 's' in self.vocab:
            self.mask_ |= 0b001001010000

        if self.sparse:
            self.register('chord', [None, 1], np.int)
        else:
            self.register('chord', [None, len(self._classes)], np.bool)

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
        ann = super(ChordTagTransformer, self).empty(duration)

        ann.append(time=0,
                   duration=duration,
                   value='X', confidence=0)

        return ann

    def vocabulary(self):
        qualities = []

        if '3' in self.vocab or '5' in self.vocab:
            qualities.extend(['min', 'maj'])

        if '5' in self.vocab:
            qualities.extend(['dim', 'aug'])

        if '6' in self.vocab:
            qualities.extend(['min6', 'maj6'])

        if '7' in self.vocab:
            qualities.extend(['min7', 'maj7', '7', 'dim7', 'hdim7', 'minmaj7'])

        if 's' in self.vocab:
            qualities.extend(['sus2', 'sus4'])

        labels = ['N', 'X']

        for chord in product(PITCHES, qualities):
            labels.append('{}:{}'.format(*chord))

        return labels

    def simplify(self, chord):
        '''Simplify a chord string down to the vocabulary space'''
        # Drop inversions
        chord = re.sub(r'/.*$', r'', chord)
        # Drop any additional or suppressed tones
        chord = re.sub(r'\(.*?\)', r'', chord)
        # Drop dangling : indicators
        chord = re.sub(r':$', r'', chord)

        # Encode the chord
        root, pitches, _ = mir_eval.chord.encode(chord)

        # Build the query
        # To map the binary vector pitches down to bit masked integer,
        # we just dot against powers of 2
        P = 2**np.arange(12, dtype=int)
        query = self.mask_ & pitches[::-1].dot(P)

        if root < 0 and chord[0].upper() == 'N':
            return 'N'
        if query not in QUALITIES:
            return 'X'

        return '{}:{}'.format(PITCHES[root], QUALITIES[query])

    def transform_annotation(self, ann, duration):
        '''Transform an annotation to chord-tag encoding

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['chord'] : np.ndarray, shape=(n, n_labels)
                A time-varying binary encoding of the chords
        '''

        intervals, values = ann.to_interval_values()

        chords = []
        for v in values:
            chords.extend(self.encoder.transform([self.simplify(v)]))

        dtype = self.fields[self.scope('chord')].dtype

        chords = np.asarray(chords)

        if self.sparse:
            chords = chords[:, np.newaxis]

        target = self.encode_intervals(duration, intervals, chords,
                                       multi=False, dtype=dtype)

        return {'chord': target}

    def inverse(self, encoded, duration=None):
        '''Inverse transformation'''

        ann = jams.Annotation(self.namespace, duration=duration)

        for start, end, value in self.decode_intervals(encoded,
                                                       duration=duration,
                                                       multi=False,
                                                       sparse=self.sparse):
            if self.sparse:
                value_dec = self.encoder.inverse_transform(value)
            else:
                value_dec = self.encoder.inverse_transform(np.atleast_2d(value))

            for vd in value_dec:
                ann.append(time=start, duration=end-start, value=vd)

        return ann
