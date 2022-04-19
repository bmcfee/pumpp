#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Key recognition task transformer'''

from itertools import product
import logging

import numpy as np
import mir_eval
import jams

from librosa import note_to_midi, midi_to_note, time_to_frames, key_to_degrees
from librosa.sequence import transition_loop

from .base import BaseTaskTransformer
from ..exceptions import ParameterError
from ..labels import LabelBinarizer, LabelEncoder, MultiLabelBinarizer

__all__ = ['KeyTransformer', 'KeyTagTransformer']

C_MAJOR_PITCHES = key_to_degrees('C:maj')
MODES = ['ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']
QUALITY = {'major' : 0, 'minor' : -3}


def _encode_key_str(key_str, sparse):
        '''Helper function to go from jams `key_mode` annotation value strings to 12-D 
        numpy membership vec, representing the pitch profile.

        Parameters
        ----------
        key_str : str
            String in the style of 'key_mode' jams annotation values.
        sparse : bool
            Whether or not to use sparse encoding for the tonic field.

        Returns
        -------
        (pitch_profile, tonic) : tuple
            pitch_profile : np.ndarray, shape = (1, 12), dtype = bool
                a 12-D row vector that's encodes the membership of each pitch class for 
                a given `key_str`.
            tonic : int or np.ndarray, shape = (1, 13), dtype = bool
                a int in the range [0, 12] to indicate the pitch class of the tonic. 12
                being atonal. The type will depend on the `sparse` parameter
        '''
        
        key_str_split = key_str.split(':')
        
        # Look at the Tonic first
        if key_str_split[0] == 'N':
            tonic = 12
        else:
            tonic = note_to_midi(key_str_split[0]) % 12

        # Now look at quality/mode and build pitch_profile
        # First construct the profile in C for a given mode/quality
        c_major_profile = np.zeros(12)
        for pc in C_MAJOR_PITCHES:
            c_major_profile[pc] = 1

        # When there is no tonal center, pitch profile is all zeros.
        if tonic == 12:
            pitch_profile = np.zeros(12, dtype=bool)
        else:
            # When there is no quality, major assumed.
            if len(key_str_split) == 1:
                quality = 'major'
            else:
                quality = key_str_split[1]

            if quality in MODES:
                mode_transpose_int = -1 * C_MAJOR_PITCHES[MODES.index(quality)]
            elif quality in QUALITY.keys():
                mode_transpose_int = -1 * QUALITY[quality]
            else:
                logging.info(
                    '{} is not a recognized quality. Using major instead.'.format(quality)
                )
                mode_transpose_int = 0

            # roll the profile to fit different modes.        
            mode_profile_in_c = np.roll(c_major_profile, mode_transpose_int)
            # Add the leading tone to the minor profiles
            if quality == 'minor':
                mode_profile_in_c[11] = 1
            
            # Now roll the profile again to get the right tonic.
            pitch_profile = np.roll(mode_profile_in_c, tonic)

        if not sparse:
            tonic_vec = np.zeros(13, dtype=bool)
            tonic_vec[tonic] = 1
            tonic = tonic_vec

        return (pitch_profile, tonic)


class KeyTransformer(BaseTaskTransformer):
    '''Key annotation transformer.

    This transformer uses a (pitch_profile, tonic) decomposition of key_mode 
    annotation, where the mode is reflected in the 12-D pitch_profile vector.

    Attributes
    ----------
    name : str
        The name of the key trnsformer

    sr : number > 0
        The sampling rate of audio

    hop_length : int > 0
        The number of samples between each annotation frame

    sparse : bool
        If True, tonic value is sparsely encoded as integers in [0, 12].
        If False, tonic value is densely encoded as 13-dimensional booleans.
    '''
    def __init__(self, name='key', sr=22050, hop_length=512, sparse=False):
        '''Initialize a key task transformer'''

        super(KeyTransformer, self).__init__(name=name,
                                             namespace='key_mode',
                                             sr=sr, hop_length=hop_length)
        self.sparse = sparse
 
        self.register('pitch_profile', [None, 12], bool)
        if self.sparse:
            self.register('tonic', [None, 1], int)
        else:
            self.register('tonic', [None, 13], bool)

    
    def empty(self, duration):
        '''Empty key annotation

        Parameters
        ----------
        duration : number
            The length (in seconds) of the empty annotation

        Returns
        -------
        ann : jams.Annotation
            A key_mode annotation consisting of a single `no-key` observation.
        '''
        ann = super(KeyTransformer, self).empty(duration)

        ann.append(time=0,
                   duration=duration,
                   value='N', confidence=0)

        return ann

    def transform_annotation(self, ann, duration):
        '''Apply the key transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The key_mode annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['pitch_profile'] : np.ndarray, shape=(n, 12)
            data['tonic'] : np.ndarray, shape=(n, 13) or (n, 1)

            `pitch_profile` is a binary matrix indicating pitch class
            activation at each frame.

            `tonic` is a one-hot matrix indicating the tonal center's 
            pitch class at each frame.

            If sparsely encoded, `tonic` is a integer
            in the range [0, 12] where 12 indicates atonal.

            If densely encoded, `tonic` has an extra
            final dimension which is active when it is atonal.
        '''
        # get list of observations
        intervals, keys = ann.to_interval_values()

        # Get the dtype for tonic
        if self.sparse:
            dtype = int
        else:
            dtype = bool

        # If we don't have any labeled intervals, fill in a 'N'
        if not keys:
            intervals = np.asarray([[0, duration]])
            keys = ['N']

        # Suppress all intervals not in the encoder
        pitch_profiles = []
        tonics = []

        # default value when data is missing
        if self.sparse:
            fill = 12
        else:
            fill = False

        for key in keys:
            pitch_profile, tonic = _encode_key_str(key, self.sparse)
            pitch_profiles.append(pitch_profile)
            tonics.append(tonic if isinstance(tonic, np.ndarray) else [tonic])
        
        pitch_profiles = np.asarray(pitch_profiles, dtype=bool)
        tonics = np.asarray(tonics, dtype=dtype)

        target_pitch_profile = self.encode_intervals(duration, intervals, pitch_profiles)

        target_tonic = self.encode_intervals(duration, intervals, tonics,
                                             multi=False,
                                             dtype=dtype,
                                             fill=fill)

        return {'pitch_profile': target_pitch_profile,
                'tonic': target_tonic}

    def inverse(self, pitch_profile, tonic, duration=None):
        raise NotImplementedError('There are some ambiguities, also streaming profiles are difficult')

class KeyTagTransformer(BaseTaskTransformer):
    '''Chord transformer that uses a tag-space encoding for key labels.

    Attributes
    ----------
    name : str
        name of the transformer

    sr : number > 0
        Sampling rate of audio

    hop_length : int > 0
        Hop length for annotation frames

    sparse : Bool
        Whether or not to use sparse encoding for the labels

    p_self : None, float in (0, 1), or np.ndarray [shape=(n_labels,)]
        Optional self-loop probability(ies), used for Viterbi decoding

    p_state : None or np.ndarray [shape=(n_labels,)]
        Optional marginal probability for each chord class

    p_init : None or np.ndarray [shape=(n_labels,)]
        Optional initial probability for each chord class

    See Also
    --------
    KeyTransformer
    ChordTagTransformer
    '''
    def __init__(self, name='key_tag',
                 sr=22050, hop_length=512, sparse=False,
                 p_self=None, p_init=None, p_state=None):

        super(KeyTagTransformer, self).__init__(name=name,
                                                namespace='key_mode',
                                                sr=sr,
                                                hop_length=hop_length)
        
        labels = self.vocabulary()
        self.sparse = sparse

        if self.sparse:
            self.encoder = LabelEncoder()
        else:
            self.encoder = LabelBinarizer()
        self.encoder.fit(labels)
        self._classes = set(self.encoder.classes_)

        self.set_transition(p_self)

        if p_init is not None:
            if len(p_init) != len(self._classes):
                raise ParameterError('Invalid p_init.shape={} for vocabulary of size {}'.format(p_init.shape, len(self._classes)))

        self.p_init = p_init

        if p_state is not None:
            if len(p_state) != len(self._classes):
                raise ParameterError('Invalid p_state.shape={} for vocabulary of size {}'.format(p_state.shape, len(self._classes)))

        self.p_state = p_state

        if self.sparse:
            self.register('tag', [None, 1], int)
        else:
            self.register('tag', [None, len(self._classes)], bool)

    def set_transition(self, p_self):
        '''Set the transition matrix according to self-loop probabilities.

        Parameters
        ----------
        p_self : None, float in (0, 1), or np.ndarray [shape=(n_labels,)]
            Optional self-loop probability(ies), used for Viterbi decoding
        '''
        if p_self is None:
            self.transition = None
        else:
            self.transition = transition_loop(len(self._classes), p_self)

    def empty(self, duration):
        '''Empty key annotations

        Parameters
        ----------
        duration : number
            The length (in seconds) of the empty annotation

        Returns
        -------
        ann : jams.Annotation
            A key annotation consisting of a single `N` observation.
        '''
        ann = super(KeyTagTransformer, self).empty(duration)

        ann.append(time=0,
                   duration=duration,
                   value='N', confidence=0)

        return ann

    def vocabulary(self):
        ''' Build the vocabulary for all key_mode strings

        Returns
        -------
        labels : list
            list of string labels.
        '''
        qualities = MODES + list(QUALITY.keys())
        tonics = midi_to_note(list(range(12)), octave=False, unicode=False)
        
        labels = ['N']

        for key_mode in product(tonics, qualities):
            labels.append('{}:{}'.format(*key_mode))

        return labels

    def enharmonic(self, key_str):
        '''Force the tonic spelling to fit our tonic list 
        by spelling out of vocab keys enharmonically.

        Parameters
        ----------
        key_str : str
            The key_mode string in jams style.

        Returns
        -------
        key_str : str
            The key_mode string spelled enharmonically to fit our vocab.
        '''
        key_list = key_str.split(':')
        # spell the tonic enharmonically if necessary
        if key_list[0] != 'N':
            key_list[0] = midi_to_note(note_to_midi(key_list[0]), octave=False, unicode=False)
            if len(key_list) == 1:
                key_list.append('major')

        return ':'.join(key_list)

    def transform_annotation(self, ann, duration):
        '''Transform an annotation to key-tag encoding

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            if self.sparse = True
            data['tag'] : np.ndarray, shape=(n, n_labels) or shape=(n,)
                A time-varying binary encoding of the keys. 
                The shape depends on self.sparse.
        '''
        intervals, values = ann.to_interval_values()

        keys = []
        for v in values:
            keys.extend(self.encoder.transform([self.enharmonic(v)]))

        dtype = self.fields[self.scope('tag')].dtype

        keys = np.asarray(keys)

        if self.sparse:
            keys = keys[:, np.newaxis]
        
        target = self.encode_intervals(duration, intervals, keys,
                                       multi=False, dtype=dtype)

        return {'tag': target}

    def inverse(self, encoded, duration=None):
        '''Inverse transformation'''

        ann = jams.Annotation(self.namespace, duration=duration)
            
        for start, end, value in self.decode_intervals(encoded,
                                                       duration=duration,
                                                       multi=False,
                                                       sparse=self.sparse,
                                                       transition=self.transition,
                                                       p_init=self.p_init,
                                                       p_state=self.p_state):

            # Map start:end to frames
            f_start, f_end = time_to_frames([start, end],
                                            sr=self.sr,
                                            hop_length=self.hop_length)

            # Reverse the index
            if self.sparse:
                # Compute the confidence
                if encoded.shape[1] == 1:
                    # This case is for full-confidence prediction (just the index)
                    confidence = 1.
                else:
                    confidence = np.mean(encoded[f_start:f_end+1, value])

                value_dec = self.encoder.inverse_transform(value)
            else:
                confidence = np.mean(encoded[f_start:f_end+1, np.argmax(value)])
                value_dec = self.encoder.inverse_transform(np.atleast_2d(value))

            for vd in value_dec:
                ann.append(time=start,
                           duration=end-start,
                           value=vd,
                           confidence=float(confidence))

        return ann
