#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Key recognition task transformer'''

import numpy as np
import mir_eval
import jams

from librosa import note_to_midi

from .base import BaseTaskTransformer
from ..labels import MultiLabelBinarizer

__all__ = ['KeyTransformer']

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

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([list(range(12))])
        self._classes = set(self.encoder.classes_)
        self.sparse = sparse

        # Maybe use floats as pitch_profile datatype to allow for probabilistic profiles?... Need discussion...
        self.register('pitch_profile', [None, 12], np.bool)
        if self.sparse:
            self.register('tonic', [None, 1], np.int)
        else:
            self.register('tonic', [None, 13], np.bool)

    def _encode_key_str(self, key_str):
        '''Helper function to go from jams `key_mode` annotation value strings to 12-D 
        numpy membership vec, representing the pitch profile.

        Parameters
        ----------
        key_str : str
            String in the style of 'key_mode' jams annotation values.

        Returns
        -------
        (pitch_profile, tonic) : tuple
            pitch_profile : np.ndarray, shape = (1, 12), dtype = np.bool
                a 12-D row vector that's encodes the membership of each pitch class for 
                a given `key_str`.
            tonic : int or np.ndarray, shape = (1, 13), dtype = np.bool
                a int in the range [0, 12] to indicate the pitch class of the tonic. 12
                being atonal.
        '''
        C_MAJOR = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        C_MAJOR_PITCHES = note_to_midi(C_MAJOR) % 12
        MODES = ['ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']
        QUALITY = {'major' : 0, 'minor' : -3}

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
            pitch_profile = np.zeros(12, dtype=np.bool)
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

            mode_profile_in_c = np.roll(c_major_profile, mode_transpose_int)
            
            # Now roll the profile again to get the right tonic.
            pitch_profile = np.roll(mode_profile_in_c, tonic)

        if not self.sparse:
            tonic_vec = np.zeros(13, dtype=np.bool)
            tonic_vec[tonic] = 1
            tonic = tonic_vec

        return (pitch_profile, tonic)

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
            final dimension which is active when there it is atonal.
        '''
        # get list of observations
        intervals, keys = ann.to_interval_values()

        # Get the dtype for tonic
        if self.sparse:
            dtype = np.int
        else:
            dtype = np.bool

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
            pitch_profile, tonic = self._encode_key_str(key)
            pitch_profiles.append(pitch_profile)
            tonics.append(tonic if type(tonic) is np.ndarray else [tonic])
        
        pitch_profiles = np.asarray(pitch_profiles, dtype=np.bool)
        tonics = np.asarray(tonics, dtype=dtype)

        target_pitch_profile = self.encode_intervals(duration, intervals, pitch_profiles)

        target_tonic = self.encode_intervals(duration, intervals, tonics,
                                             multi=False,
                                             dtype=dtype,
                                             fill=fill)

        return {'pitch_profile': target_pitch_profile,
                'tonic': target_tonic}

    def inverse(self, pitch_profile, tonic, duration=None):
        raise NotImplementedError('There are some ambiguities')

    