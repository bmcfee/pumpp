#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Key recognition task transformer'''

import numpy as np
import mir_eval
import jams

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

        # using floats as pitch_profile datatype to allow for probabilistic profiles... Need discussion...
        self.register('pitch_profile', [None, 12], np.float32)
        if self.sparse:
            self.register('tonic', [None, 1], np.int)
        else:
            self.register('tonic', [None, 13], np.bool)

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
        #TODO
        raise NotImplementedError

    def inverse(self, pitch_profile, tonic, duration=None):
        #TODO
        raise NotImplementedError