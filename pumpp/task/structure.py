#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Segment and structure tasks'''

import numpy as np
from mir_eval.util import intervals_to_samples, index_labels, adjust_intervals

from .base import BaseTaskTransformer

__all__ = ['StructureTransformer']


class StructureTransformer(BaseTaskTransformer):
    '''Structure agreement transformer.

    This transformer maps a labeled, flat structural segmentation
    to an `n*n` boolean matrix indicating whether two frames
    belong to a similarly labeled segment or not.

    Attributes
    ----------
    name : str
        The name of this transformer

    sr : number > 0
        The audio sampling rate

    hop_length : int > 0
        The number of samples between each annotation frame
    '''

    def __init__(self, name='structure', sr=22050, hop_length=512):
        '''Initialize a structure agreement transformer'''

        super(StructureTransformer, self).__init__(name=name,
                                                   namespace='segment_open',
                                                   sr=sr,
                                                   hop_length=hop_length)

        self.register('agree', [None, None], np.bool)

    def empty(self, duration):
        ann = super(StructureTransformer, self).empty(duration)
        ann.append(time=0, duration=duration, value='none', confidence=0)
        return ann

    def transform_annotation(self, ann, duration):
        '''Apply the structure agreement transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The segment annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['agree'] : np.ndarray, shape=(n, n), dtype=bool
        '''

        intervals, values = ann.to_interval_values()

        intervals, values = adjust_intervals(intervals, values,
                                             t_min=0, t_max=duration)
        # Re-index the labels
        ids, _ = index_labels(values)

        rate = float(self.hop_length) / self.sr
        # Sample segment labels on our frame grid
        _, labels = intervals_to_samples(intervals, ids, sample_size=rate)

        # Make the agreement matrix
        return {'agree': np.equal.outer(labels, labels)}

    def inverse(self, agree, duration=None):

        raise NotImplementedError('Segment agreement cannot be inverted')
