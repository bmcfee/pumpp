#!/usr/bin/env python
# -*- enconding: utf-8 -*-
'''Instantaneous event coding'''

import numpy as np
import jams

from .base import BaseTaskTransformer

__all__ = ['BeatTransformer']


class BeatTransformer(BaseTaskTransformer):

    def __init__(self, name='beat', sr=22050, hop_length=512):
        super(BeatTransformer, self).__init__('beat',
                                              name=name,
                                              fill_na=0,
                                              sr=sr,
                                              hop_length=hop_length)

    def empty(self, duration):
        return jams.Annotation(namespace='beat')

    def transform_annotation(self, ann, duration):

        mask_downbeat = False

        intervals, values = ann.data.to_interval_values()
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

        return {'beat': target_beat, 'downbeat': target_downbeat,
                'mask_downbeat': mask_downbeat}
