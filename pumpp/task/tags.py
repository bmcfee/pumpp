#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tag task transformers'''

import numpy as np

from .base import BaseTaskTransformer
from sklearn.preprocessing import MultiLabelBinarizer

# FIXME: rename timeseries, global -> dynamic, static

__all__ = ['TimeSeriesLabelTransformer', 'GlobalLabelTransformer']


class TimeSeriesLabelTransformer(BaseTaskTransformer):

    def __init__(self, namespace, name, labels, sr=22050, hop_length=512):
        '''Initialize a time-series label transformer

        Parameters
        ----------
        jam : jams.JAMS
            The JAMS object container

        n_samples : int > 0
            The number of samples in the audio frame

        label_encoder : sklearn.preprocessing.MultiLabelBinarizer
            The (pre-constructed) label encoder
        '''

        super(TimeSeriesLabelTransformer, self).__init__(namespace, 0,
                                                         sr=sr,
                                                         hop_length=hop_length)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)
        self.name = name

    def transform(self, jam):

        ann = self.find_annotation(jam)

        intervals = np.asarray([[0.0, jam.file_metadata.duration]])
        values = [None]
        mask = False

        if ann:
            ann_int, ann_val = ann.data.to_interval_values()
            intervals = np.vstack([intervals, ann_int])
            values.extend(ann_val)
            mask = True

        # Suppress all intervals not in the encoder
        tags = []
        for v in values:
            if v in self._classes:
                tags.extend(self.encoder.transform([[v]]))
            else:
                tags.extend(self.encoder.transform([[]]))

        tags = np.asarray(tags)
        target = self.encode_intervals(jam.file_metadata.duration,
                                       intervals,
                                       tags)
        return {'output_{:s}'.format(self.name): target,
                'mask_{:s}'.format(self.name): mask}


class GlobalLabelTransformer(BaseTaskTransformer):

    def __init__(self, namespace, name, labels):
        '''Initialize a global label transformer

        Parameters
        ----------
        jam : jams.JAMS
            The JAMS object container
        '''

        super(GlobalLabelTransformer, self).__init__(namespace, 0, sr=1, hop_length=1)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)
        self.name = name

    def transform(self, jam):

        ann = self.find_annotation(jam)

        intervals = np.asarray([[0, 1]])
        values = [None]
        mask = False

        if ann:
            values = list(ann.data.value)
            intervals = np.tile(intervals, [len(values), 1])
            mask = True

        # Suppress all intervals not in the encoder
        tags = [v for v in values if v in self._classes]
        if len(tags):
            target = self.encoder.transform([tags]).max(axis=0)
        else:
            target = np.zeros(len(self._classes), dtype=np.int)

        return {'output_{:s}'.format(self.name): target,
                'mask_{:s}'.format(self.name): mask}
