#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tag task transformers'''

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import jams

from .base import BaseTaskTransformer

__all__ = ['DynamicLabelTransformer', 'StaticLabelTransformer']


class DynamicLabelTransformer(BaseTaskTransformer):

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

        super(DynamicLabelTransformer, self).__init__(namespace,
                                                      name=name,
                                                      fill_na=0,
                                                      sr=sr,
                                                      hop_length=hop_length)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)

    def empty(self, duration):
        ann = jams.Annotation(namespace=self.namespace)
        ann.append(time=0, duration=duration, value=None)
        return ann

    def transform_annotation(self, ann, duration):

        intervals, values = ann.data.to_interval_values()

        # Suppress all intervals not in the encoder
        tags = []
        for v in values:
            if v in self._classes:
                tags.extend(self.encoder.transform([[v]]))
            else:
                tags.extend(self.encoder.transform([[]]))

        tags = np.asarray(tags)
        target = self.encode_intervals(duration, intervals, tags)

        return {'tags': target}


class StaticLabelTransformer(BaseTaskTransformer):

    def __init__(self, namespace, name, labels):
        '''Initialize a global label transformer

        Parameters
        ----------
        jam : jams.JAMS
            The JAMS object container
        '''

        super(StaticLabelTransformer, self).__init__(namespace,
                                                     name=name,
                                                     fill_na=0,
                                                     sr=1, hop_length=1)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)

    def empty(self, duration):
        ann = jams.Annotation(namespace=self.namespace)
        return ann

    def transform_annotation(self, ann, duration):

        intervals = np.asarray([[0, 1]])
        values = list(ann.data.value)
        intervals = np.tile(intervals, [len(values), 1])

        # Suppress all intervals not in the encoder
        tags = [v for v in values if v in self._classes]
        if len(tags):
            target = self.encoder.transform([tags]).max(axis=0)
        else:
            target = np.zeros(len(self._classes), dtype=np.int)

        return {'tags': target}
