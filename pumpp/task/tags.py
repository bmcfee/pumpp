#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tag task transformers'''

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

import jams

from .base import BaseTaskTransformer

__all__ = ['DynamicLabelTransformer', 'StaticLabelTransformer']


class DynamicLabelTransformer(BaseTaskTransformer):
    '''Time-series label transformer.

    Attributes
    ----------
    name : str
        The name of this transformer object

    namespace : str
        The JAMS namespace for this task

    labels : list of str [optional]
        The list of labels for this task.

        If not provided, it will attempt to infer the label set from the
        namespace definition.

    sr : number > 0
        The audio sampling rate

    hop_length : int > 0
        The hop length for annotation frames

    See Also
    --------
    StaticLabelTransformer
    '''
    def __init__(self, name, namespace, labels=None, sr=22050, hop_length=512):
        super(DynamicLabelTransformer, self).__init__(name=name,
                                                      namespace=namespace,
                                                      sr=sr,
                                                      hop_length=hop_length)

        if labels is None:
            labels = jams.schema.values(namespace)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)

        self.register('tags', [None, len(self._classes)], np.bool)

    def empty(self, duration):
        '''Empty label annotations.

        Constructs a single observation with an empty value (None).

        Parameters
        ----------
        duration : number > 0
            The duration of the annotation
        '''
        ann = super(DynamicLabelTransformer, self).empty(duration)
        ann.append(time=0, duration=duration, value=None)
        return ann

    def transform_annotation(self, ann, duration):
        '''Transform an annotation to dynamic label encoding.

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['tags'] : np.ndarray, shape=(n, n_labels)
                A time-varying binary encoding of the labels
        '''
        intervals, values = ann.to_interval_values()

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

    def inverse(self, encoded, duration=None):
        '''Inverse transformation'''

        ann = jams.Annotation(namespace=self.namespace, duration=duration)
        for start, end, value in self.decode_intervals(encoded,
                                                       duration=duration):
            value_dec = self.encoder.inverse_transform(np.atleast_2d(value))[0]

            for vd in value_dec:
                ann.append(time=start, duration=end-start, value=vd)

        return ann


class StaticLabelTransformer(BaseTaskTransformer):
    '''Static label transformer.

    Attributes
    ----------
    name : str
        The name of this transformer object

    namespace : str
        The JAMS namespace for this task

    labels : list of str [optional]
        The list of labels for this task.

        If not provided, it will attempt to infer the label set from the
        namespace definition.

    See Also
    --------
    DynamicLabelTransformer
    '''

    def __init__(self, name, namespace, labels=None):
        super(StaticLabelTransformer, self).__init__(name=name,
                                                     namespace=namespace,
                                                     sr=1, hop_length=1)

        if labels is None:
            labels = jams.schema.values(namespace)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)
        self.register('tags', [len(self._classes)], np.bool)

    def transform_annotation(self, ann, duration):
        '''Transform an annotation to static label encoding.

        Parameters
        ----------
        ann : jams.Annotation
            The annotation to convert

        duration : number > 0
            The duration of the track

        Returns
        -------
        data : dict
            data['tags'] : np.ndarray, shape=(n_labels,)
                A static binary encoding of the labels
        '''
        intervals = np.asarray([[0, 1]])
        values = list([obs.value for obs in ann])
        intervals = np.tile(intervals, [len(values), 1])

        # Suppress all intervals not in the encoder
        tags = [v for v in values if v in self._classes]
        if len(tags):
            target = self.encoder.transform([tags]).astype(np.bool).max(axis=0)
        else:
            target = np.zeros(len(self._classes), dtype=np.bool)

        return {'tags': target}

    def inverse(self, encoded, duration=None):
        '''Inverse static tag transformation'''

        ann = jams.Annotation(namespace=self.namespace, duration=duration)

        if np.isrealobj(encoded):
            encoded = (encoded >= 0.5)

        for vd in self.encoder.inverse_transform(np.atleast_2d(encoded))[0]:
            ann.append(time=0, duration=duration, value=vd)

        return ann
