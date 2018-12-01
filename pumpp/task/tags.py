#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tag task transformers'''

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from librosa import time_to_frames
from librosa.sequence import transition_loop

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

    p_self : None, float in (0, 1), or np.ndarray [shape=(n_labels,)]
        Optional self-loop probability(ies), used for Viterbi decoding

    p_state : None or np.ndarray [shape=(n_labels,)]
        Optional marginal probability for each class

    p_init : None or np.ndarray [shape=(n_labels,)]
        Optional initial probability for each class


    See Also
    --------
    StaticLabelTransformer
    '''
    def __init__(self, name, namespace, labels=None, sr=22050, hop_length=512,
                 p_self=None, p_init=None, p_state=None):
        super(DynamicLabelTransformer, self).__init__(name=name,
                                                      namespace=namespace,
                                                      sr=sr,
                                                      hop_length=hop_length)

        if labels is None:
            labels = jams.schema.values(namespace)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)
        
        if p_self is None:
            self.transition = None
        else:
            self.transition = np.empty((len(self._classes), 2, 2))
            if np.isscalar(p_self):
                p_self = p_self * np.ones(len(self._classes))

            for i in range(len(self._classes)):
                self.transition[i] = transition_loop(2, p_self[i])

        if p_init is not None:
            if len(p_init) != len(self._classes):
                raise ParameterError('Invalid p_init.shape={} for vocabulary {} size={}'.format(p_init.shape, vocab, len(self._classes)))

        self.p_init = p_init

        if p_state is not None:
            if len(p_state) != len(self._classes):
                raise ParameterError('Invalid p_state.shape={} for vocabulary {} size={}'.format(p_state.shape, vocab, len(self._classes)))

        self.p_state = p_state
        
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
                                                       duration=duration,
                                                       transition=self.transition,
                                                       p_init=self.p_init,
                                                       p_state=self.p_state):
            # Map start:end to frames
            f_start, f_end = time_to_frames([start, end],
                                            sr=self.sr,
                                            hop_length=self.hop_length)

            confidence = np.mean(encoded[f_start:f_end+1, value])

            value_dec = self.encoder.inverse_transform(np.atleast_2d(value))[0]

            for vd in value_dec:
                ann.append(time=start,
                           duration=end-start,
                           value=vd,
                           confidence=confidence)

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
            detected = (encoded >= 0.5)
        else:
            detected = encoded

        for vd in self.encoder.inverse_transform(np.atleast_2d(detected))[0]:
            vid = np.flatnonzero(self.encoder.transform(np.atleast_2d(vd)))
            ann.append(time=0,
                       duration=duration,
                       value=vd,
                       confidence=encoded[vid])
        return ann
