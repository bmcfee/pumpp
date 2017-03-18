#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Core functionality
==================
.. autosummary::
    :toctree: generated/

    Pump
    transform
'''

import librosa
import jams

from .exceptions import ParameterError
from .task import BaseTaskTransformer
from .feature import FeatureExtractor
from .sampler import Sampler


def transform(audio_f, jam, *ops):
    '''Apply a set of operations to a track

    Parameters
    ----------
    audio_f : str
        The path to the audio file

    jam : str, jams.JAMS, or file-like
        A JAMS object, or path to a JAMS file.

        If not provided, an empty jams object will be created.

    ops : list of task.BaseTaskTransform or feature.FeatureExtractor
        The operators to apply to the input data

    Returns
    -------
    data : dict
        Extracted features and annotation encodings
    '''

    # Load the audio
    y, sr = librosa.load(audio_f, sr=None, mono=True)

    if jam is None:
        jam = jams.JAMS()
        jam.file_metadata.duration = librosa.get_duration(y=y, sr=sr)

    # Load the jams
    if not isinstance(jam, jams.JAMS):
        jam = jams.load(jam)

    data = dict()

    for op in ops:
        if isinstance(op, BaseTaskTransformer):
            data.update(op.transform(jam))
        elif isinstance(op, FeatureExtractor):
            data.update(op.transform(y, sr))
    return data


class Pump(object):
    '''Top-level pump object.

    This class is used to collect feature and task transformers

    Attributes
    ----------
    ops : list of (BaseTaskTransformer, FeatureExtractor)
        The operations to apply

    Examples
    --------
    Create a CQT and chord transformer

    >>> p_cqt = pumpp.feature.CQT('cqt', sr=44100, hop_length=1024)
    >>> p_chord = pumpp.task.ChordTagTransformer(sr=44100, hop_length=1024)
    >>> pump = pumpp.Pump(p_cqt, p_chord)
    >>> data = pump.transform('/my/audio/file.mp3', '/my/jams/annotation.jams')


    See Also
    --------
    transform
    '''

    def __init__(self, *ops):

        self.ops = []
        for op in ops:
            self.add(op)

    def add(self, op):
        '''Add an operation to this pump.

        Parameters
        ----------
        op : BaseTaskTransformer, FeatureExtractor
            The operation to add

        Raises
        ------
        ParameterError
            if `op` is not of a correct type
        '''
        if not isinstance(op, (BaseTaskTransformer, FeatureExtractor)):
            raise ParameterError('op={} must be one of '
                                 '(BaseTaskTransformer, FeatureExtractor)'
                                 .format(op))

        self.ops.append(op)

    def transform(self, audio_f, jam=None):
        '''Apply the transformations to an audio file, and optionally JAMS object.

        Parameters
        ----------
        audio_f : str
            Path to audio file

        jam : optional, `jams.JAMS`, str or file-like
            Optional JAMS object/path to JAMS file/open file descriptor.

            If provided, this will provide data for task transformers.

        Returns
        -------
        data : dict
            Data dictionary containing the transformed audio (and annotations)
        '''

        return transform(audio_f, jam, *self.ops)

    def sampler(self, n_samples, duration):
        '''Construct a sampler object for this pump's operators.

        Parameters
        ----------
        n_samples : None or int > 0
            The number of samples to generate

        duration : int > 0
            The duration (in frames) of each sample patch

        Returns
        -------
        sampler : pumpp.Sampler
            The sampler object

        See Also
        --------
        pumpp.sampler.Sampler
        '''

        return Sampler(n_samples, duration, *self.ops)

    @property
    def fields(self):
        out = dict()
        for op in self.ops:
            out.update(**op.fields)

        return out
