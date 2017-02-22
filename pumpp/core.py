#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Core functionality
==================
.. autosummary::
    :toctree: generated/

    transform
'''

import librosa
import jams

from .task import BaseTaskTransformer
from .feature import FeatureExtractor


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
