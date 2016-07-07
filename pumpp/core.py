#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Core functionality'''

import librosa
import jams

from .task import BaseTaskTransformer
from .feature import FeatureExtractor


def transform(audio_f, jams_f, *ops):
    '''Apply a set of operations to a track

    Parameters
    ----------
    audio_f : str
        The path to the audio file

    jams_f : str
        The path to the jams file

    ops : list of pumpp.task.BaseTaskTransform or pumpp.feature.FeatureExtractor
    '''

    # Load the audio
    y, sr = librosa.load(audio_f, sr=None, mono=True)

    # Load the jams
    jam = jams.load(jams_f)

    data = dict()

    for op in ops:
        if isinstance(op, BaseTaskTransformer):
            data.update(op.transform(jam))
        elif isinstance(op, FeatureExtractor):
            data.update(op.transform(y, sr))
    return data
