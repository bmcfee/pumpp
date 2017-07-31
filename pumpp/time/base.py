#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Feature extraction base class'''

import numpy as np

from librosa import resample

from ..base import Scope


class TimeSync(Scope):
    '''The base time-sync class.

    Attributes
    ----------
    name : str
        The name for this time synchronizer
    '''
    def __init__(self, name):

        super(TimeSync, self).__init__(name)
        self.register('_intervals', [None, 2], np.float32)

    def transform(self, y, sr, data, ops):
        '''Transform an audio signal

        Parameters
        ----------
        y : np.ndarray
            The audio signal

        sr : number > 0
            The native sampling rate of y

        Returns
        -------
        dict
            Data dictionary containing features extracted from y

        See Also
        --------
        transform_audio
        '''
        if sr != self.sr:
            y = resample(y, sr, self.sr)

        intervals = self.transform_audio(y)
        ival_time = intervals['_intervals']
        data_out = {self.scope(k): intervals[k] for k in intervals}

        # Iterate over ops
        #   resample corresponding fields
        #   merge results within this scope
        for operator in ops:
            data_s = operator.sync(data, ival_time)
            data_out.update({self.scope(k): data_s[k] for k in data_s})

        return data_out

    def transform_audio(self, y):
        raise NotImplementedError
