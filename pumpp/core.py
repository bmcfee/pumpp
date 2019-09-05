#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Core functionality
==================
.. autosummary::
    :toctree: generated/

    Pump
'''
import os
import librosa
import jams

from .base import Slicer
from .exceptions import ParameterError
from .task import BaseTaskTransformer
from .feature import FeatureExtractor
from .sampler import Sampler
from . import util


class Pump(Slicer):
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
    >>> data = pump.transform(audio_f='/my/audio/file.mp3',
    ...                       jam='/my/jams/annotation.jams')

    Or use the call interface:

    >>> data = pump(audio_f='/my/audio/file.mp3',
    ...             jam='/my/jams/annotation.jams')

    Or apply to audio in memory, and without existing annotations:

    >>> y, sr = librosa.load('/my/audio/file.mp3')
    >>> data = pump(y=y, sr=sr)

    Access all the fields produced by this pump:

    >>> pump.fields
    {'chord/chord': Tensor(shape=(None, 170), dtype=<class 'bool'>),
     'cqt/mag': Tensor(shape=(None, 288), dtype=<class 'numpy.float32'>),
     'cqt/phase': Tensor(shape=(None, 288), dtype=<class 'numpy.float32'>)}

    Access a constituent operator by name:

    >>> pump['chord'].fields
    {'chord/chord': Tensor(shape=(None, 170), dtype=<class 'bool'>)}
    '''

    def __init__(self, *ops, cache_dir=None):

        self.ops = []
        self.opmap = dict()
        super(Pump, self).__init__(*ops)
        self.cache_dir = cache_dir

    def add(self, operator):
        '''Add an operation to this pump.

        Parameters
        ----------
        operator : BaseTaskTransformer, FeatureExtractor
            The operation to add

        Raises
        ------
        ParameterError
            if `op` is not of a correct type
        '''
        if not isinstance(operator, (BaseTaskTransformer, FeatureExtractor)):
            raise ParameterError('operator={} must be one of '
                                 '(BaseTaskTransformer, FeatureExtractor)'
                                 .format(operator))

        if operator.name in self.opmap:
            raise ParameterError('Duplicate operator name detected: '
                                 '{}'.format(operator))

        super(Pump, self).add(operator)
        self.opmap[operator.name] = operator
        self.ops.append(operator)

    def transform(self, audio_f=None, jam=None, y=None, sr=None, crop=False,
                  data=None, refresh=False):
        '''Apply the transformations to an audio file, and optionally JAMS object.

        Parameters
        ----------
        audio_f : str
            Path to audio file

        jam : optional, `jams.JAMS`, str or file-like
            Optional JAMS object/path to JAMS file/open file descriptor.

            If provided, this will provide data for task transformers.

        y : np.ndarray
        sr : number > 0
            If provided, operate directly on an existing audio buffer `y` at
            sampling rate `sr` rather than load from `audio_f`.

        crop : bool
            If `True`, then data are cropped to a common time index across all
            fields.  Otherwise, data may have different time extents.

        data : optional, dict
            Optional data dict containing already computed features. Fields in
            dict will be skipped unless ``refresh`` is True.

        refresh : bool
            Recompute features.

        Returns
        -------
        data : dict
            Data dictionary containing the transformed audio (and annotations)

        Raises
        ------
        ParameterError
            At least one of `audio_f` or `(y, sr)` must be provided.

        '''
        data = dict() if data is None else data

        # initialize data, load from cache if dir provided.
        if self.cache_dir and audio_f:
            cache_id = util.get_cache_id(audio_f)
            cache_file = os.path.join(self.cache_dir, cache_id + '.h5')

            if not refresh and os.path.isfile(cache_file):
                data = util.load_h5(cache_file, data=data, fields=self.fields)

        # check if all fields exist - potentially avoid loading audio
        # TODO: check BaseTaskTransformer and FeatureExtractor separately
        initial_data_keys = set(data)
        if set(self.fields).issubset(initial_data_keys):
            return data

        if y is None:
            if audio_f is None:
                raise ParameterError('At least one of `y` or `audio_f` '
                                     'must be provided')

            # Load the audio
            y, sr = librosa.load(audio_f, sr=sr, mono=True)

        if sr is None:
            raise ParameterError('If audio is provided as `y`, you must '
                                 'specify the sampling rate as sr=')

        if jam is None:
            jam = jams.JAMS()
            jam.file_metadata.duration = librosa.get_duration(y=y, sr=sr)

        # Load the jams
        if not isinstance(jam, jams.JAMS):
            jam = jams.load(jam)

        for operator in self.ops:
            # skip keys that already exist
            if set(operator.fields).issubset(initial_data_keys):
                continue

            if isinstance(operator, BaseTaskTransformer):
                data.update(operator.transform(jam))
            elif isinstance(operator, FeatureExtractor):
                data.update(operator.transform(y, sr))
        if crop:
            data = self.crop(data)

        # save for future use
        if self.cache_dir and audio_f:
            if refresh or initial_data_keys != set(data):
                util.save_h5(cache_file, **{k: data[k] for k in self.fields})
        return data

    def sampler(self, n_samples, duration, random_state=None):
        '''Construct a sampler object for this pump's operators.

        Parameters
        ----------
        n_samples : None or int > 0
            The number of samples to generate

        duration : int > 0
            The duration (in frames) of each sample patch

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.

        Returns
        -------
        sampler : pumpp.Sampler
            The sampler object

        See Also
        --------
        pumpp.sampler.Sampler
        '''

        return Sampler(n_samples, duration,
                       random_state=random_state,
                       *self.ops)

    @property
    def fields(self):
        '''A dictionary of fields constructed by this pump'''
        out = dict()
        for operator in self.ops:
            out.update(**operator.fields)

        return out

    def layers(self, api='keras'):
        '''Construct input layers for all feature transformers
        in the pump.

        Parameters
        ----------
        api : {'keras', ...}
            Which API to use for layer construction

        Returns
        -------
        layers : {field: keras.layers.Input}
            A dictionary of keras input layers, keyed by the corresponding
            fields.
        '''

        layermap = dict()
        for operator in self.ops:
            if hasattr(operator, 'layers'):
                layermap.update(operator.layers(api=api))
        return layermap

    def __getitem__(self, key):
        return self.opmap.get(key)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def __str__(self):
        rstr = '<Pump [{:d} operators, {:d} fields]>'.format(len(self.ops),
                                                             len(self.fields))
        for key in self.opmap:
            rstr += "\n  - '{}': {}".format(key, type(self.opmap[key]))
            for field in self.opmap[key].fields:
                rstr += "\n    - '{}': {}".format(field, self.opmap[key].fields[field])
        return rstr

    def _repr_html_(self):

        rstr = '<dl class="row">'
        for key in self.opmap:
            rstr += '\n  <dt class="col-sm-3">{:s}</dt>'.format(key)
            rstr += '\n  <dd class="col-sm-9">{}'.format(self.opmap[key])

            rstr += '<ul>'
            for fkey, field in self.opmap[key].fields.items():
                rstr += '\n  <li>{:s} [shape={}, dtype={}]</li>'.format(fkey,
                                                                        field.shape,
                                                                        repr(field.dtype))
            rstr += '</ul></dd>'
        rstr += '</dl>'
        return rstr
