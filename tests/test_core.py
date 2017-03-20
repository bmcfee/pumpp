#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Miscellaneous utility tests'''

import pytest
import numpy as np

import pumpp
import jams
import keras as K


@pytest.fixture(params=[11025, 22050])
def sr(request):
    return request.param


@pytest.fixture(params=[128, 512])
def hop_length(request):
    return request.param


@pytest.fixture(params=[None,
                        'tests/data/test.jams',
                        jams.load('tests/data/test.jams')])
def jam(request):
    return request.param


@pytest.mark.parametrize('audio_f', ['tests/data/test.ogg'])
def test_transform(audio_f, jam, sr, hop_length):

    ops = [pumpp.feature.STFT(name='stft', sr=sr,
                              hop_length=hop_length,
                              n_fft=2*hop_length),

           pumpp.task.BeatTransformer(name='beat', sr=sr,
                                      hop_length=hop_length),

           pumpp.task.ChordTransformer(name='chord', sr=sr,
                                       hop_length=hop_length),

           pumpp.task.StaticLabelTransformer(name='tags',
                                             namespace='tag_open',
                                             labels=['rock', 'jazz'])]

    data = pumpp.transform(audio_f, jam, *ops)

    # Fields we should have:
    assert set(data.keys()) == set(['stft/mag', 'stft/phase',
                                    'beat/beat', 'beat/downbeat',
                                    'beat/_valid',
                                    'beat/mask_downbeat',
                                    'chord/pitch', 'chord/root', 'chord/bass',
                                    'chord/_valid',
                                    'tags/tags', 'tags/_valid'])

    # time shapes should be the same for annotations
    assert data['beat/beat'].shape[1] == data['beat/downbeat'].shape[1]
    assert data['beat/beat'].shape[1] == data['chord/pitch'].shape[1]
    assert data['beat/beat'].shape[1] == data['chord/root'].shape[1]
    assert data['beat/beat'].shape[1] == data['chord/bass'].shape[1]

    # Audio features can be off by at most a frame
    assert (np.abs(data['stft/mag'].shape[1] - data['beat/beat'].shape[1])
            * hop_length / float(sr)) <= 0.05
    pass


@pytest.mark.parametrize('audio_f', ['tests/data/test.ogg'])
def test_pump(audio_f, jam, sr, hop_length):

    ops = [pumpp.feature.STFT(name='stft', sr=sr,
                              hop_length=hop_length,
                              n_fft=2*hop_length),

           pumpp.task.BeatTransformer(name='beat', sr=sr,
                                      hop_length=hop_length),

           pumpp.task.ChordTransformer(name='chord', sr=sr,
                                       hop_length=hop_length),

           pumpp.task.StaticLabelTransformer(name='tags',
                                             namespace='tag_open',
                                             labels=['rock', 'jazz'])]

    data1 = pumpp.transform(audio_f, jam, *ops)

    pump = pumpp.Pump(*ops)
    data2 = pump.transform(audio_f, jam)

    assert data1.keys() == data2.keys()

    for key in data1:
        assert np.allclose(data1[key], data2[key])

    fields = dict()
    for op in ops:
        fields.update(**op.fields)

    assert pump.fields == fields


@pytest.mark.parametrize('audio_f', ['tests/data/test.ogg'])
def test_pump_empty(audio_f, jam, sr, hop_length):

    pump = pumpp.Pump()
    data = pump.transform(audio_f, jam)
    assert data == dict()


def test_pump_add(sr, hop_length):

    ops = [pumpp.feature.STFT(name='stft', sr=sr,
                              hop_length=hop_length,
                              n_fft=2*hop_length),

           pumpp.task.BeatTransformer(name='beat', sr=sr,
                                      hop_length=hop_length),

           pumpp.task.ChordTransformer(name='chord', sr=sr,
                                       hop_length=hop_length),

           pumpp.task.StaticLabelTransformer(name='tags',
                                             namespace='tag_open',
                                             labels=['rock', 'jazz'])]

    pump = pumpp.Pump()
    assert pump.ops == []

    for op in ops:
        pump.add(op)
        assert op in pump.ops


@pytest.mark.xfail(raises=pumpp.ParameterError)
def test_pump_add_bad():

    pumpp.Pump('foo')


@pytest.mark.parametrize('n_samples', [None, 10])
@pytest.mark.parametrize('duration', [1, 5])
def test_pump_sampler(sr, hop_length, n_samples, duration):
    ops = [pumpp.feature.STFT(name='stft', sr=sr,
                              hop_length=hop_length,
                              n_fft=2*hop_length),

           pumpp.task.BeatTransformer(name='beat', sr=sr,
                                      hop_length=hop_length)]

    P = pumpp.Pump(*ops)

    S1 = pumpp.Sampler(n_samples, duration, *ops)
    S2 = P.sampler(n_samples, duration)

    assert S1._time == S2._time
    assert S1.n_samples == S2.n_samples
    assert S1.duration == S2.duration


@pytest.mark.skip
def test_pump_layers(sr, hop_length):
    ops = [pumpp.feature.STFT(name='stft', sr=sr,
                              hop_length=hop_length,
                              n_fft=2*hop_length),

           pumpp.feature.CQT(name='cqt', sr=sr,
                             hop_length=hop_length),

           pumpp.task.BeatTransformer(name='beat', sr=sr,
                                      hop_length=hop_length)]

    P = pumpp.Pump(*ops)

    L1 = P.layers()
    L2 = dict()
    L2.update(ops[0].layers())
    L2.update(ops[1].layers())

    assert L1.keys() == L2.keys()

    for k in L1:
        assert L1[k].dtype == L2[k].dtype
        for d1, d2 in zip(L1[k].shape, L2[k].shape):
            assert str(d1) == str(d2)
