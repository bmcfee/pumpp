#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Miscellaneous utility tests'''

import pytest
import numpy as np

import pumpp
import jams


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
