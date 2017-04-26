#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Miscellaneous utility tests'''

import pytest
import numpy as np

import librosa
import jams

import pumpp


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


@pytest.mark.parametrize('audio_f', [None, 'tests/data/test.ogg'])
@pytest.mark.parametrize('y', [None, 'tests/data/test.ogg'])
@pytest.mark.parametrize('sr2', [None, 22050, 44100])
def test_pump(audio_f, jam, y, sr, sr2, hop_length):

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

    P = pumpp.Pump(*ops)

    if audio_f is None and y is None:
        # no input
        with pytest.raises(pumpp.ParameterError):
            data = P.transform(audio_f=audio_f, jam=jam, y=y, sr=sr2)
    elif y is not None and sr2 is None:
        # input buffer, but no sampling rate
        y = librosa.load(y, sr=sr2)[0]
        with pytest.raises(pumpp.ParameterError):
            data = P.transform(audio_f=audio_f, jam=jam, y=y, sr=sr2)
    elif y is not None:
        y = librosa.load(y, sr=sr2)[0]
        data = P.transform(audio_f=audio_f, jam=jam, y=y, sr=sr2)
    else:

        fields = set(['stft/mag',
                      'stft/phase',
                      'beat/beat',
                      'beat/downbeat',
                      'beat/mask_downbeat',
                      'chord/pitch',
                      'chord/root',
                      'chord/bass',
                      'tags/tags'])

        valids = set(['beat/_valid', 'chord/_valid', 'tags/_valid'])

        assert set(P.fields.keys()) == fields

        data = P.transform(audio_f=audio_f, jam=jam, y=y, sr=sr2)
        data2 = P(audio_f=audio_f, jam=jam, y=y, sr=sr2)

        # Fields we should have:
        assert set(data.keys()) == fields | valids

        # time shapes should be the same for annotations
        assert data['beat/beat'].shape[1] == data['beat/downbeat'].shape[1]
        assert data['beat/beat'].shape[1] == data['chord/pitch'].shape[1]
        assert data['beat/beat'].shape[1] == data['chord/root'].shape[1]
        assert data['beat/beat'].shape[1] == data['chord/bass'].shape[1]

        # Audio features can be off by at most a frame
        assert (np.abs(data['stft/mag'].shape[1] - data['beat/beat'].shape[1])
                * hop_length / float(sr)) <= 0.05

        assert data.keys() == data2.keys()
        for k in data:
            assert np.allclose(data[k], data2[k])


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


@pytest.mark.xfail(raises=pumpp.ParameterError)
def test_pump_add_twice(sr, hop_length):

    op = pumpp.feature.STFT(name='stft', sr=sr,
                            hop_length=hop_length,
                            n_fft=2*hop_length)

    P = pumpp.Pump()

    P.add(op)
    P.add(op)


@pytest.mark.xfail(raises=KeyError)
def test_pump_badkey(sr, hop_length):

    op = pumpp.feature.STFT(name='stft', sr=sr,
                            hop_length=hop_length,
                            n_fft=2*hop_length)

    P = pumpp.Pump(op)

    P['bad key']


@pytest.mark.parametrize('n_samples', [None, 10])
@pytest.mark.parametrize('duration', [1, 5])
@pytest.mark.parametrize('rng', [None, 1])
def test_pump_sampler(sr, hop_length, n_samples, duration, rng):
    ops = [pumpp.feature.STFT(name='stft', sr=sr,
                              hop_length=hop_length,
                              n_fft=2*hop_length),

           pumpp.task.BeatTransformer(name='beat', sr=sr,
                                      hop_length=hop_length)]

    P = pumpp.Pump(*ops)

    S1 = pumpp.Sampler(n_samples, duration, random_state=rng, *ops)
    S2 = P.sampler(n_samples, duration, random_state=rng)

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
