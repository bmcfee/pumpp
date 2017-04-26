#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Testing the sampler module'''

import numpy as np

import pytest

import pumpp


# Make a fixture with some audio and task output
@pytest.fixture(params=[11025], scope='module')
def sr(request):
    return request.param


@pytest.fixture(params=[512], scope='module')
def hop_length(request):
    return request.param


@pytest.fixture(scope='module')
def ops(sr, hop_length):

    ops = []

    # Let's put on two feature extractors
    ops.append(pumpp.feature.STFT(name='stft', sr=sr,
                                  hop_length=hop_length,
                                  n_fft=hop_length))

    ops.append(pumpp.feature.Tempogram(name='rhythm', sr=sr,
                                       hop_length=hop_length,
                                       win_length=hop_length))

    # A time-varying annotation
    ops.append(pumpp.task.ChordTransformer(name='chord', sr=sr,
                                           hop_length=hop_length))

    # And a static annotation
    ops.append(pumpp.task.VectorTransformer(namespace='vector',
                                            dimension=32,
                                            name='vec'))

    yield ops


@pytest.fixture(scope='module')
def data(ops):

    audio_f = 'tests/data/test.ogg'
    jams_f = 'tests/data/test.jams'

    P = pumpp.Pump(*ops)
    return P.transform(audio_f=audio_f, jam=jams_f)


@pytest.fixture(params=[4, 16, None], scope='module')
def n_samples(request):
    return request.param


@pytest.fixture(params=[16, 32], scope='module')
def duration(request):
    return request.param


@pytest.fixture(params=[None, 16, 256,
                        pytest.mark.xfail(-1, raises=pumpp.ParameterError)],
                scope='module')
def stride(request):
    return request.param


@pytest.fixture(params=[None, 20170401, np.random.RandomState(100),
                        pytest.mark.xfail('bad rng',
                                          raises=pumpp.ParameterError)],
                scope='module')
def rng(request):
    return request.param


def test_sampler(data, ops, n_samples, duration, rng):

    MAX_SAMPLES = 30
    sampler = pumpp.Sampler(n_samples, duration, *ops, random_state=rng)

    # Build the set of reference keys that we want to track
    ref_keys = set()
    for op in ops:
        ref_keys |= set(op.fields.keys())

    for datum, n in zip(sampler(data), range(MAX_SAMPLES)):
        # First, test that we have the right fields
        assert set(datum.keys()) == ref_keys

        # Now test that shape is preserved in the right way
        for key in datum:
            ref_shape = list(data[key].shape)
            if sampler._time.get(key, None) is not None:
                ref_shape[sampler._time[key]] = duration

            # Check that all keys have length=1
            assert datum[key].shape[0] == 1
            assert list(datum[key].shape[1:]) == ref_shape[1:]

    # Test that we got the right number of samples out
    if n_samples is None:
        assert n == MAX_SAMPLES - 1
    else:
        assert n == n_samples - 1


def test_sequential_sampler(data, ops, duration, stride, rng):
    sampler = pumpp.SequentialSampler(duration, *ops, stride=stride, random_state=rng)

    # Build the set of reference keys that we want to track
    ref_keys = set()
    for op in ops:
        ref_keys |= set(op.fields.keys())

    for datum in sampler(data):
        # First, test that we have the right fields
        assert set(datum.keys()) == ref_keys

        # Now test that shape is preserved in the right way
        for key in datum:
            ref_shape = list(data[key].shape)
            if sampler._time.get(key, None) is not None:
                ref_shape[sampler._time[key]] = duration

            # Check that all keys have length=1
            assert datum[key].shape[0] == 1
            assert list(datum[key].shape[1:]) == ref_shape[1:]
