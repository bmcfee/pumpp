#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Testing the sampler module'''

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

    ops = [pumpp.feature.STFT(name='stft', sr=sr,
                              hop_length=hop_length,
                              n_fft=hop_length)]

    ops.append(pumpp.feature.Tempogram(name='rhythm', sr=sr,
                                       hop_length=hop_length,
                                       win_length=hop_length))

    ops.append(pumpp.task.BeatTransformer(name='beat', sr=sr,
                                          hop_length=hop_length))

    yield ops


@pytest.fixture(scope='module')
def data(ops):

    audio_f = 'tests/data/test.ogg'
    jams_f = 'tests/data/test.jams'

    return pumpp.transform(audio_f, jams_f, *ops)


@pytest.fixture(params=[4, 16, None], scope='module')
def n_samples(request):
    return request.param


@pytest.fixture(params=[16, 32], scope='module')
def duration(request):
    return request.param


def test_sampler(data, ops, n_samples, duration):

    MAX_SAMPLES = 30
    sampler = pumpp.Sampler(n_samples, duration, *ops)

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
            if sampler._time[key] is not None:
                ref_shape[sampler._time[key]] = duration

            assert list(datum[key].shape) == ref_shape

    # Test that we got the right number of samples out
    if n_samples is None:
        assert n == MAX_SAMPLES - 1
    else:
        assert n == n_samples - 1
