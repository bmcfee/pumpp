#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import numpy as np
import pytest
import librosa

import pumpp

from test_task import shape_match, type_match


@pytest.fixture(params=[None, 22050, 16000])
def audio(request):
    y, sr_out = librosa.load(librosa.util.example_audio_file(),
                             sr=request.param,
                             duration=5)
    return {'y': y, 'sr': sr_out}


@pytest.fixture(params=[512, 1024])
def n_fft(request):
    return request.param

@pytest.fixture(params=[32, 128])
def n_mels(request):
    return request.param


@pytest.fixture()
def SR():
    return 22050


@pytest.fixture()
def HOP_LENGTH():
    return 512


@pytest.fixture(params=[1, 3])
def over_sample(request):
    return request.param


@pytest.fixture(params=[1, 4])
def n_octaves(request):
    return request.param



# STFT features

def test_feature_stft_fields(SR, HOP_LENGTH, n_fft):

    ext = pumpp.feature.STFT(name='stft',
                             sr=SR, hop_length=HOP_LENGTH,
                             n_fft=n_fft)

    # Check the fields
    assert set(ext.fields.keys()) == set(['stft/mag', 'stft/phase'])

    assert ext.fields['stft/mag'].shape == (None, 1 + n_fft // 2)
    assert ext.fields['stft/phase'].shape == (None, 1 + n_fft // 2)
    assert ext.fields['stft/mag'].dtype is np.float32
    assert ext.fields['stft/phase'].dtype is np.float32


def test_feature_stft_mag_fields(SR, HOP_LENGTH, n_fft):

    ext = pumpp.feature.STFTMag(name='stft',
                                sr=SR, hop_length=HOP_LENGTH,
                                n_fft=n_fft)

    # Check the fields
    assert set(ext.fields.keys()) == set(['stft/mag'])

    assert ext.fields['stft/mag'].shape == (None, 1 + n_fft // 2)
    assert ext.fields['stft/mag'].dtype is np.float32


def test_feature_stft_phasediff_fields(SR, HOP_LENGTH, n_fft):

    ext = pumpp.feature.STFTPhaseDiff(name='stft',
                                      sr=SR, hop_length=HOP_LENGTH,
                                      n_fft=n_fft)

    # Check the fields
    assert set(ext.fields.keys()) == set(['stft/mag', 'stft/dphase'])

    assert ext.fields['stft/mag'].shape == (None, 1 + n_fft // 2)
    assert ext.fields['stft/dphase'].shape == (None, 1 + n_fft // 2)
    assert ext.fields['stft/mag'].dtype is np.float32
    assert ext.fields['stft/dphase'].dtype is np.float32


def test_feature_stft(audio, SR, HOP_LENGTH, n_fft):

    ext = pumpp.feature.STFT(name='stft',
                             sr=SR, hop_length=HOP_LENGTH,
                             n_fft=n_fft)

    output = ext.transform(**audio)

    assert set(output.keys()) == set(ext.fields.keys())

    for key in ext.fields:
        assert shape_match(output[key].shape[1:], ext.fields[key].shape)
        assert type_match(output[key].dtype, ext.fields[key].dtype)


def test_feature_stft_phasediff(audio, SR, HOP_LENGTH, n_fft):

    ext = pumpp.feature.STFTPhaseDiff(name='stft',
                                      sr=SR, hop_length=HOP_LENGTH,
                                      n_fft=n_fft)

    output = ext.transform(**audio)

    # Check the fields
    assert set(output.keys()) == set(ext.fields.keys())

    for key in ext.fields:
        assert shape_match(output[key].shape[1:], ext.fields[key].shape)
        assert type_match(output[key].dtype, ext.fields[key].dtype)


def test_feature_stft_mag(audio, SR, HOP_LENGTH, n_fft):

    ext = pumpp.feature.STFTMag(name='stft',
                                sr=SR, hop_length=HOP_LENGTH,
                                n_fft=n_fft)

    output = ext.transform(**audio)

    # Check the fields
    assert set(output.keys()) == set(ext.fields.keys())

    for key in ext.fields:
        assert shape_match(output[key].shape[1:], ext.fields[key].shape)
        assert type_match(output[key].dtype, ext.fields[key].dtype)


# Mel features
def test_feature_mel_fields(SR, HOP_LENGTH, n_fft, n_mels):

    ext = pumpp.feature.Mel(name='mel',
                            sr=SR, hop_length=HOP_LENGTH,
                            n_fft=n_fft, n_mels=n_mels)

    # Check the fields
    assert set(ext.fields.keys()) == set(['mel/mag'])

    assert ext.fields['mel/mag'].shape == (None, n_mels)
    assert ext.fields['mel/mag'].dtype is np.float32


def test_feature_mel(audio, SR, HOP_LENGTH, n_fft, n_mels):

    ext = pumpp.feature.Mel(name='mel',
                            sr=SR, hop_length=HOP_LENGTH,
                            n_fft=n_fft, n_mels=n_mels)

    output = ext.transform(**audio)

    # Check the fields
    assert set(output.keys()) == set(ext.fields.keys())

    for key in ext.fields:
        assert shape_match(output[key].shape[1:], ext.fields[key].shape)
        assert type_match(output[key].dtype, ext.fields[key].dtype)


# CQT features

def test_feature_cqt_fields(SR, HOP_LENGTH, over_sample, n_octaves):

    ext = pumpp.feature.CQT(name='cqt',
                            sr=SR, hop_length=HOP_LENGTH,
                            n_octaves=n_octaves,
                            over_sample=over_sample)

    # Check the fields
    assert set(ext.fields.keys()) == set(['cqt/mag', 'cqt/phase'])

    assert ext.fields['cqt/mag'].shape == (None, over_sample * n_octaves * 12)
    assert ext.fields['cqt/phase'].shape == (None, over_sample * n_octaves * 12)
    assert ext.fields['cqt/mag'].dtype is np.float32
    assert ext.fields['cqt/phase'].dtype is np.float32

def test_feature_cqtmag_fields(SR, HOP_LENGTH, over_sample, n_octaves):

    ext = pumpp.feature.CQTMag(name='cqt',
                            sr=SR, hop_length=HOP_LENGTH,
                            n_octaves=n_octaves,
                            over_sample=over_sample)

    # Check the fields
    assert set(ext.fields.keys()) == set(['cqt/mag'])

    assert ext.fields['cqt/mag'].shape == (None, over_sample * n_octaves * 12)
    assert ext.fields['cqt/mag'].dtype is np.float32

def test_feature_cqtphasediff_fields(SR, HOP_LENGTH, over_sample, n_octaves):

    ext = pumpp.feature.CQTPhaseDiff(name='cqt',
                                     sr=SR, hop_length=HOP_LENGTH,
                                     n_octaves=n_octaves,
                                     over_sample=over_sample)

    # Check the fields
    assert set(ext.fields.keys()) == set(['cqt/mag', 'cqt/dphase'])

    assert ext.fields['cqt/mag'].shape == (None, over_sample * n_octaves * 12)
    assert ext.fields['cqt/dphase'].shape == (None, over_sample * n_octaves * 12)
    assert ext.fields['cqt/mag'].dtype is np.float32
    assert ext.fields['cqt/dphase'].dtype is np.float32

def test_feature_cqt(audio, SR, HOP_LENGTH, over_sample, n_octaves):

    ext = pumpp.feature.CQT(name='cqt',
                            sr=SR, hop_length=HOP_LENGTH,
                            n_octaves=n_octaves,
                            over_sample=over_sample)

    output = ext.transform(**audio)

    assert set(output.keys()) == set(ext.fields.keys())

    for key in ext.fields:
        assert shape_match(output[key].shape[1:], ext.fields[key].shape)
        assert type_match(output[key].dtype, ext.fields[key].dtype)

def test_feature_cqtmag(audio, SR, HOP_LENGTH, over_sample, n_octaves):

    ext = pumpp.feature.CQTMag(name='cqt',
                            sr=SR, hop_length=HOP_LENGTH,
                            n_octaves=n_octaves,
                            over_sample=over_sample)

    output = ext.transform(**audio)

    assert set(output.keys()) == set(ext.fields.keys())

    for key in ext.fields:
        assert shape_match(output[key].shape[1:], ext.fields[key].shape)
        assert type_match(output[key].dtype, ext.fields[key].dtype)

def test_feature_cqtphasediff(audio, SR, HOP_LENGTH, over_sample, n_octaves):

    ext = pumpp.feature.CQTPhaseDiff(name='cqt',
                                     sr=SR, hop_length=HOP_LENGTH,
                                     n_octaves=n_octaves,
                                     over_sample=over_sample)

    output = ext.transform(**audio)

    assert set(output.keys()) == set(ext.fields.keys())

    for key in ext.fields:
        assert shape_match(output[key].shape[1:], ext.fields[key].shape)
        assert type_match(output[key].dtype, ext.fields[key].dtype)

