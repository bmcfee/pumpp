#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for task decoding'''

import numpy as np
import pytest
import jams

import pumpp

# Sampling rate and hop are simple here to keep things
# divisible for inverse checks


@pytest.fixture()
def sr():
    return 10


@pytest.fixture()
def hop_length():
    return 1


@pytest.fixture()
def tag_gtzan():

    ann = jams.Annotation(namespace='tag_gtzan', duration=10)

    ann.append(time=0, duration=5, value='blues')
    ann.append(time=1.5, duration=1.5, value='reggae')

    return ann


def test_decode_tags_dynamic(sr, hop_length, tag_gtzan):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tc = pumpp.task.DynamicLabelTransformer('genre', 'tag_gtzan',
                                            hop_length=hop_length,
                                            sr=sr)

    data = tc.transform_annotation(tag_gtzan, tag_gtzan.duration)

    inverse = tc.inverse(data['tags'], duration=tag_gtzan.duration)
    data2 = tc.transform_annotation(inverse, tag_gtzan.duration)

    assert np.allclose(data['tags'], data2['tags'])


def test_decode_tags_static(tag_gtzan):

    tc = pumpp.task.StaticLabelTransformer('genre', 'tag_gtzan')

    data = tc.transform_annotation(tag_gtzan, tag_gtzan.duration)
    inverse = tc.inverse(data['tags'], tag_gtzan.duration)
    data2 = tc.transform_annotation(inverse, tag_gtzan.duration)

    assert np.allclose(data['tags'], data2['tags'])
