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
def ann_tag():

    ann = jams.Annotation(namespace='tag_gtzan', duration=10)

    ann.append(time=0, duration=5, value='blues')
    ann.append(time=1.5, duration=1.5, value='reggae')

    return ann


@pytest.fixture()
def ann_beat():
    ann = jams.Annotation(namespace='beat', duration=10)

    # for n, i in enumerate(np.arange(0, 10, 0.5)):
    #    ann.append(time=i, duration=0, value=1 + (n % 4))

    # Make up two measures of 4/4, plus two pickup beats
    for t, v in [(0, -2), (0.5, -1),
                 (1, 1), (1.5, 2), (2, 3), (3, 4),
                 (3.5, 1), (4, 2), (4.5, 3), (5, 4),
                 (5.5, 1), (6, 2), (6.5, 3), (7, 4)]:

        ann.append(time=t, duration=0, value=v)

    return ann


def test_decode_tags_dynamic(sr, hop_length, ann_tag):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tc = pumpp.task.DynamicLabelTransformer('genre', 'tag_gtzan',
                                            hop_length=hop_length,
                                            sr=sr)

    data = tc.transform_annotation(ann_tag, ann_tag.duration)

    inverse = tc.inverse(data['tags'], duration=ann_tag.duration)
    data2 = tc.transform_annotation(inverse, ann_tag.duration)

    assert np.allclose(data['tags'], data2['tags'])


def test_decode_tags_static(ann_tag):

    tc = pumpp.task.StaticLabelTransformer('genre', 'tag_gtzan')

    data = tc.transform_annotation(ann_tag, ann_tag.duration)
    inverse = tc.inverse(data['tags'], ann_tag.duration)
    data2 = tc.transform_annotation(inverse, ann_tag.duration)

    assert np.allclose(data['tags'], data2['tags'])


def test_decode_beat(sr, hop_length, ann_beat):

    tc = pumpp.task.BeatTransformer('beat', sr=sr, hop_length=hop_length)

    data = tc.transform_annotation(ann_beat, ann_beat.duration)
    inverse = tc.inverse(data['beat'], duration=ann_beat.duration)
    data2 = tc.transform_annotation(inverse, ann_beat.duration)

    assert np.allclose(data['beat'], data2['beat'])


def test_decode_beat_downbeat(sr, hop_length, ann_beat):

    tc = pumpp.task.BeatTransformer('beat', sr=sr, hop_length=hop_length)

    data = tc.transform_annotation(ann_beat, ann_beat.duration)
    inverse = tc.inverse(data['beat'], downbeat=data['downbeat'],
                         duration=ann_beat.duration)
    data2 = tc.transform_annotation(inverse, ann_beat.duration)

    assert np.allclose(data['beat'], data2['beat'])
