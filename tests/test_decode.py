#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for task decoding'''

import numpy as np
import pytest
import jams

import pumpp


@pytest.fixture()
def sr():
    return 10


@pytest.fixture()
def hop_length():
    return 1


@pytest.fixture()
def tag_dynamic():

    ann = jams.Annotation(namespace='tag_gtzan', duration=10)

    ann.append(time=0, duration=5, value='blues')
    ann.append(time=1.5, duration=1.5, value='reggae')

    return ann


def test_decode_tags_dynamic(sr, hop_length, tag_dynamic):

    tc = pumpp.task.DynamicLabelTransformer('genre', 'tag_gtzan',
                                            hop_length=hop_length,
                                            sr=sr)

    data = tc.transform_annotation(tag_dynamic, tag_dynamic.duration)

    inverse = tc.inverse(data['tags'])
    data2 = tc.transform_annotation(inverse, tag_dynamic.duration)

    assert np.allclose(data['tags'], data2['tags'])
