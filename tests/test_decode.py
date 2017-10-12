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
def ann_vector():

    ann = jams.Annotation(namespace='vector', duration=1)

    ann.append(time=0, duration=0, value=np.arange(32))
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


@pytest.fixture()
def ann_chord():

    ann = jams.Annotation(namespace='chord', duration=5)

    for t, c in [(0, 'C'),
                 (1, 'C:maj'),
                 (2, 'D:min/3'),
                 (3, 'F#:7(*5)'),
                 (4, 'G:sus2')]:
        ann.append(time=t, duration=1, value=c)

    return ann


@pytest.fixture()
def ann_segment():

    ann = jams.Annotation(namespace='segment_open', duration=5)

    for t, c in [(0, 'A'),
                 (1, 'B'),
                 (2, 'A'),
                 (3, 'B'),
                 (4, 'C')]:
        ann.append(time=t, duration=1, value=c)

    return ann


def test_decode_tags_dynamic_hard(sr, hop_length, ann_tag):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tc = pumpp.task.DynamicLabelTransformer('genre', 'tag_gtzan',
                                            hop_length=hop_length,
                                            sr=sr)

    data = tc.transform_annotation(ann_tag, ann_tag.duration)

    inverse = tc.inverse(data['tags'], duration=ann_tag.duration)
    for obs in inverse:
        assert 0. <= obs.confidence <= 1.
    data2 = tc.transform_annotation(inverse, ann_tag.duration)

    assert np.allclose(data['tags'], data2['tags'])


def test_decode_tags_dynamic_soft(sr, hop_length, ann_tag):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tc = pumpp.task.DynamicLabelTransformer('genre', 'tag_gtzan',
                                            hop_length=hop_length,
                                            sr=sr)

    data = tc.transform_annotation(ann_tag, ann_tag.duration)

    # Soften the data, but preserve the decisions
    tags_predict = data['tags'] * 0.51 + 0.1
    inverse = tc.inverse(tags_predict, duration=ann_tag.duration)
    for obs in inverse:
        assert 0. <= obs.confidence <= 1.
    data2 = tc.transform_annotation(inverse, ann_tag.duration)

    assert np.allclose(data['tags'], data2['tags'])


def test_decode_tags_static_hard(ann_tag):

    tc = pumpp.task.StaticLabelTransformer('genre', 'tag_gtzan')

    data = tc.transform_annotation(ann_tag, ann_tag.duration)
    inverse = tc.inverse(data['tags'], ann_tag.duration)
    for obs in inverse:
        assert 0. <= obs.confidence <= 1.
    data2 = tc.transform_annotation(inverse, ann_tag.duration)

    assert np.allclose(data['tags'], data2['tags'])


def test_decode_tags_static_soft(ann_tag):

    tc = pumpp.task.StaticLabelTransformer('genre', 'tag_gtzan')

    data = tc.transform_annotation(ann_tag, ann_tag.duration)
    tags_predict = data['tags'] * 0.51 + 0.1

    inverse = tc.inverse(tags_predict, ann_tag.duration)
    for obs in inverse:
        assert 0. <= obs.confidence <= 1.
    data2 = tc.transform_annotation(inverse, ann_tag.duration)

    assert np.allclose(data['tags'], data2['tags'])


def test_decode_beat_hard(sr, hop_length, ann_beat):

    tc = pumpp.task.BeatTransformer('beat', sr=sr, hop_length=hop_length)

    data = tc.transform_annotation(ann_beat, ann_beat.duration)
    inverse = tc.inverse(data['beat'], duration=ann_beat.duration)
    data2 = tc.transform_annotation(inverse, ann_beat.duration)

    assert np.allclose(data['beat'], data2['beat'])


def test_decode_beat_soft(sr, hop_length, ann_beat):

    tc = pumpp.task.BeatTransformer('beat', sr=sr, hop_length=hop_length)

    data = tc.transform_annotation(ann_beat, ann_beat.duration)
    beat_pred = data['beat'] * 0.51 + 0.1

    inverse = tc.inverse(beat_pred, duration=ann_beat.duration)
    data2 = tc.transform_annotation(inverse, ann_beat.duration)

    assert np.allclose(data['beat'], data2['beat'])


def test_decode_beat_downbeat_hard(sr, hop_length, ann_beat):

    tc = pumpp.task.BeatTransformer('beat', sr=sr, hop_length=hop_length)

    data = tc.transform_annotation(ann_beat, ann_beat.duration)
    inverse = tc.inverse(data['beat'], downbeat=data['downbeat'],
                         duration=ann_beat.duration)
    data2 = tc.transform_annotation(inverse, ann_beat.duration)

    assert np.allclose(data['beat'], data2['beat'])


def test_decode_beat_downbeat_soft(sr, hop_length, ann_beat):

    tc = pumpp.task.BeatTransformer('beat', sr=sr, hop_length=hop_length)

    data = tc.transform_annotation(ann_beat, ann_beat.duration)
    beat_pred = data['beat'] * 0.51 + 0.1
    dbeat_pred = data['downbeat'] * 0.51 + 0.1
    inverse = tc.inverse(beat_pred, downbeat=dbeat_pred,
                         duration=ann_beat.duration)
    data2 = tc.transform_annotation(inverse, ann_beat.duration)

    assert np.allclose(data['beat'], data2['beat'])


def test_decode_vector(ann_vector):

    tc = pumpp.task.VectorTransformer('cf', 'vector', 32)

    data = tc.transform_annotation(ann_vector, ann_vector.duration)

    inverse = tc.inverse(data['vector'], duration=ann_vector.duration)

    data2 = tc.transform_annotation(inverse, ann_vector.duration)

    assert np.allclose(data['vector'], data2['vector'])


@pytest.mark.xfail(raises=NotImplementedError)
def test_decode_chord(sr, hop_length, ann_chord):

    tc = pumpp.task.ChordTransformer('chord', sr=sr, hop_length=hop_length)

    data = tc.transform_annotation(ann_chord, ann_chord.duration)
    inverse = tc.inverse(data['pitch'], data['root'], data['bass'],
                         duration=ann_chord.duration)
    data2 = tc.transform_annotation(inverse, ann_chord.duration)

    assert np.allclose(data['pitch'], data2['pitch'])
    assert np.allclose(data['root'], data2['root'])
    assert np.allclose(data['bass'], data2['bass'])


@pytest.mark.xfail(raises=NotImplementedError)
def test_decode_simplechord(sr, hop_length, ann_chord):

    tc = pumpp.task.SimpleChordTransformer('chord', sr=sr,
                                           hop_length=hop_length)

    data = tc.transform_annotation(ann_chord, ann_chord.duration)
    inverse = tc.inverse(data['pitch'], duration=ann_chord.duration)
    data2 = tc.transform_annotation(inverse, ann_chord.duration)

    assert np.allclose(data['pitch'], data2['pitch'])


def test_decode_chordtag_hard_dense(sr, hop_length, ann_chord):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tc = pumpp.task.ChordTagTransformer('chord', vocab='3567s',
                                        hop_length=hop_length,
                                        sr=sr, sparse=False)

    data = tc.transform_annotation(ann_chord, ann_chord.duration)

    inverse = tc.inverse(data['chord'], duration=ann_chord.duration)
    for obs in inverse:
        assert 0 <= obs.confidence <= 1.
    data2 = tc.transform_annotation(inverse, ann_chord.duration)

    assert np.allclose(data['chord'], data2['chord'])


def test_decode_chordtag_soft_dense(sr, hop_length, ann_chord):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tc = pumpp.task.ChordTagTransformer('chord', vocab='3567s',
                                        hop_length=hop_length,
                                        sr=sr, sparse=False)

    data = tc.transform_annotation(ann_chord, ann_chord.duration)

    chord_predict = data['chord'] * 0.51 + 0.1
    inverse = tc.inverse(chord_predict, duration=ann_chord.duration)

    for obs in inverse:
        assert 0 <= obs.confidence <= 1.

    data2 = tc.transform_annotation(inverse, ann_chord.duration)

    assert np.allclose(data['chord'], data2['chord'])


def test_decode_chordtag_hard_sparse_sparse(sr, hop_length, ann_chord):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tc = pumpp.task.ChordTagTransformer('chord', vocab='3567s',
                                        hop_length=hop_length,
                                        sr=sr, sparse=True)

    data = tc.transform_annotation(ann_chord, ann_chord.duration)

    inverse = tc.inverse(data['chord'], duration=ann_chord.duration)
    for obs in inverse:
        assert 0 <= obs.confidence <= 1.
    data2 = tc.transform_annotation(inverse, ann_chord.duration)

    assert np.allclose(data['chord'], data2['chord'])


def test_decode_chordtag_hard_dense_sparse(sr, hop_length, ann_chord):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tcd = pumpp.task.ChordTagTransformer('chord', vocab='3567s',
                                         hop_length=hop_length,
                                         sr=sr, sparse=False)

    tcs = pumpp.task.ChordTagTransformer('chord', vocab='3567s',
                                         hop_length=hop_length,
                                         sr=sr, sparse=True)

    # Make a hard, dense encoding of the data
    data = tcd.transform_annotation(ann_chord, ann_chord.duration)

    # Invert using the sparse encoder
    inverse = tcs.inverse(data['chord'], duration=ann_chord.duration)
    for obs in inverse:
        assert 0 <= obs.confidence <= 1.
    data2 = tcs.transform_annotation(inverse, ann_chord.duration)

    dense_positions = np.where(data['chord'])[1]
    sparse_positions = data2['chord'][:, 0]
    assert np.allclose(dense_positions, sparse_positions)


def test_decode_chordtag_soft_dense_sparse(sr, hop_length, ann_chord):

    # This test encodes an annotation, decodes it, and then re-encodes it
    # It passes if the re-encoded version matches the initial encoding
    tcd = pumpp.task.ChordTagTransformer('chord', vocab='3567s',
                                         hop_length=hop_length,
                                         sr=sr, sparse=False)

    tcs = pumpp.task.ChordTagTransformer('chord', vocab='3567s',
                                         hop_length=hop_length,
                                         sr=sr, sparse=True)

    # Make a soft, dense encoding of the data
    data = tcd.transform_annotation(ann_chord, ann_chord.duration)

    chord_predict = data['chord'] * 0.51 + 0.1
    # Invert using the sparse encoder
    inverse = tcs.inverse(chord_predict, duration=ann_chord.duration)
    for obs in inverse:
        assert 0 <= obs.confidence <= 1.
    data2 = tcs.transform_annotation(inverse, ann_chord.duration)

    dense_positions = np.where(data['chord'])[1]
    sparse_positions = data2['chord'][:, 0]
    assert np.allclose(dense_positions, sparse_positions)


@pytest.mark.xfail(raises=NotImplementedError)
def test_decode_structure(sr, hop_length, ann_segment):

    tc = pumpp.task.StructureTransformer('struct', sr=sr,
                                         hop_length=hop_length)

    data = tc.transform_annotation(ann_segment, ann_segment.duration)
    inverse = tc.inverse(data['agree'], duration=ann_segment.duration)
    data2 = tc.transform_annotation(inverse, ann_segment.duration)

    assert np.allclose(data['agree'], data2['agree'])


@pytest.mark.xfail(raises=NotImplementedError)
def test_decode_beatpos(sr, hop_length, ann_beat):

    tc = pumpp.task.BeatPositionTransformer('beat', sr=sr,
                                            max_divisions=12,
                                            hop_length=hop_length)

    data = tc.transform_annotation(ann_beat, ann_beat.duration)
    inverse = tc.inverse(data['position'], duration=ann_beat.duration)
    data2 = tc.transform_annotation(inverse, ann_beat.duration)

    assert np.allclose(data['position'], data2['position'])
