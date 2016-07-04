#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import pumpp

import numpy as np

import jams

from nose.tools import eq_, raises

SR = 22050
HOP_LENGTH = 512
N_REPEAT = SR // HOP_LENGTH


def test_task_chord_present():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=5.0))

    ann = jams.Annotation(namespace='chord')

    ann.append(time=0, duration=1.0, value='C:maj')
    ann.append(time=1, duration=1.0, value='C:maj/3')
    ann.append(time=3, duration=1.0, value='D:maj')
    ann.append(time=4, duration=1.0, value='N')

    jam.annotations.append(ann)

    # One second = one frame
    T = pumpp.task.ChordTransformer(name='chord')

    output = T.transform(jam)

    # Make sure we have the mask
    eq_(output['chord/mask'], True)

    # Ideal vectors:
    # pcp = Cmaj, Cmaj, N, Dmaj, N
    # root: C, C, N, D, N
    # bass: C, E, N, D, N
    pcp_true = np.asarray([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    root_true = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    bass_true = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    assert np.allclose(output['chord/pitches'], np.repeat(pcp_true, N_REPEAT, axis=0))
    assert np.allclose(output['chord/root'], np.repeat(root_true, N_REPEAT, axis=0))
    assert np.allclose(output['chord/bass'], np.repeat(bass_true, N_REPEAT, axis=0))


def test_task_chord_absent():

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    T = pumpp.task.ChordTransformer(name='chord')

    output = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(output['chord/mask'], False)

    # Check the shape
    assert np.allclose(output['chord/pitches'].shape, [1, 4 * N_REPEAT, 12])
    assert np.allclose(output['chord/root'].shape, [1, 4 * N_REPEAT, 13])
    assert np.allclose(output['chord/bass'].shape, [1, 4 * N_REPEAT, 13])

    # Make sure it's empty
    assert not np.any(output['chord/pitches'])
    assert not np.any(output['chord/root'][:, :, :12])
    assert not np.any(output['chord/bass'][:, :, :12])
    assert np.all(output['chord/root'][:, :, 12])
    assert np.all(output['chord/bass'][:, :, 12])


def test_task_simple_chord_present():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=5.0))

    ann = jams.Annotation(namespace='chord')

    ann.append(time=0, duration=1.0, value='C:maj')
    ann.append(time=1, duration=1.0, value='C:maj/3')
    ann.append(time=3, duration=1.0, value='D:maj')
    ann.append(time=4, duration=1.0, value='N')

    jam.annotations.append(ann)

    # One second = one frame
    T = pumpp.task.SimpleChordTransformer(name='chord_s')

    output = T.transform(jam)

    # Make sure we have the mask
    eq_(output['chord_s/mask'], True)

    # Ideal vectors:
    # pcp = Cmaj, Cmaj, N, Dmaj, N
    pcp_true = np.asarray([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert np.allclose(output['chord_s/pitches'], np.repeat(pcp_true, N_REPEAT, axis=0))

def test_task_simple_chord_absent():

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    T = pumpp.task.SimpleChordTransformer(name='chord_s')

    output = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(output['chord_s/mask'], False)

    # Check the shape
    assert np.allclose(output['chord_s/pitches'].shape, [1, 4 * N_REPEAT, 12])

    # Make sure it's empty
    assert not np.any(output['chord_s/pitches'])


def test_task_tslabel_present():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='tag_open')

    ann.append(time=0, duration=1.0, value='alpha')
    ann.append(time=0, duration=1.0, value='beta')
    ann.append(time=1, duration=1.0, value='23')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)
    T = pumpp.task.DynamicLabelTransformer(namespace='tag_open',
                                              name='madeup',
                                              labels=labels)

    output = T.transform(jam)

    # Mask should be true
    eq_(output['madeup/mask'], True)

    y = output['madeup/tags']

    # Check the shape
    assert np.allclose(y.shape, [1, 4 * N_REPEAT, len(labels)])

    # Decode the labels
    predictions = T.encoder.inverse_transform(y[0, ::N_REPEAT])

    true_labels = [['alpha', 'beta'], [], [], ['disco']]

    for t, p in zip(true_labels, predictions):
        eq_(set(t), set(p))


def test_task_tslabel_absent():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    T = pumpp.task.DynamicLabelTransformer(namespace='tag_open',
                                           name='madeup',
                                           labels=labels)

    output = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(output['madeup/mask'], False)
    y = output['madeup/tags']

    # Check the shape
    assert np.allclose(y.shape, [1, 4 * N_REPEAT, len(labels)])

    # Make sure it's empty
    assert not np.any(y)


def test_task_glabel_absent():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    T = pumpp.task.StaticLabelTransformer(namespace='tag_open',
                                          name='madeup',
                                          labels=labels)

    output = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(output['madeup/mask'], False)

    # Check the shape
    eq_(output['madeup/tags'].ndim, 2)
    eq_(output['madeup/tags'].shape[1], len(labels))

    # Make sure it's empty
    assert not np.any(output['madeup/tags'])


def test_task_glabel_present():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='tag_open')

    ann.append(time=0, duration=1.0, value='alpha')
    ann.append(time=0, duration=1.0, value='beta')
    ann.append(time=1, duration=1.0, value='23')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)
    T = pumpp.task.StaticLabelTransformer(namespace='tag_open',
                                          name='madeup',
                                          labels=labels)

    output = T.transform(jam)

    # Mask should be true
    eq_(output['madeup/mask'], True)

    # Check the shape
    eq_(output['madeup/tags'].ndim, 2)
    eq_(output['madeup/tags'].shape[1], len(labels))

    # Decode the labels
    predictions = T.encoder.inverse_transform(output['madeup/tags'][0].reshape((1, -1)))[0]

    true_labels = ['alpha', 'beta', 'disco']

    eq_(set(true_labels), set(predictions))


def test_task_vector_absent():

    def __test(dimension, name):

        var_name = '{:s}/vector'.format(name)
        mask_name = '{:s}/mask'.format(name)

        jam = jams.JAMS(file_metadata=dict(duration=4.0))
        T = pumpp.task.VectorTransformer(namespace='vector',
                                         dimension=dimension,
                                         name=name)

        output = T.transform(jam)

        # Mask should be false since we have no matching namespace
        eq_(output[mask_name], False)

        # Check the shape
        eq_(output[var_name].ndim, 2)
        eq_(output[var_name].shape[1], dimension)

        # Make sure it's empty
        assert not np.any(output[var_name])

    for dimension in [1, 2, 4]:
        yield __test, dimension, 'collab'
        yield __test, dimension, 'vec'


def test_task_vector_present():

    def __test(target_dimension, data_dimension, name):
        var_name = '{:s}/vector'.format(name)
        mask_name = '{:s}/mask'.format(name)

        jam = jams.JAMS(file_metadata=dict(duration=4.0))
        T = pumpp.task.VectorTransformer(namespace='vector',
                                         dimension=target_dimension,
                                         name=name)

        ann = jams.Annotation(namespace='vector')
        ann.append(time=0, duration=1,
                   value=list(np.random.randn(data_dimension)))

        jam.annotations.append(ann)

        output = T.transform(jam)

        # Mask should be false since we have no matching namespace
        eq_(output[mask_name], True)

        # Check the shape
        eq_(output[var_name].ndim, 2)
        eq_(output[var_name].shape[1], target_dimension)

        # Make sure it's empty
        assert np.allclose(output[var_name], ann.data.loc[0].value)

    for target_d in [1, 2, 4]:
        for data_d in [1, 2, 4]:
            if target_d != data_d:
                tf = raises(RuntimeError)(__test)
            else:
                tf = __test
            yield tf, target_d, data_d, 'collab'
            yield tf, target_d, data_d, 'vec'


def test_task_beat_present():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='beat')

    ann.append(time=0, duration=0.0, value=1)
    ann.append(time=1, duration=0.0, value=2)
    ann.append(time=2, duration=0.0, value=3)
    ann.append(time=3, duration=0.0, value=1)

    jam.annotations.append(ann)

    T = pumpp.task.BeatTransformer(name='beat')

    output = T.transform(jam)

    # Make sure we have the masks
    eq_(output['beat/mask'], True)
    eq_(output['beat/mask_downbeat'], True)

    # The first channel measures beats
    # The second channel measures downbeats
    assert np.allclose(output['beat/beat'].shape, [1, 4 * N_REPEAT, 1])
    assert np.allclose(output['beat/downbeat'].shape, [1, 4 * N_REPEAT, 1])

    # Ideal vectors:
    #   a beat every second (two samples)
    #   a downbeat every three seconds (6 samples)

    beat_true = np.asarray([[1, 1, 1, 1]]).T
    downbeat_true = np.asarray([[1, 0, 0, 1]]).T

    assert np.allclose(output['beat/beat'][0, ::N_REPEAT], beat_true)
    assert np.allclose(output['beat/downbeat'][0, ::N_REPEAT], downbeat_true)


def test_task_beat_nometer():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='beat')

    ann.append(time=0, duration=0.0)
    ann.append(time=1, duration=0.0)
    ann.append(time=2, duration=0.0)
    ann.append(time=3, duration=0.0)

    jam.annotations.append(ann)

    # One second = one frame
    T = pumpp.task.BeatTransformer(name='beat')

    output = T.transform(jam)

    # Make sure we have the mask
    eq_(output['beat/mask'], True)
    eq_(output['beat/mask_downbeat'], False)

    # Check the shape: 4 seconds at 2 samples per second
    assert np.allclose(output['beat/beat'].shape, [1, 4 * N_REPEAT, 1])
    assert np.allclose(output['beat/downbeat'].shape, [1, 4 * N_REPEAT, 1])

    # Ideal vectors:
    #   a beat every second (two samples)
    #   no downbeats

    beat_true = np.asarray([1, 1, 1, 1])
    downbeat_true = np.asarray([0, 0, 0, 0])

    assert np.allclose(output['beat/beat'][0, ::N_REPEAT], beat_true)
    assert np.allclose(output['beat/downbeat'][0, ::N_REPEAT], downbeat_true)


def test_task_beat_absent():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    # One second = one frame
    T = pumpp.task.BeatTransformer(name='beat')

    output = T.transform(jam)

    # Make sure we have the mask
    eq_(output['beat/mask'], False)
    eq_(output['beat/mask_downbeat'], False)

    # Check the shape: 4 seconds at 2 samples per second
    assert np.allclose(output['beat/beat'].shape, [1, 4 * N_REPEAT, 1])
    assert np.allclose(output['beat/downbeat'].shape, [1, 4 * N_REPEAT, 1])
    assert not np.any(output['beat/beat'])
    assert not np.any(output['beat/downbeat'])
