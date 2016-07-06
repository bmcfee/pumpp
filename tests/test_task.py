#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import numpy as np
import pytest
import jams

import pumpp


SR = 22050
HOP_LENGTH = 512
N_REPEAT = SR // HOP_LENGTH


def shape_match(sh1, sh2):

    for i, j in zip(sh1, sh2):
        if j is None:
            continue
        if i != j:
            return False

    return True

def type_match(x, y):

    return np.issubdtype(x, y) and np.issubdtype(y, x)


def test_task_chord_fields():

    trans = pumpp.task.ChordTransformer(name='mychord')

    assert set(trans.fields.keys()) == set(['mychord/pitch',
                                            'mychord/root',
                                            'mychord/bass'])

    assert trans.fields['mychord/pitch'].shape == (None, 12)
    assert trans.fields['mychord/pitch'].dtype is np.bool
    assert trans.fields['mychord/root'].shape == (None, 13)
    assert trans.fields['mychord/root'].dtype is np.bool
    assert trans.fields['mychord/bass'].shape == (None, 13)
    assert trans.fields['mychord/bass'].dtype is np.bool

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
    trans = pumpp.task.ChordTransformer(name='chord')

    output = trans.transform(jam)

    # Make sure we have the mask
    assert np.all(output['chord/_valid'] == [0, 5 * trans.sr // trans.hop_length])

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

    assert np.all(output['chord/pitch'] == np.repeat(pcp_true, N_REPEAT, axis=0))
    assert np.all(output['chord/root'] == np.repeat(root_true, N_REPEAT, axis=0))
    assert np.all(output['chord/bass'] == np.repeat(bass_true, N_REPEAT, axis=0))

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_chord_absent():

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.ChordTransformer(name='chord')

    output = trans.transform(jam)

    # Valid range is 0 since we have no matching namespace
    assert not np.any(output['chord/_valid'])

    # Check the shape
    assert output['chord/pitch'].shape == (1, 4 * N_REPEAT, 12)
    assert output['chord/root'].shape == (1, 4 * N_REPEAT, 13)
    assert output['chord/bass'].shape == (1, 4 * N_REPEAT, 13)

    # Make sure it's empty
    assert not np.any(output['chord/pitch'])
    assert not np.any(output['chord/root'][:, :, :12])
    assert not np.any(output['chord/bass'][:, :, :12])
    assert np.all(output['chord/root'][:, :, 12])
    assert np.all(output['chord/bass'][:, :, 12])

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_simple_chord_fields():

    trans = pumpp.task.SimpleChordTransformer(name='simple_chord')

    assert set(trans.fields.keys()) == set(['simple_chord/pitch'])
    assert trans.fields['simple_chord/pitch'].shape == (None, 12)
    assert trans.fields['simple_chord/pitch'].dtype is np.bool


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
    trans = pumpp.task.SimpleChordTransformer(name='chord_s')

    output = trans.transform(jam)

    # Make sure we have the mask
    assert np.all(output['chord_s/_valid'] == [0, 5 * trans.sr // trans.hop_length])

    # Ideal vectors:
    # pcp = Cmaj, Cmaj, N, Dmaj, N
    pcp_true = np.asarray([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert np.all(output['chord_s/pitch'] == np.repeat(pcp_true, N_REPEAT, axis=0))
    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_simple_chord_absent():

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.SimpleChordTransformer(name='chord_s')

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output['chord_s/_valid'])

    # Check the shape
    assert output['chord_s/pitch'].shape == (1, 4 * N_REPEAT, 12)

    # Make sure it's empty
    assert not np.any(output['chord_s/pitch'])

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_dlabel_present():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='tag_open')

    ann.append(time=0, duration=1.0, value='alpha')
    ann.append(time=0, duration=1.0, value='beta')
    ann.append(time=1, duration=1.0, value='23')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)
    trans = pumpp.task.DynamicLabelTransformer(namespace='tag_open',
                                               name='madeup',
                                               labels=labels)

    output = trans.transform(jam)

    # Mask should be true
    assert np.all(output['madeup/_valid'] == [0, 4 * trans.sr // trans.hop_length])

    y = output['madeup/tags']

    # Check the shape
    assert y.shape == (1, 4 * N_REPEAT, len(labels))

    # Decode the labels
    predictions = trans.encoder.inverse_transform(y[0, ::N_REPEAT])

    true_labels = [['alpha', 'beta'], [], [], ['disco']]

    for truth, pred in zip(true_labels, predictions):
        assert set(truth) == set(pred)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)



def test_task_dlabel_absent():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.DynamicLabelTransformer(namespace='tag_open',
                                               name='madeup',
                                               labels=labels)

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output['madeup/_valid'])

    y = output['madeup/tags']

    # Check the shape
    assert y.shape == (1, 4 * N_REPEAT, len(labels))

    # Make sure it's empty
    assert not np.any(y)
    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)



def test_task_slabel_absent():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.StaticLabelTransformer(namespace='tag_open',
                                              name='madeup',
                                              labels=labels)

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output['madeup/_valid'])

    # Check the shape
    assert output['madeup/tags'].ndim == 2
    assert output['madeup/tags'].shape[1] == len(labels)

    # Make sure it's empty
    assert not np.any(output['madeup/tags'])

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_slabel_present():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='tag_open')

    ann.append(time=0, duration=1.0, value='alpha')
    ann.append(time=0, duration=1.0, value='beta')
    ann.append(time=1, duration=1.0, value='23')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)
    trans = pumpp.task.StaticLabelTransformer(namespace='tag_open',
                                              name='madeup',
                                              labels=labels)

    output = trans.transform(jam)

    # Mask should be true
    assert np.all(output['madeup/_valid'] == [0, 4 * trans.sr // trans.hop_length])

    # Check the shape
    assert output['madeup/tags'].ndim == 2
    assert output['madeup/tags'].shape[1] == len(labels)

    # Decode the labels
    predictions = trans.encoder.inverse_transform(output['madeup/tags'][0].reshape((1, -1)))[0]

    true_labels = ['alpha', 'beta', 'disco']

    assert set(true_labels) == set(predictions)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)



@pytest.mark.parametrize('dimension', [1, 2, 4])
@pytest.mark.parametrize('name', ['collab', 'vec'])
def test_task_vector_absent(dimension, name):

    var_name = '{:s}/vector'.format(name)
    mask_name = '{:s}/_valid'.format(name)

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.VectorTransformer(namespace='vector',
                                         dimension=dimension,
                                         name=name)

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output[mask_name])

    # Check the shape
    assert output[var_name].ndim == 2
    assert output[var_name].shape[1] == dimension

    # Make sure it's empty
    assert not np.any(output[var_name])

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


@pytest.mark.parametrize('name', ['collab', 'vector'])
@pytest.mark.parametrize('target_dimension, data_dimension',
                         [(1, 1), (2, 2), (4, 4),
                          pytest.mark.xfail((2, 3), raises=RuntimeError)])
def test_task_vector_present(target_dimension, data_dimension, name):
    var_name = '{:s}/vector'.format(name)
    mask_name = '{:s}/_valid'.format(name)

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.VectorTransformer(namespace='vector',
                                         dimension=target_dimension,
                                         name=name)

    ann = jams.Annotation(namespace='vector')
    ann.append(time=0, duration=1,
               value=list(np.random.randn(data_dimension)))

    jam.annotations.append(ann)

    output = trans.transform(jam)

    assert np.all(output[mask_name] == [0, 4 * trans.sr // trans.hop_length])

    # Check the shape
    assert output[var_name].ndim == 2
    assert output[var_name].shape[1] == target_dimension

    # Make sure it's empty
    assert np.allclose(output[var_name], ann.data.loc[0].value)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_beat_present():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='beat')

    ann.append(time=0, duration=0.0, value=1)
    ann.append(time=1, duration=0.0, value=2)
    ann.append(time=2, duration=0.0, value=3)
    ann.append(time=3, duration=0.0, value=1)

    jam.annotations.append(ann)

    trans = pumpp.task.BeatTransformer(name='beat')

    output = trans.transform(jam)

    # Make sure we have the masks
    assert np.all(output['beat/_valid'] == [0, 4 * trans.sr // trans.hop_length])
    assert output['beat/mask_downbeat']

    # The first channel measures beats
    # The second channel measures downbeats
    assert output['beat/beat'].shape == (1, 4 * N_REPEAT, 1)
    assert output['beat/downbeat'].shape == (1, 4 * N_REPEAT, 1)

    # Ideal vectors:
    #   a beat every second (two samples)
    #   a downbeat every three seconds (6 samples)

    beat_true = np.asarray([[1, 1, 1, 1]]).T
    downbeat_true = np.asarray([[1, 0, 0, 1]]).T

    assert np.all(output['beat/beat'][0, ::N_REPEAT] == beat_true)
    assert np.all(output['beat/downbeat'][0, ::N_REPEAT] == downbeat_true)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


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
    trans = pumpp.task.BeatTransformer(name='beat')

    output = trans.transform(jam)

    # Make sure we have the mask
    assert np.all(output['beat/_valid'] == [0, 4 * trans.sr // trans.hop_length])
    assert not output['beat/mask_downbeat']

    # Check the shape: 4 seconds at 2 samples per second
    assert output['beat/beat'].shape == (1, 4 * N_REPEAT, 1)
    assert output['beat/downbeat'].shape == (1, 4 * N_REPEAT, 1)

    # Ideal vectors:
    #   a beat every second (two samples)
    #   no downbeats

    beat_true = np.asarray([1, 1, 1, 1])
    downbeat_true = np.asarray([0, 0, 0, 0])

    assert np.all(output['beat/beat'][0, ::N_REPEAT] == beat_true)
    assert np.all(output['beat/downbeat'][0, ::N_REPEAT] == downbeat_true)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_beat_absent():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    # One second = one frame
    trans = pumpp.task.BeatTransformer(name='beat')

    output = trans.transform(jam)

    # Make sure we have the mask
    assert not np.any(output['beat/_valid'])
    assert not output['beat/mask_downbeat']

    # Check the shape: 4 seconds at 2 samples per second
    assert output['beat/beat'].shape == (1, 4 * N_REPEAT, 1)
    assert output['beat/downbeat'].shape == (1, 4 * N_REPEAT, 1)
    assert not np.any(output['beat/beat'])
    assert not np.any(output['beat/downbeat'])

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_noprefix():

    labels = ['foo', 'bar', 'baz']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.StaticLabelTransformer(namespace='tag_open',
                                              name=None, labels=labels) 

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output['_valid'])

    # Check the shape
    assert output['tags'].ndim == 2
    assert output['tags'].shape[1] == len(labels)

    # Make sure it's empty
    assert not np.any(output['tags'])

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)

