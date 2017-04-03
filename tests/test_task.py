#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import numpy as np
import pytest
import jams

import pumpp


@pytest.fixture()
def SR():
    return 22050


@pytest.fixture()
def HOP_LENGTH():
    return 512


@pytest.fixture(params=['3', '35', '3567', '3567s'])
def VOCAB(request):
    yield request.param


@pytest.fixture(params=[False, True])
def SPARSE(request):
    return request.param


def shape_match(sh1, sh2):

    for i, j in zip(sh1, sh2):
        if j is None:
            continue
        if i != j:
            return False

    return True


def type_match(x, y):

    return np.issubdtype(x, y) and np.issubdtype(y, x)


def test_task_chord_fields(SPARSE):

    trans = pumpp.task.ChordTransformer(name='mychord', sparse=SPARSE)

    assert set(trans.fields.keys()) == set(['mychord/pitch',
                                            'mychord/root',
                                            'mychord/bass'])

    assert trans.fields['mychord/pitch'].shape == (None, 12)
    assert trans.fields['mychord/pitch'].dtype is np.bool

    if SPARSE:
        assert trans.fields['mychord/root'].shape == (None, 1)
        assert np.issubdtype(trans.fields['mychord/root'].dtype, np.int)
        assert trans.fields['mychord/bass'].shape == (None, 1)
        assert np.issubdtype(trans.fields['mychord/bass'].dtype, np.int)
    else:
        assert trans.fields['mychord/root'].shape == (None, 13)
        assert trans.fields['mychord/root'].dtype is np.bool
        assert trans.fields['mychord/bass'].shape == (None, 13)
        assert trans.fields['mychord/bass'].dtype is np.bool


def test_task_chord_present(SR, HOP_LENGTH, SPARSE):

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=5.0))

    ann = jams.Annotation(namespace='chord')

    ann.append(time=0, duration=1.0, value='C:maj')
    ann.append(time=1, duration=1.0, value='C:maj/3')
    ann.append(time=3, duration=1.0, value='D:maj')
    ann.append(time=4, duration=1.0, value='N')

    jam.annotations.append(ann)

    # One second = one frame
    trans = pumpp.task.ChordTransformer(name='chord',
                                        sr=SR, hop_length=HOP_LENGTH,
                                        sparse=SPARSE)

    output = trans.transform(jam)

    # Make sure we have the mask
    assert np.all(output['chord/_valid'] == [0, 5 * trans.sr //
                                             trans.hop_length])

    # Ideal vectors:
    # pcp = Cmaj, Cmaj, N, Dmaj, N
    # root: C, C, N, D, N
    # bass: C, E, N, D, N
    pcp_true = np.asarray([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    if SPARSE:
        root_true = np.asarray([[0],  [0],  [12], [2],  [12]])
        bass_true = np.asarray([[0],  [4],  [12], [2],  [12]])

    else:
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

    assert np.all(output['chord/pitch'] == np.repeat(pcp_true,
                                                     (SR // HOP_LENGTH),
                                                     axis=0))
    assert np.all(output['chord/root'] == np.repeat(root_true,
                                                    (SR // HOP_LENGTH),
                                                    axis=0))
    assert np.all(output['chord/bass'] == np.repeat(bass_true,
                                                    (SR // HOP_LENGTH),
                                                    axis=0))

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_chord_absent(SR, HOP_LENGTH, SPARSE):

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.ChordTransformer(name='chord',
                                        sr=SR, hop_length=HOP_LENGTH,
                                        sparse=SPARSE)

    output = trans.transform(jam)

    # Valid range is 0 since we have no matching namespace
    assert not np.any(output['chord/_valid'])

    # Check the shape
    assert output['chord/pitch'].shape == (1, 4 * (SR // HOP_LENGTH), 12)

    # Make sure it's empty
    assert not np.any(output['chord/pitch'])
    if SPARSE:
        assert output['chord/root'].shape == (1, 4 * (SR // HOP_LENGTH), 1)
        assert output['chord/bass'].shape == (1, 4 * (SR // HOP_LENGTH), 1)
        assert np.all(output['chord/root'] == 12)
        assert np.all(output['chord/bass'] == 12)
    else:
        assert output['chord/root'].shape == (1, 4 * (SR // HOP_LENGTH), 13)
        assert output['chord/bass'].shape == (1, 4 * (SR // HOP_LENGTH), 13)
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


def test_task_simple_chord_present(SR, HOP_LENGTH):

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
    assert np.all(output['chord_s/_valid'] == [0, 5 * trans.sr //
                                               trans.hop_length])

    # Ideal vectors:
    # pcp = Cmaj, Cmaj, N, Dmaj, N
    pcp_true = np.asarray([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert np.all(output['chord_s/pitch'] == np.repeat(pcp_true,
                                                       (SR // HOP_LENGTH),
                                                       axis=0))
    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_simple_chord_absent(SR, HOP_LENGTH):

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.SimpleChordTransformer(name='chord_s')

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output['chord_s/_valid'])

    # Check the shape
    assert output['chord_s/pitch'].shape == (1, 4 * (SR // HOP_LENGTH), 12)

    # Make sure it's empty
    assert not np.any(output['chord_s/pitch'])

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_dlabel_present(SR, HOP_LENGTH):
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='tag_open')

    ann.append(time=0, duration=1.0, value='alpha')
    ann.append(time=0, duration=1.0, value='beta')
    ann.append(time=1, duration=1.0, value='some nonsense')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)
    trans = pumpp.task.DynamicLabelTransformer(name='madeup',
                                               namespace='tag_open',
                                               labels=labels)

    output = trans.transform(jam)

    # Mask should be true
    assert np.all(output['madeup/_valid'] == [0, 4 * trans.sr //
                                              trans.hop_length])

    y = output['madeup/tags']

    # Check the shape
    assert y.shape == (1, 4 * (SR // HOP_LENGTH), len(labels))

    # Decode the labels
    predictions = trans.encoder.inverse_transform(y[0, ::(SR // HOP_LENGTH)])

    true_labels = [['alpha', 'beta'], [], [], ['disco']]

    for truth, pred in zip(true_labels, predictions):
        assert set(truth) == set(pred)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_dlabel_absent(SR, HOP_LENGTH):
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.DynamicLabelTransformer(name='madeup',
                                               namespace='tag_open',
                                               labels=labels)

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output['madeup/_valid'])

    y = output['madeup/tags']

    # Check the shape
    assert y.shape == (1, 4 * (SR // HOP_LENGTH), len(labels))

    # Make sure it's empty
    assert not np.any(y)
    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_dlabel_auto(SR, HOP_LENGTH):
    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.DynamicLabelTransformer(name='genre',
                                               namespace='tag_gtzan')

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output['genre/_valid'])

    y = output['genre/tags']

    # Check the shape
    assert y.shape == (1, 4 * (SR // HOP_LENGTH), 10)

    # Make sure it's empty
    assert not np.any(y)
    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_slabel_absent():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.StaticLabelTransformer(name='madeup',
                                              namespace='tag_open',
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
    ann.append(time=1, duration=1.0, value='some nonsense')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)
    trans = pumpp.task.StaticLabelTransformer(name='madeup',
                                              namespace='tag_open',
                                              labels=labels)

    output = trans.transform(jam)

    # Mask should be true
    assert np.all(output['madeup/_valid'] == [0, 4 * trans.sr //
                                              trans.hop_length])

    # Check the shape
    assert output['madeup/tags'].ndim == 2
    assert output['madeup/tags'].shape[1] == len(labels)

    # Decode the labels
    y_pred = output['madeup/tags'][0]
    predictions = trans.encoder.inverse_transform(y_pred.reshape((1, -1)))[0]

    true_labels = ['alpha', 'beta', 'disco']

    assert set(true_labels) == set(predictions)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_slabel_auto():
    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.StaticLabelTransformer(name='genre',
                                              namespace='tag_gtzan')

    output = trans.transform(jam)

    # Mask should be false since we have no matching namespace
    assert not np.any(output['genre/_valid'])

    # Check the shape
    assert output['genre/tags'].ndim == 2
    assert output['genre/tags'].shape[1] == 10

    # Make sure it's empty
    assert not np.any(output['genre/tags'])

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
        assert shape_match(output[key].shape, trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


@pytest.mark.parametrize('name', ['collab', 'vector'])
@pytest.mark.parametrize('target_dimension, data_dimension',
                         [(1, 1), (2, 2), (4, 4),
                          pytest.mark.xfail((2, 3), raises=pumpp.DataError)])
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
        assert shape_match(output[key].shape, trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_beat_present(SR, HOP_LENGTH):

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
    assert np.all(output['beat/_valid'] == [0, 4 * trans.sr //
                                            trans.hop_length])
    assert output['beat/mask_downbeat']

    # The first channel measures beats
    # The second channel measures downbeats
    assert output['beat/beat'].shape == (1, 4 * (SR // HOP_LENGTH), 1)
    assert output['beat/downbeat'].shape == (1, 4 * (SR // HOP_LENGTH), 1)

    # Ideal vectors:
    #   a beat every second (two samples)
    #   a downbeat every three seconds (6 samples)

    beat_true = np.asarray([[1, 1, 1, 1]]).T
    downbeat_true = np.asarray([[1, 0, 0, 1]]).T

    assert np.all(output['beat/beat'][0, ::(SR // HOP_LENGTH)] == beat_true)
    assert np.all(output['beat/downbeat'][0, ::(SR // HOP_LENGTH)] ==
                  downbeat_true)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_beat_nometer(SR, HOP_LENGTH):

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
    assert np.all(output['beat/_valid'] == [0, 4 * trans.sr //
                                            trans.hop_length])
    assert not output['beat/mask_downbeat']

    # Check the shape: 4 seconds at 2 samples per second
    assert output['beat/beat'].shape == (1, 4 * (SR // HOP_LENGTH), 1)
    assert output['beat/downbeat'].shape == (1, 4 * (SR // HOP_LENGTH), 1)

    # Ideal vectors:
    #   a beat every second (two samples)
    #   no downbeats

    beat_true = np.asarray([1, 1, 1, 1])
    downbeat_true = np.asarray([0, 0, 0, 0])

    assert np.all(output['beat/beat'][0, ::(SR // HOP_LENGTH)] == beat_true)
    assert np.all(output['beat/downbeat'][0, ::(SR // HOP_LENGTH)] ==
                  downbeat_true)

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_task_beat_absent(SR, HOP_LENGTH):

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    # One second = one frame
    trans = pumpp.task.BeatTransformer(name='beat')

    output = trans.transform(jam)

    # Make sure we have the mask
    assert not np.any(output['beat/_valid'])
    assert not output['beat/mask_downbeat']

    # Check the shape: 4 seconds at 2 samples per second
    assert output['beat/beat'].shape == (1, 4 * (SR // HOP_LENGTH), 1)
    assert output['beat/downbeat'].shape == (1, 4 * (SR // HOP_LENGTH), 1)
    assert not np.any(output['beat/beat'])
    assert not np.any(output['beat/downbeat'])

    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)


def test_transform_noprefix():

    labels = ['foo', 'bar', 'baz']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.StaticLabelTransformer(name=None,
                                              namespace='tag_open',
                                              labels=labels)

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


def test_transform_query():

    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='tag_open')

    ann.append(time=0, duration=1.0, value='alpha')
    ann.append(time=0, duration=1.0, value='beta')
    ann.append(time=1, duration=1.0, value='some nonsense')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)

    jam.annotations.append(jams.Annotation(namespace='tag_gtzan'))
    jam.annotations.append(jams.Annotation(namespace='tag_cal500'))

    trans = pumpp.task.StaticLabelTransformer(name='multi',
                                              namespace='tag_open',
                                              labels=labels)

    # First test with no query
    output = trans.transform(jam)
    assert output['multi/tags'].shape[0] == 3

    output = trans.transform(jam, query=dict(namespace='tag_open|tag_cal500'))
    assert output['multi/tags'].shape[0] == 2

    # This should make a null search, which produces a dummy (empty) annotation
    output = trans.transform(jam, query=dict(namespace='chord'))
    assert output['multi/tags'].shape[0] == 1
    assert not np.any(output['multi/_valid'])


def test_transform_coerce():

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.ChordTransformer(name='chord')
    jam.annotations.append(jams.Annotation(namespace='chord'))
    jam.annotations.append(jams.Annotation(namespace='chord_harte'))
    jam.annotations.append(jams.Annotation(namespace='tag_gtzan'))

    out = trans.transform(jam)

    assert out['chord/pitch'].shape[0] == 2


@pytest.mark.parametrize('vocab, vocab_size',
                         [('3', 26),
                          ('3s', 50),
                          ('35', 50),
                          ('35s', 74),
                          ('356', 74),
                          ('356s', 98),
                          ('3567', 146),
                          ('3567s', 170),
                          pytest.mark.xfail(('bad vocab', 1),
                                            raises=pumpp.ParameterError),
                          pytest.mark.xfail(('5', 1),
                                            raises=pumpp.ParameterError),
                          pytest.mark.xfail(('36', 1),
                                            raises=pumpp.ParameterError),
                          pytest.mark.xfail(('357', 1),
                                            raises=pumpp.ParameterError)])
def test_task_chord_tag_fields(vocab, vocab_size, SPARSE):

    trans = pumpp.task.ChordTagTransformer(name='mychord', vocab=vocab, sparse=SPARSE)

    assert set(trans.fields.keys()) == set(['mychord/chord'])

    if SPARSE:
        assert trans.fields['mychord/chord'].shape == (None, 1)
        assert np.issubdtype(trans.fields['mychord/chord'].dtype, np.int)
    else:
        assert trans.fields['mychord/chord'].shape == (None, vocab_size)
        assert trans.fields['mychord/chord'].dtype is np.bool


def test_task_chord_tag_present(SR, HOP_LENGTH, VOCAB, SPARSE):

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=13.0))

    ann = jams.Annotation(namespace='chord')

    Y_true = ['C:maj',          # 0
              'C:min6/3',       # 1
              'C:maj6/3',       # 2
              'Db:maj(*3)',     # 3
              'N',              # 4
              'C#:dim7(*3)/5',  # 5
              'C#:7',           # 6
              'C#:maj7',        # 7
              'C#:minmaj7',     # 8
              'C#:min7',        # 9
              'C#:hdim7',       # 10
              'G:sus2',         # 11
              'G:sus4']         # 12

    Y_true_out = ['C:maj',
                  'C:min6',
                  'C:maj6',
                  'C#:maj',
                  'N',
                  'C#:dim7',
                  'C#:7',
                  'C#:maj7',
                  'C#:minmaj7',
                  'C#:min7',
                  'C#:hdim7',
                  'G:sus2',
                  'G:sus4']

    if 's' not in VOCAB:
        Y_true_out[11] = 'X'       # sus2 -> X
        Y_true_out[12] = 'X'       # sus4 -> X
    if '6' not in VOCAB:
        Y_true_out[1] = 'C:min'    # min6 -> maj
        Y_true_out[2] = 'C:maj'    # maj6 -> maj
    if '7' not in VOCAB:
        Y_true_out[5] = 'C#:dim'   # dim7 -> dim
        Y_true_out[6] = 'C#:maj'   # 7 -> maj
        Y_true_out[7] = 'C#:maj'   # maj7 -> maj
        Y_true_out[8] = 'C#:min'   # minmaj7 -> min
        Y_true_out[9] = 'C#:min'   # min7 -> min
        Y_true_out[10] = 'C#:dim'  # hdim7 -> dim
    if '5' not in VOCAB:
        Y_true_out[5] = 'C#:min'   # dim7 -> dim -> min
        Y_true_out[6] = 'C#:maj'   # 7 -> maj
        Y_true_out[7] = 'C#:maj'   # maj7 -> maj
        Y_true_out[8] = 'C#:min'   # minmaj7 -> min
        Y_true_out[9] = 'C#:min'   # min7 -> min
        Y_true_out[10] = 'C#:min'  # hdim7 -> dim -> min

    for i, y in enumerate(Y_true):
        ann.append(time=i, duration=1.0, value=y)

    jam.annotations.append(ann)

    trans = pumpp.task.ChordTagTransformer(name='chord', vocab=VOCAB,
                                           sr=SR, hop_length=HOP_LENGTH,
                                           sparse=SPARSE)

    output = trans.transform(jam)

    # Make sure we have the mask
    assert np.all(output['chord/_valid'] == [0, 13 * trans.sr //
                                             trans.hop_length])

    # Decode the label encoding
    Y_pred = trans.encoder.inverse_transform(output['chord/chord'][0])
    if SPARSE:
        # Sparse label encoders use an extra output dimension
        Y_pred = Y_pred[:, 0]

    Y_expected = np.repeat(Y_true_out, (SR // HOP_LENGTH), axis=0)

    assert np.all(Y_pred == Y_expected)


def test_task_chord_tag_absent(SR, HOP_LENGTH, VOCAB, SPARSE):

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    trans = pumpp.task.ChordTagTransformer(name='chord',
                                           vocab=VOCAB,
                                           sr=SR, hop_length=HOP_LENGTH,
                                           sparse=SPARSE)

    output = trans.transform(jam)

    # Valid range is 0 since we have no matching namespace
    assert not np.any(output['chord/_valid'])

    # Make sure it's all no-chord
    Y_pred = trans.encoder.inverse_transform(output['chord/chord'][0])

    assert all([_ == 'X' for _ in Y_pred])

    # Check the shape
    for key in trans.fields:
        assert shape_match(output[key].shape[1:], trans.fields[key].shape)
        assert type_match(output[key].dtype, trans.fields[key].dtype)
