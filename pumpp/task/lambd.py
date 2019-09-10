#!/usr/bin/env python
# -*- enconding: utf-8 -*-
'''Generic annotation transformation'''

import numpy as np
import jams

from librosa import time_to_frames
from .base import BaseTaskTransformer
from .. import util

__all__ = ['LambdaTransformer']

class LambdaTransformer(BaseTaskTransformer):
    '''General annotation transformer.

    This tries to offer a flexible way of extracting jams observation values.

    Attributes
    ----------
    name : str
        The name of this transformer object

    namespace : str
        The JAMS namespace for this task

    fields : list of tuples/str - with signature (name, shape, dtype)
        The list of field definitions to be returned by ``reduce(values)``.
        These become the arguments to ``self.register``

        If value is of type ``object``, the fields can most likely be inferred
        from the schema.  If no fields are specified, all fields from the schema
        will be used.
        If fields are passed as strings, the dtype will be derived from the schema,
        and given a shape of (None, 1).

        If value is not of type `object`, a single field `(namespace, (None,1), value_type)`
        will be assumed.

    query : same format as ``observation.value``, callable [optional]
        a query compatible with ``match_query``.

        see ``match_query`` for a description.

    reduce : callable [optional]
        A function that takes the list of the observation values in each interval
        and returns a dictionary of extracted data keys as expected by Pump.
        if ``multi`` is False, the function will take a single observation
        value (useful for value_type == 'object', but possibly trivial otherwise.).

        If ``reduce`` is not specified, an appropriate function will be derived from
        the ``fields`` parameter and the jams annotation schema.

            If the schema value type is ``'object'``, it will set the output data dict
            to the values for each field taken from the observation dict.
            If ``multi`` is True, each field will contain a list of values, otherwise,
            each key will be the value of that field.

            If the value type is not an object and only one field is defined,
            the value is returned as {fields[0].name: value}.

    multi : bool [optional]:
        Whether to support multiple events in a single interval. If ``multi`` is False,
        it will select the value index according to `LambdaTransformer.SINGLE_INDEX`.
        If ``multi`` is True, the ``reduce`` function will receive all values in
        the interval.

    sr : number > 0
        The audio sampling rate

    hop_length : int > 0
        The hop length for annotation frames

    Examples
    --------
    ```python
    # all fields
    LambdaTransformer('scaper', 'scaper')
    # fields (shape=(None, 1), dtype=inferred): [
    #    label, source_file, source_time, event_time, event_duration, snr,
    #    time_stretch, pitch_shift, role],

    # all fields all events
    LambdaTransformer('scaper', 'scaper', multi=True)
    # fields (shape=(None, None), dtype=inferred): [
    #    label, source_file, source_time, event_time, event_duration, snr,
    #    time_stretch, pitch_shift, role],

    # infer dtype from schema
    LambdaTransformer('scaper', 'scaper', ['label', 'snr'])

    # equivalent with explicit dtype
    LambdaTransformer('scaper', 'scaper', [
        ('label', (None, 1), np.object_),
        ('snr', (None, 1), np.float),
    ])

    # snr of only foreground events (takes the last occurring event)
    LambdaTransformer(
        'scaper', 'scaper', ['snr'],
        query={'role': 'foreground'})

    # get encoded labels of all foreground events ()
    encoder = MultiLabelBinarizer().fit([POSSIBLE_LABELS])
    LambdaTransformer(
        'scaper', 'scaper', fields=['label'],
        query={'role': 'foreground'},
        multi=True,
        reduce=lambda values: {
            # TODO: make sure the shape comes out right
            'label': np.sum(encoder.transform([[v['label'] for v in values]]), axis=0),
        },
    )
    ```

    '''

    def __init__(self, name, namespace, fields=(), query=None, reduce=None,
                 multi=False, sample_index=-1, squeeze=True, **kw):
        super().__init__(name, namespace, **kw)
        self.value_query = query
        self.multi = multi
        self.squeeze = squeeze

        # infer missing fields / dtype for missing reducer
        fields, value_has_defined_keys = _check_fields(fields, namespace, multi)

        # get default reducer when none is specified
        if not self.reducer and not reduce:
            reduce = _default_reducer(fields, value_has_defined_keys, multi)

        # store reducer
        if reduce:
            self.reducer = reduce

        # register fields
        for name_, shape, dtype in fields:
            self.register(name_, shape, dtype)

        # fills any missing values from self.reducer
        self.FILL_DICT = {
            name_: fill_value(dtype) for name_, shape, dtype in fields
        }
        # select which value in the event of multiple values in interval
        self.SINGLE_INDEX = sample_index

    # Either override by passing a function like `__init__(reduce=lambda x: ...)`
    # or by overriding with a subclass.
    reducer = None

    def transform_annotation(self, ann, duration):
        # get observations as lists
        intervals, values = ann.to_interval_values()

        # filter values that match query
        intervals, values = zip(*[
            (i, d) for i, d in zip(intervals, values)
            if util.match_query(d, self.value_query)
        ])

        # get temporal slices, using one hot observation encoding
        idxs = np.eye(len(intervals))
        targets = self.encode_intervals(duration, intervals, idxs,
                                        multi=True, dtype=int, fill=0)

        # get the values in each interval window
        idxs = (np.where(i)[0] for i in targets)

        if self.multi:
            target_vals = ([values[j] for j in i] for i in idxs)
        else:
            target_vals = (values[i[self.SINGLE_INDEX]]
                           if len(i) else None for i in idxs)

        # reduce into a data dict for each interval
        data = [dict(self.FILL_DICT, **self.reducer(e)) for e in target_vals]

        # merge the list of dicts into a dict of lists/arrays
        return {
            key: np.array([np.asarray(d[key]) for d in data])
            for key in set().union(*data)
        }

    def transform(self, jam, query=None):
        results = super().transform(jam, query)
        if self.squeeze:
            results = {
                k: d[0] if d.shape[0] == 1 else d
                for k, d in results.items()
            }
        return results

    def inverse(self, x, duration=None):
        raise NotImplementedError('Lambda annotations cannot be inverted')



############
# Utilities
############

# Mapping of js primitives to numpy types
__TYPE_MAP__ = dict(integer=np.int_,
                    boolean=np.bool_,
                    number=np.float_,
                    object=np.object_,
                    array=np.object_,
                    string=np.str_,
                    null=np.float_)

def _get_dtype(spec):
    '''Get the dtype given a jams namespace schema.

    This handles differently from jams.schema.__get_type in that it:
     - explicitly assigns np.str_ to strings
     - handles "type" as an array of types. if they're not the same base-type assign np.object_
    '''
    if 'type' in spec:
        if isinstance(spec['type'], (list, tuple)):
            # get dtype for each type in list
            types = [jams.schema.__TYPE_MAP__.get(t, np.object_)
                     for t in spec['type']]

            # If they're not all equal, return object
            if all([t == types[0] for t in types]):
                return types[0]
            return np.object_

        return __TYPE_MAP__.get(spec['type'], np.object_)

    if 'enum' in spec:
        # Enums map to objects
        types = [np.dtype(type(v)).type for v in spec['enum']]

        if all([t == types[0] for t in types]):
            return types[0]
        return np.object_

    if 'oneOf' in spec:
        # Recurse
        types = [_get_dtype(v) for v in spec['oneOf']]

        # If they're not all equal, return object
        if all([t == types[0] for t in types]):
            return types[0]

    return np.object_


__FILL_VALUE_MAP__ = [
    (np.floating, np.nan),
    (np.complexfloating, np.nan),
    (np.str_, ''),
]

def fill_value(dtype):
    '''Get a fill-value for a given dtype

    Parameters
    ----------
    dtype : type

    Returns
    -------
    `np.nan` if `dtype` is real or complex
    `''` if string
    `0` otherwise
    '''
    return dtype(next((
        v for dt, v in __FILL_VALUE_MAP__
        if np.issubdtype(dtype, dt)), 0))


def _check_fields(fields, namespace, multi):
    '''
    Validate the fields passed and try to infer fields from incomplete info.

    see `LambdaTransformer.__init__` for argument descriptions.

    Returns
    -------
        fields : list of tuples
            the fully qualified field spec tuples
        value_type : np.dtype
            the dtype of the observation value

    '''
    schema = jams.schema.namespace(namespace)
    value_type, _ = jams.schema.get_dtypes(namespace)

    if isinstance(fields, str):
        fields = [fields]
    else:
        fields = list(fields)

    # get object field dtypes
    try:
        valschema = schema['properties']['value']
        dtypes = {
            name: _get_dtype(spec) for name, spec in
            valschema['properties'].items()
        }
        # NOTE: cannot use value_type because other things map to np.object_
        value_has_defined_keys = (
            valschema['type'] == 'object' and 'properties' in valschema)
    except KeyError:
        value_has_defined_keys = False
        dtypes = {}

    # get default shape
    if multi:
        default_shape = (None, None)
    else:
        default_shape = (None, 1)

    if dtypes:
        # set default fields as all object props
        if not fields:
            fields = [(name, default_shape, dtype)
                      for name, dtype in dtypes.items()]

        # check if any fields were passed as strings and try to infer
        # them from the schema
        else:
            for i, f in enumerate(fields):
                if isinstance(f, str):
                    fields[i] = (f, default_shape, dtypes[f])
    else:
        # if no fields specified and no value object keys, set a single default field
        if not fields:
            fields = [(namespace, default_shape, value_type)]
        # check for fields defined as strings. assume they are all the same type as value.
        else:
            for i, f in enumerate(fields):
                if isinstance(f, str):
                    fields[i] = (f, default_shape, value_type)

    # double check that everything looks how we want it.
    assert fields, 'at least one field must be defined.'

    assert all(isinstance(f, (list, tuple)) and len(f) == 3 for f in fields), (
        'all fields must be a tuple of length 3: (name, shape, dtype)')

    return fields, value_has_defined_keys


def _default_reducer(fields, value_has_defined_keys, multi):
    '''
    Validate the fields passed and try to infer fields from incomplete info.

    see `LambdaTransformer.__init__` for argument descriptions.

    Arguments:
    ----------
    fields : list of tuples/str

    value_type : dtype
        the dtype of observation values

    multi : bool
        whether reduce should accept single or multiple values.

    '''
    fields = [f[0] for f in fields]
    if value_has_defined_keys:
        if multi: # values will be a list of dicts
            def reduce(values):
                return {k: [v[k] for v in values] for k in fields}

        else: # value will be a dict, or None
            def reduce(value):
                if not value:
                    return {}
                return {k: value[k] for k in fields if k in value}

    else: # value will be some unspecified type
        if len(fields) == 1:
            field = fields[0]
            # field = list(fields.keys())[0]
            def reduce(value):
                return {field: value}

        else:
            raise RuntimeError('For multi-field non-object transformers, you must define '
                               '`reduce`, the mapping from list of observation '
                               'values to data dict.')
    return reduce
