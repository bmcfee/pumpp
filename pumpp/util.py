import jams


def match_query(data, query, strict_keys=False):
    '''
    Recursively determine if a datum matches a query.

    data : any
        the data to match the query against

    query : same/similar to data
        a set of tests matching the form of `data`.

        If `query` is callable, it will be called with `data` as an argument.

        If both `data` and `query` are dicts, each key in `query` will be tested against
        each key in `data` recursively. If a key from `query` doesn't exist in `data`,
        it will return False.

        If both `data` and `query` are iterables, they will be compared elementwise.

        If `query` is a set, it will check if `data` is a member of the set.

        If none of those, it will compare using jams.core.match_query which performs
        regex matching if both are strings, or tests for equality otherwise.

    strict_keys : bool [optional]
        Whether to throw an error if ``query`` has keys outside of ``data``.

    Returns:
    --------
    bool representing whether the query matches.

    Raises:
    -------
    ValueError :
        If query is a dict, but data is not.

        If strict_keys is True and query has keys outside data.

        If query is an iterable, but data is not.

        If query and data are both iterables, but not the same length.

    Examples:
    ---------
    assert True  == match_query(5, 5)
    assert True  == match_query(5, lambda x: x > 4)
    assert True  == match_query({'a': 'abc', 'b': 'def'}, {'b': 'def'})
    assert False == match_query({'a': 'abc', 'b': 'def'}, {'b': 'abc'})
    assert True  == match_query({'a': 'abc', 'b': 'def'}, {'b': lambda x: 'd' in x})
    assert False == match_query({'a': 'abc', 'b': 'def'}, {'a': 'abc', 'b': 'deg'})
    assert True  == match_query([1, 2, 3, 4, 5], lambda x: 5 in x)
    assert True  == match_query([1, 2, 3, 4, 5], [1, 2, 3, 4, lambda x: x in (5, 6, 7)])
    assert True  == match_query('def', 'abc|def')
    assert False == match_query({'b': 'def'}, {'b': 'def', 'c': 'ghi'})

    '''

    if query is None:
        return True

    if callable(query):
        return query(data)

    if isinstance(query, dict):
        if isinstance(data, dict):
            if strict_keys and not all(k in data for k in query):
                raise ValueError('`query` has keys outside of `data` and '
                                 '`strict_keys` was set.')

            return all(k in data and match_query(data[k], query[k])
                       for k in query)
        else:
            raise ValueError('`query` was a dict, but `data` was not.')

    if isinstance(query, list):
        if isinstance(data, list):
            if len(data) != len(query):
                raise ValueError('`query` and `data` must be a list of the '
                                 'same length.')

            return all(match_query(d, q) for d, q in zip(data, query))
        else:
            raise ValueError('`query` was an list, but `data` was not.')

    if isinstance(query, set):
        return data in query

    return jams.core.match_query(data, query)
