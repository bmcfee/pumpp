#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import namedtuple

__all__ = ['Tensor', 'Scope']

Tensor = namedtuple('Tensor', ['shape', 'dtype'])


class Scope(object):

    def __init__(self, name):

        self.name = name
        self.fields = dict()

    def scope(self, key):

        if self.name is None:
            return key
        return '{:s}/{:s}'.format(self.name, key)

    def register(self, field, shape, dtype):
        # TODO: validate shape and dtype here
        self.fields[self.scope(field)] = Tensor(tuple(shape), dtype)

