#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import namedtuple

__all__ = ['Tensor']

Tensor = namedtuple('Tensor', ['shape', 'dtype'])
