#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Exception classes for pumpp'''


class PumppError(Exception):
    '''The root pumpp exception class'''
    pass


class DataError(PumppError):
    '''Exceptions relating to data errors'''
    pass


class ParameterError(PumppError):
    '''Exceptions relating to function and method parameters'''
    pass
