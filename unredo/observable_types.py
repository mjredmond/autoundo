"""
BSD 3-Clause License

Copyright (c) 2017, mjredmond
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
from __future__ import print_function, absolute_import
from six import iteritems, iterkeys, itervalues
from six.moves import range

from inspect import getmro
from collections import OrderedDict

from copy import copy
import numpy as np
from zlib import compress, decompress


_observables = {}

immutables = frozenset([int, float, str, bytes, frozenset])


def register_observable(cls):
    for type_ in cls.types:
        _observables[type_] = cls

    return cls


def get_observable(base_obj, observable_ids):

    if len(observable_ids) == 1:
        try:
            return _observables[type(base_obj)](base_obj)
        except KeyError:
            raise TypeError('Cannot create observable for type %s.  You need to provide an implementation of AbstractObservable.' % str(type(base_obj)))

    obj = get_obj(base_obj, observable_ids)

    if type(obj) in immutables:


    try:
        _observable_type = _observables[type(obj)]
    except KeyError as e:
        mro = set(getmro(type(base_obj)))
        match = mro.union(set(_observables))

        if len(match) == 0:
            raise e

        _observable_type = _observables[match.pop()]

    _observable = _observable_type(obj)

    if len(observable_ids) == 1:
        return _observable
    else:



def get_obj(base_obj, observable_ids):
    obj = base_obj

    for _id in observable_ids[1:]:
        obj = getattr(obj, _id)

    return obj


class AbstractObservable(object):
    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError


class MutableObservable(AbstractObservable):

    types = ()

    def __init__(self, obj):
        assert isinstance(obj, self.types)
        self.obj = obj


class Undefined(object):
    pass


class ImmutableObservable(MutableObservable):
    def __init__(self, base_obj, observable_ids):
        super(ImmutableObservable, self).__init__(base_obj)
        self.base_obj = base_obj
        self.observable_ids = observable_ids

    def get_state(self):
        try:
            get_obj(self.base_obj, self.observable_ids)
        except AttributeError:
            return Undefined

    def set_state(self, state):
        try:
            obj = get_obj(self.base_obj, self.observable_ids[:-1])
        except AttributeError:
            raise RuntimeError('Cannot find base object for immutable in %s!' % '.'.join(self.observable_ids))

        if state is Undefined:
            delattr(obj, self.observable_ids[-1])
        else:
            setattr(obj, self.observable_ids[-1], state)


@register_observable
class ListObservable(MutableObservable):

    types = (list,)

    def get_state(self):
        return list(self.obj)

    def set_state(self, state):
        del self.obj[:]
        self.obj.extend(state)


@register_observable
class DictSetObservable(MutableObservable):

    types = (dict, set, OrderedDict)

    def get_state(self):
        return copy(self.obj)

    def set_state(self, state):
        self.obj.clear()
        self.obj.update(state)


@register_observable
class NumpyObservable(MutableObservable):

    types = (np.ndarray,)

    def get_state(self):
        return compress(self.obj.tobytes(), level=9), self.obj.dtype, self.obj.shape

    def set_state(self, state):
        arr = np.frombuffer(decompress(state[0]), dtype=state[1]).astype(dtype=state[1].reshape(state[2]))
        self.obj.resize(state[2])
        np.copyto(self.obj, arr)
