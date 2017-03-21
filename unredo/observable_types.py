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
from array import array
from zlib import compress, decompress


_observables = {}

immutables = frozenset([int, float, str, bytes, frozenset])

_counter = 0


def reset_counter():
    global _counter
    _counter = 0


def _increment_counter():
    global _counter
    _counter += 1


def register_observable(cls):
    for type_ in cls.types:
        if type_ in _observables:
            raise ValueError('Observable for type %s has already been registered!' % str(type_))
        _observables[type_] = cls

    return cls


# helper functions

def get_observable(base_obj, observable_ids):

    if len(observable_ids) == 1:
        try:
            return _observables[type(base_obj)](base_obj)
        except KeyError:
            raise TypeError(
                'Cannot create observable for type %s.  '
                'You need to provide an implementation of AbstractObservable.' % str(type(base_obj))
            )

    objs = get_all_objs(base_obj, observable_ids)

    return AttributeObservable(objs[0], observable_ids)


def get_obj(base_obj, observable_ids):
    obj = base_obj

    for _id in observable_ids[1:]:
        obj = getattr(obj, _id)

    return obj


def get_all_objs(base_obj, observable_ids):
    objs = [base_obj]

    obj = base_obj

    for _id in observable_ids[1:]:
        obj = getattr(obj, _id)
        objs.append(obj)

    return objs


# observable classes

class AbstractObservable(object):
    def __init__(self):
        global _counter
        self.counter = _counter
        _increment_counter()

        self._observable_id = None

    @property
    def observable_id(self):
        return self._observable_id

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False

        if self.get_state() != other.get_state():
            return False

        return True


class MutableObservable(AbstractObservable):

    types = ()

    def __init__(self, obj):
        super(MutableObservable, self).__init__()

        assert isinstance(obj, self.types)
        self.obj = obj

        self.initial_state = self.get_state()

        self._observable_id = id(self.obj)

    def restore(self):
        self.set_state(self.initial_state)


class ImmutableObserver(AbstractObservable):
    def __init__(self, value):
        super(ImmutableObserver, self).__init__()

        self.value = value
        self.initial_state = value

    def get_state(self):
        return self.value

    def set_state(self, state):
        self.value = state

    def restore(self):
        self.value = self.initial_state


class Undefined(object):
    pass


class AttributeObservable(AbstractObservable):
    def __init__(self, base_obj, observable_ids):
        super(AttributeObservable, self).__init__()

        self.base_obj = base_obj
        self.observable_ids = observable_ids

        try:
            obj = type(get_obj(self.base_obj, self.observable_ids))
        except AttributeError:
            obj = Undefined

        if obj in immutables:
            self._observable = None
        else:
            obj = get_obj(self.base_obj, self.observable_ids)
            self._observable = get_observable(obj, self.observable_ids[-1:])
            self.counter = self._observable.counter

        self.initial_state = self.get_state()

        self._observable_id = (id(self.base_obj), '.'.join(self.observable_ids))

    def restore(self):
        self.set_state(self.initial_state)

        try:
            self._observable.restore()
        except AttributeError:
            pass

    def get_state(self):
        try:
            obj = get_obj(self.base_obj, self.observable_ids)
        except AttributeError:
            obj = Undefined

        try:
            obj_state = self._observable.get_state()
        except AttributeError:
            obj_state = Undefined

        return obj, obj_state

    def set_state(self, state):
        obj, obj_state = state

        if obj_state is not Undefined:
            self._observable.set_state(obj_state)

        try:
            obj_ = get_obj(self.base_obj, self.observable_ids[:-1])
        except AttributeError:
            raise RuntimeError('Cannot find base object for immutable in %s!' % '.'.join(self.observable_ids))

        if obj is Undefined:
            delattr(obj_, self.observable_ids[-1])
        else:
            setattr(obj_, self.observable_ids[-1], obj)


# concrete implementations below

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
        return compress(self.obj.tobytes()), self.obj.dtype, self.obj.shape

    def set_state(self, state):
        arr = np.frombuffer(decompress(state[0]), dtype=state[1]).astype(dtype=state[1]).reshape(state[2])
        self.obj.resize(state[2])
        np.copyto(self.obj, arr)


@register_observable
class PyArrayObservable(MutableObservable):

    types = (array,)

    def get_state(self):
        return compress(self.obj.tobytes())

    def set_state(self, state):
        del self.obj[:]
        self.obj.frombytes(decompress(state))
