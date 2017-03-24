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

from copy import copy, deepcopy
import numpy as np
from array import array
from zlib import compress, decompress


_observable_types = {}

immutable_types = frozenset([int, float, str, bytes, frozenset])
simple_types = {int, float, str, bytes}
mutable_container_types = {list, dict, set, OrderedDict}

_counter = 0


def reset_counter():
    global _counter
    _counter = 0


def _increment_counter():
    global _counter
    _counter += 1


def register_observable(cls):
    for type_ in cls.types:
        if type_ in _observable_types:
            raise ValueError('Observable for type %s has already been registered!' % str(type_))
        _observable_types[type_] = cls
    return cls


def override_observable(cls):
    for type_ in cls.types:
        _observable_types[type_] = cls
    return cls


# helper functions


def get_observable(base_obj, observable_ids=('N/A',)):

    if len(observable_ids) == 1:
        try:
            return _observable_types[type(base_obj)](base_obj)
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
        try:
            obj = getattr(obj, _id)
        except AttributeError:
            return Undefined

    return obj


def get_all_objs(base_obj, observable_ids):
    objs = [base_obj]

    obj = base_obj

    for _id in observable_ids[1:]:
        try:
            obj = getattr(obj, _id)
        except AttributeError:
            obj = Undefined
        objs.append(obj)

    return objs


# observable classes

class AbstractObservable(object):

    types = ()

    def __init__(self):
        global _counter
        self.counter = _counter
        _increment_counter()

        self.initial_state = None

        self._observable_id = None

        self._observables = []

    @property
    def observable_id(self):
        return self._observable_id

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def restore(self):
        self.set_state(self.initial_state)

        for observable in self._observables:
            observable.restore()

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False

        if self.initial_state != other.initial_state:
            return False

        if self._observables != other._observables:
            return False

        return True


class MutableObservable(AbstractObservable):
    def __init__(self, obj):
        super(MutableObservable, self).__init__()

        assert isinstance(obj, self.types)
        self.obj = obj

        self.initial_state = self.get_state()

        self._observable_id = id(self.obj)


class Undefined(object):
    pass


class AttributeObservable(AbstractObservable):
    def __init__(self, base_obj, observable_ids):
        super(AttributeObservable, self).__init__()

        self.base_obj = base_obj
        self.observable_ids = observable_ids

        try:
            obj_type = type(get_obj(self.base_obj, self.observable_ids))
        except AttributeError:
            obj_type = Undefined

        if obj_type not in immutable_types:
            obj = get_obj(self.base_obj, self.observable_ids)
            self._observables.append(get_observable(obj, self.observable_ids[-1:]))
            self.counter = self._observables[0].counter

        self.initial_state = self.get_state()

        self._observable_id = (id(self.base_obj), '.'.join(self.observable_ids))

    def get_state(self):
        try:
            obj = get_obj(self.base_obj, self.observable_ids)
        except AttributeError:
            obj = Undefined

        try:
            obj_state = self._observables[0].get_state()
        except AttributeError:
            obj_state = Undefined

        return obj, obj_state

    def set_state(self, state):
        obj, obj_state = state

        if obj_state is not Undefined:
            self._observables[0].set_state(obj_state)

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

    def __init__(self, obj):
        super(ListObservable, self).__init__(obj)

        for _obj in self.obj:
            _obj_type = type(_obj)
            if _obj_type not in simple_types:
                self._observables.append(get_observable(_obj))

    def restore(self):
        super(ListObservable, self).restore()

        for obj in self._observables:
            obj.restore()

    def get_state(self):
        return list(self.obj)

    def set_state(self, state):
        del self.obj[:]
        self.obj.extend(state)


class DictSetObservable(MutableObservable):

    def get_state(self):
        return copy(self.obj)

    def set_state(self, state):
        self.obj.clear()
        self.obj.update(state)


@register_observable
class DictObservable(DictSetObservable):

    types = (dict, OrderedDict)

    def __init__(self, obj):
        super(DictObservable, self).__init__(obj)

        for _key, _obj in iteritems(self.obj):
            _obj_type = type(_obj)
            if _obj_type not in simple_types:
                self._observables.append(get_observable(_obj))

            _key_type = type(_key)
            if _key_type not in simple_types:
                self._observables.append(get_observable(_key))

    def restore(self):
        super(DictObservable, self).restore()

        for obj in self._observables:
            obj.restore()


@register_observable
class SetObservable(DictSetObservable):
    types = (set,)

    def __init__(self, obj):
        super(SetObservable, self).__init__(obj)

        for _obj in self.obj:
            _obj_type = type(_obj)
            if _obj_type not in simple_types:
                self._observables.append(get_observable(_obj))

    def restore(self):
        super(SetObservable, self).restore()

        for obj in self._observables:
            obj.restore()


# @register_observable
class NumpyObservable(MutableObservable):

    types = (np.ndarray,)

    def get_state(self):
        return compress(self.obj.tobytes()), self.obj.dtype, self.obj.shape

    def set_state(self, state):
        arr = np.frombuffer(decompress(state[0]), dtype=state[1]).astype(dtype=state[1]).reshape(state[2])
        self.obj.resize(state[2])
        np.copyto(self.obj, arr)


# attempt to compensate for object dtypes
@register_observable
class NumpyObservable2(MutableObservable):

    types = (np.ndarray,)

    def get_state(self):
        if self.obj.dtype.names is None:
            return 0, compress(self.obj.tobytes()), self.obj.dtype, self.obj.shape
        else:
            data = {}

            for name in self.obj.dtype.names:
                arr = self.obj[name]
                if arr.dtype == 'object':
                    data[name] = arr.tolist()
                else:
                    data[name] = compress(arr.tobytes()), self.obj[name].dtype, self.obj[name].shape

            return 1, data, self.obj.dtype, self.obj.shape

    def set_state(self, state):
        if state[0] == 0:
            arr = np.frombuffer(decompress(state[1]), dtype=state[2]).astype(dtype=state[2]).reshape(state[3])
            self.obj.resize(state[3])
            np.copyto(self.obj, arr)

        elif state[0] == 1:
            data = state[1]
            dtype = state[2]
            shape = state[3]
            arr = np.empty(shape=shape, dtype=dtype)
            for name, _data in iteritems(data):
                if isinstance(_data, list):
                    arr[name] = _data
                else:
                    _dtype = _data[1]
                    _shape = _data[2]
                    _data = _data[0]
                    arr[name] = np.frombuffer(decompress(_data), dtype=_dtype).astype(dtype=_dtype).reshape(_shape)
            self.obj.resize(shape)
            np.copyto(self.obj, arr)

        else:
            raise ValueError


@register_observable
class PyArrayObservable(MutableObservable):

    types = (array,)

    def get_state(self):
        return compress(self.obj.tobytes())

    def set_state(self, state):
        del self.obj[:]
        self.obj.frombytes(decompress(state))
