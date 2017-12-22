import builtins
from array import array
from collections import OrderedDict
from copy import copy
from inspect import getmembers, getframeinfo, stack, getmodule
from os import path
from pathlib import Path
from weakref import WeakValueDictionary, WeakKeyDictionary
from zlib import compress, decompress

# from .data import AutoUndoData
# from .func_wrapper import func_wrapper
# from .observables import Observables
# from .action import AutoUndoAction
########################################################################################################################

# what is _counter for?  need to document this... is there a better way?

_counter = 0


def reset_counter():
    global _counter
    _counter = 0


def _increment_counter():
    global _counter
    _counter += 1


class AutoUndoTypeError(TypeError):
    pass


class AutoUndoData(object):
    autoundo_types = {}
    _autoundo_types = {}
    override = False

    @staticmethod
    def reset_counter():
        reset_counter()

    @classmethod
    def add(cls, kls, override=False):
        autoundo_types = kls.autoundo_types

        for _type in autoundo_types:
            if _type in cls._autoundo_types and override is False and kls.override is False:
                raise RuntimeError('Cannot override AutoUndo type! %s' % str(_type))

            cls._autoundo_types[_type] = kls

        return kls

    @classmethod
    def create(cls, obj, recurse=True, strict=True):
        try:
            return obj._autoundo_data(obj)
        except AttributeError:
            pass

        try:
            obj_type = obj._autoundo_type
        except AttributeError:
            obj_type = type(obj)

        try:
            return cls._autoundo_types[obj_type](obj, recurse, strict)
        except KeyError:
            if strict:
                raise

            try:
                return AutoUndoGenericClass(obj, recurse, strict)
            except AutoUndoTypeError:
                pass

            raise

    def __init__(self, obj, recurse=True, strict=True):
        self.observable_id = id(self)
        self.obj = obj
        self.initial_state = self.get_state()
        self.recurse = recurse
        self.strict = strict
        self.subdata = []

        if recurse is True:
            self._get_subdata()

        global _counter
        self.counter = _counter
        _increment_counter()

    def _get_subdata(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def restore(self):
        self.set_state(self.initial_state)

        if self.subdata is None:
            return

        for data in self.subdata:
            data.restore()

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False

        if self.initial_state != other.initial_state:
            return False

        if self.subdata != other.subdata:
            return False

        return True

        # def __hash__(self):
        #     return hash('%s.%d' % (self.__class__.__name__, self.observable_id))


@AutoUndoData.add
class AutoUndoImmutable(AutoUndoData):
    autoundo_types = {int, float, complex, str, bytes, bool, type(None)}

    def restore(self):
        pass

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def _get_subdata(self):
        pass

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False

        return self.obj == other.obj


@AutoUndoData.add
class AutoUndoList(AutoUndoData):
    autoundo_types = {list, }

    def get_state(self):
        return copy(self.obj)

    def set_state(self, state):
        del self.obj[:]
        self.obj.extend(state)

    def _get_subdata(self):
        [self.subdata.append(AutoUndoData.create(obj, strict=self.strict)) for obj in self.initial_state]

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False

        if self.initial_state != other.initial_state:
            return False

        if self.subdata != other.subdata:
            return False

        return True


@AutoUndoData.add
class AutoUndoByteArray(AutoUndoData):
    autoundo_types = {bytearray, }

    def get_state(self):
        return copy(self.obj)

    def set_state(self, state):
        del self.obj[:]
        self.obj.extend(state)

    def _get_subdata(self):
        pass

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False

        if self.initial_state != other.initial_state:
            return False

        if self.subdata != other.subdata:
            return False

        return True


@AutoUndoData.add
class AutoUndoSet(AutoUndoData):
    autoundo_types = {set, }

    def get_state(self):
        return copy(self.obj)

    def set_state(self, state):
        self.obj.clear()
        self.obj.update(state)

    def _get_subdata(self):
        [self.subdata.append(AutoUndoData.create(obj, strict=self.strict)) for obj in self.obj]


@AutoUndoData.add
class AutoUndoTuple(AutoUndoData):
    autoundo_types = {tuple, }

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def _get_subdata(self):
        [self.subdata.append(obj) for obj in self.obj]


@AutoUndoData.add
class AutoUndoFrozenSet(AutoUndoData):
    autoundo_types = {frozenset, }

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def _get_subdata(self):
        [self.subdata.append(AutoUndoData.create(obj, strict=self.strict)) for obj in self.obj]


@AutoUndoData.add
class AutoUndoDict(AutoUndoData):
    autoundo_types = {dict, OrderedDict, WeakKeyDictionary, WeakValueDictionary}

    def get_state(self):
        return copy(self.obj)

    def set_state(self, state):
        self.obj.clear()
        self.obj.update(state)

    def _get_subdata(self):
        for _key, _obj in iteritems(self.obj):
            self.subdata.append(AutoUndoData.create(_key, strict=self.strict))
            self.subdata.append(AutoUndoData.create(_obj, strict=self.strict))


@AutoUndoData.add
class AutoUndoPyArray(AutoUndoData):
    autoundo_types = {array, }

    def get_state(self):
        return compress(self.obj.tobytes())

    def set_state(self, state):
        del self.obj[:]
        self.obj.frombytes(decompress(self.initial_state))

    def _get_subdata(self):
        pass


try:
    import numpy as np


    @AutoUndoData.add
    class AutoUndoNumpy(AutoUndoData):
        autoundo_types = {np.ndarray, }

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
                self.obj.resize(state[3], refcheck=False)
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

        def _get_subdata(self):
            if self.recurse is True and self.initial_state[0] == 1:
                data = self.initial_state[1]
                for obj in itervalues(data):
                    if isinstance(obj, list):
                        for _ in obj:
                            self.subdata.append(AutoUndoData.create(_, strict=self.strict))

        def __eq__(self, other):
            if self.__class__ is not other.__class__:
                return False

            arr1 = self.initial_state[0]
            arr2 = other.initial_state[0]

            if not np.all(arr1 == arr2):
                return False

            if self.initial_state[1:] != other.initial_state[1:]:
                return False

            if self.subdata != other.subdata:
                return False

            return True

except ImportError:
    pass


# try:
#     import pandas as pd
#
#     @AutoUndoData.add
#     class AutoUndoPandasDataFrame(AutoUndoData):
#         autoundo_types = {pd.DataFrame, }
#
#         def get_state(self):
#             return None
#
#         def set_state(self, state):
#             pass
#
#         def _get_subdata(self):
#             self.subdata.append(AutoUndoData.create(self.obj.data, strict=self.strict))
#
# except ImportError:
#     pass


@AutoUndoData.add
class AutoUndoGenericClass(AutoUndoData):
    autoundo_types = {}

    def __init__(self, obj, recurse=True, strict=True):
        if not hasattr(obj, '__dict__'):
            raise AutoUndoTypeError("object doesn't have __dict__ attribute.")
        super(AutoUndoGenericClass, self).__init__(obj, recurse, strict)

    def _get_subdata(self):
        for obj in itervalues(self.initial_state):
            self.subdata.append(AutoUndoData.create(obj, strict=self.strict))

    def get_state(self):
        state = {}
        for key, obj in iteritems(self.obj.__dict__):
            if not key.startswith('_'):
                state[key] = obj
        return state

    def set_state(self, state):
        del_keys = set()
        for key, obj in iteritems(self.obj.__dict__):
            if not key.startswith('_'):
                del_keys.add(key)
        for key in del_keys:
            del self.obj.__dict__[key]
        self.obj.__dict__.update(state)

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False

        return self.initial_state == other.initial_state


########################################################################################################################


class AutoUndoAction(object):
    @classmethod
    def create(cls, initial_args, final_args):

        _initial_args = []
        _final_args = []

        assert len(initial_args) == len(final_args)

        for i in range(len(initial_args)):
            if initial_args[i] != final_args[i]:
                _initial_args.append(initial_args[i])
                _final_args.append(final_args[i])

        return cls(_initial_args, _final_args)

    def __init__(self, initial_args, final_args):
        self.initial_args = initial_args
        self.final_args = final_args

    def redo(self):
        for arg in self.final_args:
            arg.restore()

    def undo(self):
        for arg in self.initial_args:
            arg.restore()

    def __len__(self):
        return len(self.initial_args)


########################################################################################################################


def func_wrapper(observables, global_dict, func, unredo):
    """It returns the wrapper that is returned from the decorator.  It accesses
    UnRedo private methods.
    """

    observables = observables
    global_dict = global_dict
    func = func
    unredo = unredo

    def inner_wrapper(*args, **kwargs):
        if not unredo._enabled:
            return func(*args, **kwargs)

        initial_args = observables.get_args(global_dict, unredo.strict, *args, **kwargs)

        owns_macro = unredo._begin_macro()

        try:
            return_val = func(*args, **kwargs)
        except Exception as e:
            unredo._rollback(owns_macro)
            raise e

        final_args = observables.get_args(global_dict, unredo.strict, *args, **kwargs)

        actions = AutoUndoAction.create(initial_args, final_args)

        if len(actions) > 0:
            unredo._submit_actions(actions)

        unredo._end_macro(owns_macro)

        return return_val

    return inner_wrapper


########################################################################################################################

from six import iteritems, itervalues
from six.moves import range

from inspect import signature
from functools import wraps


class Observables(object):
    def __init__(self, strict=True):
        self.observables = None
        self.base_parameters = []
        self.param_indices = []
        self.split_params = []
        self.func_sig = None
        self.global_params = []

        self.strict = strict

    def set_observables(self, func, args=None, global_args=None):
        del self.base_parameters[:]
        del self.param_indices[:]
        del self.split_params[:]

        self.func_sig = signature(func)

        func_sig, param_pos, param_keyword = _get_func_params(func)

        self.func_sig = func_sig

        if args is None:
            args = list(param_pos)

        self.observables = args

        # for key in param_keyword:
        #     observables.remove(key)

        param_pos = {param_pos[i]: i for i in range(len(param_pos))}

        split_params = self.split_params
        base_parameters = self.base_parameters
        param_indices = self.param_indices

        if global_args is not None:
            for arg in global_args:
                self.global_params.append('globals.%s' % arg)

        for observable in args:
            tmp = observable.split('.')

            split_params.append(tmp)

            base_param = tmp[0]

            base_parameters.append(base_param)

            if base_param == 'globals':
                self.global_params.append(observable)
                continue

            # if base_param in param_keyword:
            #     raise TypeError('Keyword arguments should be immutable and thus not observable! %s' % base_param)

            try:
                param_indices.append(param_pos[base_param.replace('~', '')])
            except KeyError:
                raise ValueError("Cannot find parameter '%s' in function signature!" % base_param)

    def get_args(self, global_dict, strict, *args, **kwargs):
        args = self.func_sig.bind(*args, **kwargs).arguments

        _observables = []

        for i in range(len(self.observables)):
            base_param = self.base_parameters[i]

            if base_param == 'globals':
                continue

            split_params = list(self.split_params[i])

            # FIXME: fix recurse

            if split_params[-1].endswith('~'):
                split_params[-1] = split_params[-1][:-1]
                recurse = True
            else:
                recurse = True

            observable = AutoUndoData.create(args[base_param.replace('~', '')], recurse=recurse, strict=strict)

            _observables.append(observable)

        for param in self.global_params:
            obj_name = param.split('.')[1]
            obj = global_dict[obj_name]
            observable = AutoUndoData.create(obj, recurse=True, strict=strict)

            _observables.append(observable)

        return _observables


def _get_func_params(func):
    func_sig = signature(func)

    parameters = func_sig.parameters

    param_pos = []
    param_keyword = set()

    for param in itervalues(parameters):
        param_str = str(param)

        _param_str = param_str.split('=')[0]

        if '=' in param_str:
            param_keyword.add(_param_str)

        param_pos.append(_param_str)

    return func_sig, param_pos, param_keyword


########################################################################################################################


__autoundo_skip__ = True


class AutoUndoWrapper(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, obj):
        obj._autoundo_args = self.args
        return obj


class AutoUndoGlobals(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, obj):
        obj._autoundo_globals = self.args
        return obj


def AutoUndoArgs(*args):
    def inner(obj):
        frm = stack()[1]
        mod = getmodule(frm[0])
        wrapper = AutoUndoWrapper(*args)
        wrapper.obj = obj
        setattr(mod, '_autoundo_wrapper_%s' % obj.__name__, wrapper)
        return obj

    return inner


class AutoUndoObject(object):
    def __init__(self, obj):
        self.obj = obj

    def __call__(self, *args, **kwargs):
        return self.obj(*args, **kwargs)


class AutoUndo(object):
    roots = set()

    def __init__(self, name, limit=100, strict=True):

        self.caller = getframeinfo(stack()[1][0])
        self.root = Path(path.dirname(self.caller.filename))

        if self.root.__repr__() in self.roots:
            raise RuntimeError('Cannot have multiple AutoUndo instances in same root!')

        self.roots.add(self.root.__repr__())

        self.old_imp = builtins.__import__
        builtins.__import__ = self._import

        ############################################

        self.name = name

        self._redo_actions = []
        self._undo_actions = []

        self._current_actions = []

        self._macro_on = False

        self._wrappers = []

        self._limit = limit

        self._update_with_no_changes = True

        self._enabled = True

        self.strict = strict

    def set_update_method(self, val):
        assert isinstance(val, bool)
        self._update_with_no_changes = val

    def set_limit(self, limit):
        self._limit = limit

    def undo(self):
        try:
            actions = self._undo_actions.pop()
        except IndexError:
            return

        for action in actions:
            action.undo()

        self._redo_actions.append(actions)

    def redo(self):
        try:
            actions = self._redo_actions.pop()
        except IndexError:
            return

        for action in actions:
            action.redo()

        self._undo_actions.append(actions)

    def disable(self):
        self._enabled = False
        # for wrapper in self._wrappers:
        #     wrapper.disable()

    def enable(self):
        self._enabled = True
        # for wrapper in self._wrappers:
        #     wrapper.enable()

    def _import(self, *args, **kwargs):
        m = self.old_imp(*args, **kwargs)

        try:
            autoundo_skip = getattr(m, '__autoundo_skip__')
        except AttributeError:
            autoundo_skip = False

        if autoundo_skip:
            return m

        try:
            parents = Path(m.__file__).parents
        except AttributeError:
            parents = {}

        if self.root in parents:
            _process_import(self, m)

        return m

    def _begin_macro(self):
        if self._macro_on:
            return None

        AutoUndoData.reset_counter()
        self._macro_on = True
        self._current_actions = []
        return True

    def _end_macro(self, owns_macro):
        if not owns_macro:
            return False

        self._condense()

        if self._update_with_no_changes is False and len(self._current_actions) == 0:
            return False

        self._undo_actions.append(self._current_actions)
        del self._redo_actions[:]
        self._current_actions = []

        if len(self._undo_actions) > self._limit:
            self._undo_actions.pop(0)

        self._macro_on = False

        return True

    def _rollback(self, owns_macro):
        if not owns_macro:
            return

        if self._end_macro(owns_macro):
            self.undo()

        self._redo_actions.pop()

    def _submit_actions(self, action):
        self._current_actions.append(action)

    def _condense(self):
        _current_actions = self._current_actions

        _initial_actions = {}
        _final_actions = {}

        for action in _current_actions:
            for arg in action.initial_args:
                try:
                    _initial_actions[arg.observable_id].append(arg)
                except KeyError:
                    _initial_actions[arg.observable_id] = [arg]

            for arg in action.final_args:
                try:
                    _final_actions[arg.observable_id].append(arg)
                except KeyError:
                    _final_actions[arg.observable_id] = [arg]

        _keys = list(_initial_actions.keys())

        _initial_args = []

        for key in _keys:
            _initial_args.append(sorted(_initial_actions[key], key=lambda a: a.counter)[0])

        _keys = list(_final_actions.keys())

        _final_args = []

        for key in _keys:
            _final_args.append(sorted(_final_actions[key], key=lambda a: a.counter)[-1])

        self._current_actions = [AutoUndoAction(_initial_args, _final_args)]


def _process_import(self, m):
    members = getmembers(m)

    for member in members:
        if member[0].startswith('_'):
            continue

        if type(member[1]) is type and issubclass(member[1], AutoUndoData):
            AutoUndoData.add(member[1])
            continue

        try:
            args = member[1]._autoundo_args
        except AttributeError:
            args = None

        try:
            global_args = member[1]._autoundo_globals
        except AttributeError:
            global_args = None

        if isinstance(member[1], AutoUndoWrapper):
            continue

        try:
            if member[1].__module__ == m.__name__:
                _process_object(self, m, member[0], member[1], args, global_args)
        except AttributeError:
            pass


def _process_object(self, module, object_name, object, args=None, global_args=None):
    assert object.__module__ == module.__name__

    if type(object) is type:
        _process_class(self, module, object, args, global_args)
        return
    elif callable(object):
        _process_func(self, module, object_name, object, args, global_args)
        return


def _process_func(self, module, name, func, args=None, global_args=None):
    setattr(module, name, _wrap_func(self, module, func, args, global_args))


def _process_class(self, module, kls, kls_members=None, global_args=None):
    if kls_members is not None:
        members = ((member, getattr(kls, member)) for member in kls_members)
    else:
        members = getmembers(kls)

    for member in members:
        member_id = member[0]
        member = member[1]

        if member_id.startswith('_'):
            continue

        try:
            args = member._autoundo_args
        except AttributeError:
            args = None

        try:
            global_args = member._autoundo_globals
        except AttributeError:
            global_args = None

        setattr(kls, member_id, _wrap_func(self, module, member, args, global_args))


def _wrap_func(self, module, func, args=None, global_args=None):
    observables_ = Observables()
    observables_.set_observables(func, args, global_args)

    global_dict = module.__dict__

    inner_wrapper = wraps(func)(func_wrapper(observables_, global_dict, func, self))

    self._wrappers.append(inner_wrapper)

    return inner_wrapper
