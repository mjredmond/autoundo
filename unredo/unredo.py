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

from inspect import signature


class UnRedo(object):
    def __init__(self, name):
        self.name = name

        self._redo_actions = []
        self._undo_actions = []

        self._current_actions = []

        self._macro_on = False

        self._wrappers = []

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

    def _begin_macro(self):
        if self._macro_on:
            return None

        self._macro_on = True
        self._current_actions = []
        return True

    def _end_macro(self, owns_macro):
        if not owns_macro:
            return

        self._condense()
        self._undo_actions.append(self._current_actions)
        self._current_actions = []

        self._macro_on = False

    def _rollback(self, owns_macro):
        if not owns_macro:
            return

        self._end_macro(owns_macro)

        self.undo()

        self._redo_actions.pop()

    def _submit_actions(self, action):
        self._current_actions.append(action)

    def _condense(self):
        pass

    def __call__(self, observables, global_dict=None):

        for observable in observables:
            if 'globals' in observable:
                if global_dict is None:
                    raise ValueError('The global dict must be passed in order to observe global variables!')
                break

        if global_dict is not None:
            global_dict = GlobalDict(global_dict)

        def wrapper(func):

            observables_ = Observables()
            observables_.set_observables(func, observables)

            inner_wrapper = UnRedoWrapper(observables_, global_dict, func, self)

            self._wrappers.append(inner_wrapper)

            return inner_wrapper

        return wrapper

    def disable(self):
        for wrapper in self._wrappers:
            wrapper.disable()

    def enable(self):
        for wrapper in self._wrappers:
            wrapper.enable()


class Observables(object):
    def __init__(self):
        self.observables = None
        self.base_parameters = []
        self.param_indices = []
        self.split_params = []
        self.func_sig = None
        self.global_params = []

    def set_observables(self, func, observables):

        self.observables = observables
        del self.base_parameters[:]
        del self.param_indices[:]
        del self.split_params[:]

        func_sig = signature(func)

        parameters = func_sig.parameters

        param_pos = {}
        param_keyword = set()

        i = 0
        for param in itervalues(parameters):
            param_str = str(param)

            _param_str = param_str.split('=')[0]

            if '=' in param_str:
                param_keyword.add(_param_str)

            param_pos[_param_str] = i

            i += 1

        split_params = self.split_params
        base_parameters = self.base_parameters
        param_indices = self.param_indices

        for observable in observables:
            tmp = observable.split('.')

            split_params.append(tmp)

            base_param = tmp[0]

            base_parameters.append(base_param)

            if base_param == 'globals':
                self.global_params.append(observable)
                continue

            if base_param in param_keyword:
                raise TypeError('Keyword arguments should be immutable and thus not observable! %s' % base_param)

            try:
                param_indices.append(param_pos[base_param])
            except KeyError:
                raise ValueError("Cannot find parameter '%s' in function signature!" % base_param)

    def get_args(self, global_dict, *args, **kwargs):
        args = self.func_sig.bind(*args, **kwargs).arguments

        # return objects that get and set state



def create_actions(initial_values, final_values):
    return None, None


class GlobalDict(object):
    def __init__(self, global_dict):
        self.__dict__ = global_dict


class UnRedoWrapper(object):
    def __init__(self, observables, global_dict, func, unredo):
        self.observables, self.base_parameters, self.param_indices, self.split_observables = observables
        self.global_dict = global_dict
        self.func = func
        self.unredo = unredo
        self._enabled = True

    def __call__(self, *args, **kwargs):
        if not self._enabled:
            return self.func(*args, **kwargs)

        initial_values = self.observables.get_args(self.global_dict, *args, **kwargs)

        owns_macro = self.unredo._begin_macro()

        try:
            return_val = self.func(*args, **kwargs)
        except Exception as e:
            self.unredo._rollback(owns_macro)
            raise e

        assert isinstance(return_val, bool)

        final_values = self.observables.get_args(self.global_dict, *args, **kwargs)

        actions = create_actions(initial_values, final_values)

        self.unredo._submit_actions(actions)

        self.unredo._end_macro(owns_macro)

        return return_val

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True


def check():
    print('hello')


class Tmp(object):
    def __init__(self):
        self.func = check
        self.__call__2 = self.__class__.__call__

    def __call__(self, *args, **kwargs):
        print('hello 2')

    def __disabled_call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def disable(self):
        self.__class__.__call__ = self.__class__.__disabled_call__

    def enable(self):
        self.__class__.__call__ = self.__call__2


a = Tmp()
a()
a.disable()
a()
a.enable()
a()
