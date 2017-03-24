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

from unredo import UnRedo, register_observable, MutableObservable, get_observable

import numpy as np

undo_stack = UnRedo('mystack')


some_list = [9, 10, 11]


class MyVal(object):
    def __init__(self):
        self.data = np.array([0, 1, 2])
        self.x = 0
        self.y = 1
        self.z = 2

    def __repr__(self):
        return '%s, %s, %s, %s' % (str(self.data), str(self.x), str(self.y), str(self.z))


@register_observable
class MyValObservable(MutableObservable):
    types = (MyVal,)

    def __init__(self, obj):
        super(MyValObservable, self).__init__(obj)
        self._observables.append(get_observable(obj.data))

    def get_state(self):
        return self.obj.data, self.obj.x, self.obj.y, self.obj.z

    def set_state(self, state):
        self.obj.data = state[0]
        self.obj.x = state[1]
        self.obj.y = state[2]
        self.obj.z = state[3]


@undo_stack(
    ('a', 'b', 'c', 'd', 'e', 'f', 'globals.some_list'),
    global_dict=globals()
)
def f1(a, b, c, d, e, f):
    a.data += [1, 2, 3]

    a.data = [5, 6, 7, 8]

    a.x = -1
    a.y = -2
    a.z = -3

    # a.data = np.array([-5, -4, -3, -2, -1])

    b.insert(0, 1)
    some_list.extend([6, 7, 8])

    c.remove(3)
    c.add(9)

    del d['a']
    d['f'] = 5
    d['b'] = -10

    e[0].extend([-1, -2, -3])
    e[-1].pop()

    f_a = f['a']
    f_a[0] = -1
    f_a[4].pop()
    f_a[5].add(-1)

    f_b = f['b']
    f_b['a'].extend([4, 5, 6])
    f_b['b']['a'][0] = -1
    f_b['b']['a'][3].extend([11, 12, 13])
    f_b['b']['b'] = -1
    del f['a']


a = MyVal()
b = []
c = {0, 1, 2, 3, 4}
d = {
    'a': 1,
    'b': 2,
    'c': 3
}
e = [[0, 1, 2, 3], 3, 5, 6, [4, 5, 6]]
f = {
    'a': [0, 1, 2, 3, [4, 5, 6], {7, 8, 9, 10}],
    'b': {
        'a': [1, 2, 3],
        'b': {
            'a': [4, 5, 6, [7, 8, 9, 10]],
            'b': 1
        }
    }
}

print(a, b, c, d, e, f, some_list)

f1(a, b, c, d, e, f)

print(a, b, c, d, e, f, some_list)

undo_stack.undo()

print(a, b, c, d, e, f, some_list)

undo_stack.redo()

print(a, b, c, d, e, f, some_list)
