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

from unredo import UnRedo

import numpy as np

undo_stack = UnRedo('mystack')


some_list = []


class MyVal(object):
    def __init__(self, data):
        self.data = data


@undo_stack(
    ('a.data', 'b', 'c', 'globals.some_list'),
    global_dict=globals()
)
def f1(a, b, c):
    a.data += [1, 2, 3]

    a.data = np.array([-5, -4, -3, -2, -1])

    b.insert(0, 1)
    some_list.extend([6, 7, 8])

    c += 1

    return True


a = MyVal(np.array([-1, -2, -3]))
b = []
c = np.array([0, 0, 0, 0, 0])

print(a.data, b, c, some_list)

f1(a, b, c)

print(a.data, b, c, some_list)

undo_stack.undo()

print(a.data, b, c, some_list)

undo_stack.redo()

print(a.data, b, c, some_list)

