from __future__ import print_function, absolute_import

from autoundo import AutoUndo

import numpy as np

undo = AutoUndo('mystack', strict=False)

from example_module import MyVal, f1, some_list


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

undo.undo()

print(a, b, c, d, e, f, some_list)

undo.redo()

print(a, b, c, d, e, f, some_list)
