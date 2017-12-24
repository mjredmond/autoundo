from __future__ import print_function, absolute_import
from six import iteritems, iterkeys, itervalues
from six.moves import range

from autoundo import AutoUndoGlobals

import numpy as np


some_list = [9, 10, 11]


class MyVal(object):
    def __init__(self):
        self.data = np.array([0, 1, 2])
        self.x = 0
        self.y = 1
        self.z = 2

    def __repr__(self):
        return '%s, %s, %s, %s' % (str(self.data), str(self.x), str(self.y), str(self.z))


@AutoUndoGlobals('some_list')
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


