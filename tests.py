from typing import Optional, ClassVar, Iterator, Iterable, Any, Type, TypeVar, cast
import math
import bisect
import collections
import copy
import itertools

from groups import *

def verify_bijection(G: Type[Group]):
	for i, (a, b) in enumerate(zip(map(G._fromindex, range(len(G))), G._enumerate(), strict=True)):
		assert a == b
		assert int(a) == i
		assert not (a < a or a > a)
		assert a <= a and a >= a
	for a, b in itertools.pairwise(G):
		assert a < b and a <= b
		assert b > a and b >= a
		assert not (b < a or b <= a or a >= b or a > b)

for size in [5]:
	class Sn(SymmetricGroup):
		SIZE = size
	assert Sn.ID.value == tuple(range(size))
	verify_bijection(Sn)
	for a in Sn:
		assert a == Sn(a.value)
		assert a == Sn.from_cycles(*a.cycles())
		for k in range(-5, 6):
			assert Group._pow(a, k) == Sn._pow(a, k)
