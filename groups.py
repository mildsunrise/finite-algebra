'''
This code is NOT safe for cryptography; operation time may depend on the values.

Flexibility first, readability second, performance third (but we generally care about O(n) complexity).
'''

from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, ClassVar, Self, Iterator, Any
import math
import bisect
import collections
import copy

class classproperty(property):
	def __get__(self, owner_self, owner_cls):
		return self.fget(owner_cls)

class Group(ABC):
	'''
	Base class for finitely generated (finite or infinite) groups.

	This class is hashable, which means all subclasses are expected to be immutable.
	'''

	# first, a class should probably define constructors and __repr__ / value_repr (and maybe __str__)

	def __repr__(self):
		'''
		the default implementation returns `Class(<value repr>)` if `value_repr()` is implemented,
		otherwise `Class[<index>]`. if `value_repr()` isn't enough, it could be
		overriden to format a call to some auxiliar method that is preferred for legibility.
		'''
		name = type(self).__name__
		try:
			desc = self.value_repr()
		except NotImplementedError:
			return name + f'[{int(self)}]'
		return name + f'({desc})'

	def value_repr(self) -> str:
		'''
		returns the representation of a single value to be passed to the constructor. this
		is used by the default __repr__ implementation, but also when formatting values for higher
		order groups, such as direct products, where the constructor may be omitted as the value
		may be automatically passed to it.
		'''
		raise NotImplementedError('this group does not support short value representations')

	# core group operations:

	@classmethod
	@abstractmethod
	def _id(cls) -> Self:
		''' the group's identity element

		internal method; users should use `Class.ID` '''

	@abstractmethod
	def _mul(self, other: Self) -> Self:
		''' the group operation

		internal method; users should use the `*` operator instead,
		which validates the type of the other object. '''

	@property
	@abstractmethod
	def inv(self) -> Self:
		''' inverse element. equivalent to the notation `x ** -1` '''

	# because the group is finitely generated, there exists a bijection
	# with the naturals. the following functions implement such a bijection;
	# it is not specified which one (i.e. order of the elements) but it
	# must be a bijection (all elements must map to exactly 1 natural) and
	# it must never change. all of the following functions must expose the
	# same bijection; they must be consistent with each other:

	@abstractmethod
	@classmethod
	def _order(cls) -> Optional[int]:
		''' returns the order of the group (amount of elements it has)
		or None if the group is infinite.

		internal method; users should use `len(Class)` instead. '''

	@classmethod
	@abstractmethod
	def _fromindex(cls, x: int) -> Self:
		''' returns the group element corresponding to natural `x`.

		internal method; users should use `Class[x]` which validates it's
		an integer representing a valid index (nonnegative and less than the
		group's order). '''

	@abstractmethod
	def _index(self) -> int:
		''' returns the natural corresponding to this group element.

		internal method; users should use `int(x)` instead. '''

	@classmethod
	def _enumerate(cls) -> Iterator[Self]:
		'''
		returns an iterator over all elements of this group, in index order.
		this method is optional to implement; `iter` will otherwise use
		`__getitem__` with successive naturals. only implement if there's a more
		efficient way to iterate over the elements.

		internal method; if implemented it will be exposed under `Class.__iter__`,
		and users should use `iter(Class)` to get the described fallback behavior.
		'''
		raise NotImplementedError('this method should not be called directly')

	# comparison / equality / hashing (by default implemented through bijection, override recommended)
	# even if overriden, comparison must be consistent with bijection!

	def __hash__(self):
		return self._cmpkey().__hash__()

	def _cmpkey(self):
		'''
		internal method to return comparison key, to which comparison & hashing
		methods will delegate. for many optimizations, overriding this method rather
		than the six comparison methods will be enough.
		'''
		return self._index()

	def __lt__(self, other: Self):
		if not isinstance(other, type(self)):
			return NotImplemented
		return self._cmpkey().__lt__(type(self)._cmpkey(other))

	def __le__(self, other: Self):
		if not isinstance(other, type(self)):
			return NotImplemented
		return self._cmpkey().__le__(type(self)._cmpkey(other))

	def __gt__(self, other: Self):
		if not isinstance(other, type(self)):
			return NotImplemented
		return self._cmpkey().__gt__(type(self)._cmpkey(other))

	def __ge__(self, other: Self):
		if not isinstance(other, type(self)):
			return NotImplemented
		return self._cmpkey().__ge__(type(self)._cmpkey(other))

	def __eq__(self, other: Self):
		if not isinstance(other, type(self)):
			return NotImplemented
		return self._cmpkey().__eq__(type(self)._cmpkey(other))

	def __ne__(self, other: Self):
		if not isinstance(other, type(self)):
			return NotImplemented
		return self._cmpkey().__ne__(type(self)._cmpkey(other))

	# optional auxiliary operations

	def order(self) -> Optional[int]:
		'''
		returns the order of this element (lowest non-zero natural `x`
		satisfying `self ** x == ID`), or None if it doesn't exist (the cycle
		generated by this element is infinite).

		groups may leave this unimplemented if there isn't an efficient way
		to calculate an element's order.
		'''
		raise NotImplementedError('this group has no efficient way to calculate the order of an element')

	def _pow(self, x: int) -> Self:
		'''
		raises an element to an integer power (may be negative). the default
		implementation uses exponentiation by squaring (together with an optional
		inverse), and it may be overriden if there are more efficient ways to calculate
		powers.

		internal method; users should use `self ** x` notation, which validates for
		integers.
		'''
		# for non-small exponents, round to modulo order of the element
		if abs(x) > 4:
			try:
				order = self.order()
			except NotImplementedError:
				pass
			else:
				if order != None and abs(x) >= order:
					x = x % order
		# for negative exponents, invert the base
		if x < 0:
			self = self.inv
			x = -x
		# exponentiation by squaring
		result = type(self).ID
		mult = self
		while True:
			if x & 1: result *= mult
			x >>= 1
			if not x: break
			mult *= mult
		return result

	# operations provided by the implementation

	ID: ClassVar[Self]

	''' the group's identity element '''
	@classproperty
	def ID(cls):
		return cls._id()  # FIXME: cache

	def __bool__(self):
		return self != type(self).ID

	def __mul__(self, other: Self) -> Self:
		if isinstance(other, type(self)):
			return self._mul(other)
		return NotImplemented

	def __rmul__(self, other: Self) -> Self:
		if isinstance(other, type(self)):
			return type(self)._mul(other, self)
		return NotImplemented

	def __truediv__(self, other: Self) -> Self:
		if isinstance(other, type(self)):
			return self._mul(other.inv)
		return NotImplemented

	def __rtruediv__(self, other: Self) -> Self:
		if isinstance(other, type(self)):
			return type(self)._mul(other, self.inv)
		return NotImplemented

	def __pow__(self, other: int) -> Self:
		if not isinstance(other, int):
			return NotImplemented
		if other == -1:
			# to allow `x ** -1` notation
			return self.inv
		return self._pow(other)

	def conj(self, other: Self) -> Self:
		''' (left) conjugate an element using this element: equivalent to `self.inv * other * self` '''
		return self.inv * other * self

	def conj_by(self, other: Self) -> Self:
		''' (left) conjugate this element by `other`: equivalent to `other.inv * self * other` '''
		return type(self).conj(other, self)

	# bijection (FIXME: limit these to class itself)

	def __int__(self) -> int:
		return self._index()

	@classmethod
	def __len__(cls) -> Optional[int]:
		return cls._order() # FIXME: cache

	@classmethod
	def __getitem__(cls, index: int) -> Self:
		if not isinstance(index, int):
			raise TypeError(f'element indices must be integers, not {type(index)}')
		if not (index >= 0 and ((order := cls._order()) == None or index < order)):
			raise IndexError(f'element index {index} out of range')
		return cls._fromindex(index)

	@classmethod
	def __iter__(cls) -> Iterator[Self]:
		# FIXME: omit if _enumerate yields notimplementederror
		return cls._enumerate()

class SymmetricGroup(Group):
	'''
	symmetric group (over N_n)

	the implemented operation follows usual left action notation, meaning
	`a * b` is equivalent to the composition `a ∘ b` of their associated
	functions (b is performed first, then a).

	the implemented bijection is big-endian factorial number system, so that
	it matches Python's lexicographical comparison on the underlying tuple,
	and it maps ID to 0:

		(0, 1, ..., n-1, n) < (0, 1, ..., n, n-1) < ... < (n, n-1, ..., 0, 1) < (n, n-1, ..., 1, 0)
	'''

	SIZE: ClassVar[int]
	''' size of underlying set (must be defined by child) '''

	# class definition

	_value: tuple[int]

	@property
	def value(self):
		''' underlying permutation value (tuple of indices) '''
		return self._value

	def __init__(self, value: tuple[int]):
		SIZE = type(self).SIZE
		assert isinstance(value, tuple) and len(value) == SIZE
		assert all(0 <= k < SIZE for k in value)
		self._value = value

	def _cmpkey(self):
		return self.value

	def value_repr(self):
		desc = ','.join(map(str, self.value))
		return f'({desc})'

	# core group operations

	@classmethod
	def _id(cls):
		return cls(tuple( range(cls.SIZE) ))

	def _mul(self, other: Self) -> Self:
		return type(self)(tuple( self.value[j] for j in other.value ))

	@property
	def inv(self) -> Self:
		result = [None] * type(self).SIZE
		for i, j in enumerate(self.value):
			result[j] = i
		return type(self)(tuple(result))

	# bijection

	@classmethod
	def _order(cls):
		return math.factorial(cls.SIZE)

	def _index(self) -> int:
		SIZE = type(self).SIZE
		index = 0; seen = 0
		for i, j in enumerate(self.value):
			diff = i - (seen >> j).bit_count()
			seen |= 1 << j
			index = index * (SIZE - i) + (j - diff)
		return index

	@classmethod
	def _fromindex(cls, index: int):
		value = []
		for i in range(cls.SIZE):
			index, j = divmod(index, i + 1)
			value.append(j)
			for i2 in range(i):
				if value[i2] >= j:
					value[i2] += 1
		return cls(tuple(reversed(value)))

	@classmethod
	def _enumerate(cls) -> Iterator[Self]:
		options = collections.deque(range(cls.SIZE))
		value: list[int] = []
		def generator():
			if not len(options):
				return (yield value)
			value.append(options.popleft())
			for i in range(len(options)):
				yield from generator()
				value[-1], options[i] = options[i], value[-1]
			yield from generator()
			options.append(value.pop())
		return (cls(tuple(x)) for x in generator())

	# cycle composition / decomposition + related ops

	def cycles_iter(self) -> Iterator[list[int]]:
		''' like cycles(sort=False), but yields an iterator over the discovered cycles '''
		seen = 0
		while True:
			# consult start of next cycle to extract
			pending = ~seen
			start_bit = pending & ~(pending - 1)
			start = start_bit.bit_length() - 1
			if not (start < type(self).SIZE):
				break
			# extract cycle
			cursor, cycle = start, []
			while True:
				cycle.append(cursor)
				seen |= 1 << cursor
				cursor = self.value[cursor]
				if cursor == start: break
			yield cycle

	def cycles(self, sort=True, fixpoints=True) -> tuple[list[int]]:
		'''
		expresses this permutation as a (normalized) product of disjoint cycles.

		normalization: each cycle begins with its minimal element. cycles are first
		sorted by size (if sort=True), and then by its minimal element.

		parameters:
		 - sort: if True, sort discovered cycles by descending size (cycles of the
		   same size are still solved by ascending minimal element, as noted above).
		 - fixpoints: if False, filter out 1-cycles (fixed points).
		'''
		cycles = self.cycles_iter()
		if not fixpoints:
			cycles = filter(lambda x: len(x) != 1, cycles)
		if sort:
			cycles = sorted(cycles, key=len, reverse=True)
		return tuple(cycles)

	# FIXME: change str repr, add constructor from disjoint cycles, with maybe a strict option?

	def order(self):
		return math.lcm(*map(len, self.iter_cycles()))

	def cycle_type(self) -> tuple[int]:
		''' returns the cycle type (conjugation class) of this permutation (a partition of SIZE) in descending order '''
		return tuple(sorted(map(len, self.iter_cycles()), reverse=True))

	def sign(self) -> int:
		''' returns the sign (0 → even, 1 → odd) of this permutation '''
		# equivalent to ( SIZE - len(cycles()) ) % 2
		# FIXME: simpler / more efficient way?
		return sum(len(c) - 1 for c in self.cycles_iter()) % 2

	# FIXME: is it worth to implement pow?

	# special elements

	@classproperty
	def FULL_CYCLE(cls):
		''' the full cycle permutation, `f(i) = (i + 1) % SIZE` '''
		return cls(tuple( (i+1) % cls.SIZE for i in range(cls.SIZE) ))

	@classproperty
	def REVERSED(cls):
		''' the order reversing permutation, `f(i) = (SIZE - 1) - i` '''
		return cls(cls.ID.value[::-1])

	# group action

	def __call__(self, x: int) -> int:
		''' interprets this permutation as a function from N_n to N_n '''
		assert isinstance(x, int) and 0 <= x < type(self).SIZE
		return self.value[x]

	# (FIXME: pass __len__, __getitem__, __iter__ when the ones from Group are properly limited to the class)

	def apply(self, x: Any) -> Any:
		'''
		creates a shallow copy of the passed container and copies values from
		source to copy using this permutation. returns the copy.

		the container must support __getitem__ and __setitem__ at the domain, and
		must support `copy.copy()`.
		'''
		y = copy.copy(x)
		for i, j in enumerate(self.value):
			y[j] = x[i]
		return y

	def shuffle(self, x: Any):
		'''
		shuffles the elements of a container, in-place, using this permutation

		the container must support __getitem__ and __setitem__ at the domain.
		'''
		for cycle in self.cycles_iter():
			if len(cycle) == 1:
				continue # optimization for common case
			v = x[cycle[0]]
			for idx in range(1, len(cycle)):
				i = cycle[idx]
				v, x[i] = x[i], v
			x[cycle[0]] = v
