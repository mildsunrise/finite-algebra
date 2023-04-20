'''
This code is NOT safe for cryptography; operation time may depend on the values.

Flexibility first, readability second, performance third (but we generally care about O(n) complexity).
'''

from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, ClassVar, Self, Iterator, Iterable, Any, Type, TypeVar, Union, cast
import math
import bisect
import collections
import copy

T = TypeVar('T')

def circular_pairwise(x: Iterable[T]) -> Iterator[tuple[T, T]]:
	''' like pairwise, but with a trailing (last, first) entry '''
	x = iter(x)
	start = next(x)
	e1 = start
	for e2 in x:
		yield (e1, e2)
		e1 = e2
	yield (e1, start)

class classproperty(property):
	def __get__(self, owner_self, owner_cls):
		return cast(Any, self.fget)(owner_cls)

# FIXME: maybe turn asserts into proper exceptions

__all__ = [
	'GroupMeta', 'Group',
	'CyclicGroup',
	'SymmetricGroup',
	'DirectProduct',
	'CubeRot',
]


# GROUP
# -----

class GroupMeta(ABCMeta):
	''' Metaclass for Group.

	This is used for several things:
	 - to allow identification of Group classes
	 - to give class-level syntax for the bijection (__len__, __getitem__, __iter__)
	 - to provide some other behavior (caching, validation at class define time...)

	*Never* use this metaclass without subclassing from Group.
	'''

	# FIXME: cache ID and __len__

	@property
	def ID(self: Type[T]) -> T:
		''' the group's identity element '''
		cls = cast(Type['Group'], self)
		return cls._id()

	def __len__(self) -> int:
		cls = cast(Type['Group'], self)
		return cls._order() # FIXME: maybe translate into a nonexistent method?

	def __getitem__(self: Type[T], index: int) -> T:
		cls = cast(Type['Group'], self)
		if not isinstance(index, int):
			raise TypeError(f'element indices must be integers, not {type(index)}')
		if not (index >= 0 and ((order := cls._order()) == None or index < order)):
			raise IndexError(f'element index {index} out of range')
		return cls._fromindex(index)

class Group(metaclass=GroupMeta):
	'''
	Base class for finitely generated (finite or infinite) groups.

	This class is hashable, which means all subclasses are expected to be immutable.
	'''

	def __init_subclass__(cls, final=True, **kwargs) -> None:
		super().__init_subclass__(**kwargs)
		if not final:
			return

		try:
			cls._enumerate()
		except NotImplementedError:
			pass
		else:
			def __iter__() -> Iterator[T]:
				return cast(Type['Group'], cls)._enumerate()
			cls.__iter__ = __iter__

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

	@classmethod
	@abstractmethod
	def _order(cls) -> int:
		''' returns the order of the group (amount of elements it has)
		or raise ValueError if the group is infinite.

		internal method; users should use `len(Class)` instead '''

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

	def order(self) -> int:
		'''
		returns the order of this element (lowest non-zero natural `x`
		satisfying `self ** x == ID`), or raise ValueError if it doesn't
		exist (the cycle generated by this element is infinite).

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

	# bijection (class-level syntax is already provided by the metaclass)

	def __int__(self) -> int:
		return self._index()


# CYCLIC GROUP
# ------------

class CyclicGroup(Group, final=False):
	''' finite cyclic group (N_n) '''

	SIZE: ClassVar[int]
	''' order of the group (must be defined by child and be positive) '''

	@classmethod
	def _order(cls):
		return cls.SIZE

	# class

	_value: int

	@property
	def value(self):
		return self._value

	def __init__(self, value: int):
		assert isinstance(value, int)
		self._value = value % self.SIZE

	def value_repr(self) -> str:
		return repr(self.value)

	# bijection

	def _index(self) -> int:
		return self.value

	@classmethod
	def _fromindex(cls, x: int):
		return cls(x)

	# group operations

	@classproperty
	def G(cls):
		''' generator of the group (value 1) '''
		return cls(1)

	@classmethod
	def _id(cls) -> Self:
		return cls(0)

	def _mul(self, other: Self) -> Self:
		return type(self)(self.value + other.value)

	def inv(self):
		return type(self)(-self.value)

	# other operations

	def _pow(self, x: int):
		return type(self)(self.value * x)

	def order(self) -> int:
		return type(self).SIZE // math.gcd(type(self).SIZE, self.value)


# SYMMETRIC GROUP
# ---------------

class SymmetricGroup(Group, final=False):
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

	_value: tuple[int, ...]

	@property
	def value(self):
		''' underlying permutation value (tuple of indices) '''
		return self._value

	def __init__(self, value: tuple[int, ...]):
		SIZE = type(self).SIZE
		assert isinstance(value, tuple) and len(value) == SIZE
		assert all(0 <= k < SIZE for k in value)
		self._value = value

	def _cmpkey(self):
		return self.value

	def value_repr(self):
		return '(' + ','.join(map(str, self.value)) + ')'

	# core group operations

	@classmethod
	def _id(cls):
		return cls(tuple( range(cls.SIZE) ))

	def _mul(self, other: Self) -> Self:
		return type(self)(tuple( self.value[j] for j in other.value ))

	@property
	def inv(self) -> Self:
		result = [-1] * type(self).SIZE
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

	# cycle decomposition

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

	# cycle type & derived properties

	def cycle_type(self) -> tuple[int, ...]:
		''' returns the cycle type (conjugation class) of this permutation (a partition of SIZE) in descending order '''
		return tuple(sorted(map(len, self.cycles_iter()), reverse=True))

	def order(self) -> int:
		return math.lcm(*map(len, self.cycles_iter()))

	def sign(self) -> int:
		''' returns the sign (0 → even, 1 → odd) of this permutation '''
		# equivalent to ( SIZE - len(cycles()) ) % 2
		return sum(len(c) - 1 for c in self.cycles_iter()) % 2

	# cycle composition

	@classmethod
	def from_cycles(cls, *cycles: Iterable[int], strict=False):
		'''
		construct a permutation from disjoint cycles.

		if strict=True, fixpoints must be explicitly mentioned.
		'''
		result = [-1] * cls.SIZE
		for cycle in cycles:
			for i, j in circular_pairwise(cycle):
				assert isinstance(i, int) and 0 <= i < cls.SIZE and result[i] == -1
				result[i] = j
		if not strict:
			for i, j in enumerate(result):
				if j == -1:
					result[i] = i
		return cls(tuple(result))

	# FIXME: change str repr, maybe move 'from_cycles' to constructor?
	# also, with the current spead form (*cycles) we can't take advantage of lazyness

	def _pow(self, x: int) -> Self:
		result = [-1] * type(self).SIZE
		for cycle in self.cycles_iter():
			for i in range(len(cycle)):
				result[cycle[i]] = cycle[(i + x) % len(cycle)]
		return type(self)(tuple(result))

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

	def __len__(self, *a, **k):
		return type(self.value).__len__(self.value, *a, **k)
	def __getitem__(self, *a, **k):
		return type(self.value).__getitem__(self.value, *a, **k)
	def __iter__(self, *a, **k):
		return type(self.value).__iter__(self.value, *a, **k)

	def apply(self, x: Any) -> Any:
		'''
		creates a shallow copy of the passed container and copies values from
		source to copy using this permutation. returns the copy.

		the container must support __getitem__ and __setitem__ at the domain, and
		must support `copy.copy()`.
		'''
		y = copy.copy(x)
		for i, j in enumerate(self.value):
			if i == j:
				continue # optimization for common case
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
			for i in cycle[1:]:
				v, x[i] = x[i], v
			x[cycle[0]] = v


# DIRECT PRODUCT
# --------------

class DirectProduct(Group, final=False):
	'''
	direct product of groups, with tuple shape

	implemented bijection matches Python's lexicographical tuple comparison
	'''

	PARTS: ClassVar[tuple[Type[Group], ...]]
	''' underlying components (must be defined by child) '''

	# class definition

	_value: tuple

	@property
	def value(self):
		''' underlying value '''
		return self._value

	def __init__(self, value: tuple):
		assert isinstance(value, tuple)
		self._value = tuple( x if isinstance(x, t) else t(x) for t, x in zip(type(self).PARTS, value, strict=True) )

	def _cmpkey(self):
		return self.value

	def value_repr(self):
		def short_repr(x: Group) -> str:
			try:
				return x.value_repr()
			except NotImplementedError:
				return repr(x)
		return '(' + ', '.join(map(short_repr, self.value)) + ')'

	# pass sequence protocol to underlying tuple

	def __len__(self, *a, **k):
		return type(self.value).__len__(self.value, *a, **k)
	def __getitem__(self, *a, **k):
		return type(self.value).__getitem__(self.value, *a, **k)
	def __iter__(self, *a, **k):
		return type(self.value).__iter__(self.value, *a, **k)

	# core group operations

	@classmethod
	def _id(cls):
		return cls(tuple( g.ID for g in cls.PARTS ))

	def _mul(self, other: Self) -> Self:
		return type(self)(tuple( a * b for a, b in zip(self, other) ))

	@property
	def inv(self) -> Self:
		return type(self)(tuple( a.inv for a in self ))

	# bijection

	@classmethod
	def _order(cls):
		return math.prod(map(len, cls.PARTS))

	def _index(self) -> int:
		index = 0
		for t, x in zip(type(self).PARTS, self.value):
			index = index * len(t) + int(x)
		return index

	@classmethod
	def _fromindex(cls, index: int):
		result = []
		for t in reversed(cls.PARTS):
			index, x = divmod(index, len(t))
			result.append(t[x])
		return cls(tuple(reversed(result)))

	@classmethod
	def _enumerate(cls) -> Iterator[Self]:
		def generator(parts = cls.PARTS, prefix = ()):
			if not parts:
				return (yield cls(prefix))
			g, *parts = parts
			for x in g:
				yield from generator(parts, prefix + (x,))
		return generator()

	# other operations

	def _pow(self, x: int) -> Self:
		return type(self)(tuple( a ** x for a in self ))

	def order(self) -> int:
		return math.lcm(*( a.order() for a in self ))


# APPLICATION-SPECIFIC
# --------------------

class CubeRot(SymmetricGroup):
	''' group of rotations of a cube (isomorphic to S_4)

	polarity: this uses `Z = X.cony_by(Y)` convention.
	Z equals FULL_CYCLE; X2 equals REVERSED. '''

	SIZE = 4

	# base elements
	X: ClassVar['CubeRot']
	Y: ClassVar['CubeRot']
	Z: ClassVar['CubeRot']

	# inverses, squares
	Xp: ClassVar['CubeRot']
	Yp: ClassVar['CubeRot']
	Zp: ClassVar['CubeRot']
	X2: ClassVar['CubeRot']
	Y2: ClassVar['CubeRot']
	Z2: ClassVar['CubeRot']

	# formatting
	LABELS: ClassVar[dict['CubeRot', str]] = {}
	LABELS_REV: ClassVar[dict[str, 'CubeRot']]

	def value_repr(self):
		return repr(type(self).LABELS.get(self)) or super().value_repr()

	def __init__(self, value: Union[str, tuple[int, ...]]):
		if isinstance(value, str):
			self._value = self.LABELS_REV[value].value
		else:
			super().__init__(value)

	@classmethod
	def join(cls, orientation: int, face: tuple[int, ...]) -> 'CubeRot':
		''' opposite of split() '''
		return cls.Z ** orientation * cls(face + (3,))

	def split(self) -> tuple[int, tuple[int, ...]]:
		'''
		split this cube rotation into an (orientation, face) tuple

		it is implementation-dependent which is the identity orientation;
		but it's guaranteed that `(Z * self).split()[0] == 1 + self.split()[0]`.
		'''
		value = self.inv.value
		orientation = value.index(3)
		return orientation, value[orientation + 1:] + value[:orientation]

def __init(cls = CubeRot):
	cls.LABELS[cls.ID] = '1'
	def set_elem(name: str, label: str, elem: CubeRot):
		cls.LABELS[elem] = label
		setattr(cls, name, elem)
	def set_composition(a: CubeRot, b: CubeRot):
		cls.LABELS[a * b] = cls.LABELS[a] + ' ' + cls.LABELS[b]

	set_elem('X', 'x', CubeRot.from_cycles([0,1,3,2]))
	set_elem('Y', 'y', CubeRot.from_cycles([0,3,1,2]))
	set_elem('Z', 'z', cls.X.conj_by(cls.Y))
	assert cls.Z == cls.FULL_CYCLE

	for axis in 'XYZ':
		laxis, AXIS = axis.lower(), getattr(cls, axis)
		set_elem(axis + '2', laxis + '²', AXIS ** 2)
		set_elem(axis + 'p', laxis + "'", AXIS ** -1)

	for AXIS in (cls.X, cls.Y):
		for sense in (+1, -1):
			for orientation in range(1, 4):
				set_composition(cls.Z ** orientation, AXIS ** sense)

	for orientation in (1, -1):
		set_composition(cls.Z ** orientation, cls.X2)

	cls.LABELS_REV = { v: k for k, v in cls.LABELS.items() }
	assert len(cls.LABELS) == len(cls) and len(cls.LABELS_REV) == len(cls)

__init()
del __init
