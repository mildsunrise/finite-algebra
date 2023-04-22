from typing import Self
from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, ClassVar, Iterator, Iterable, Any, Type, TypeVar, Union, cast
import math
import bisect
import collections
import copy
import itertools
import functools
import operator
import re

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
	'GroupMeta', 'Group', 'ID',
	'CyclicGroup',
	'SymmetricGroup',
	'DirectProduct',
	'SemidirectProduct',
	'WreathProduct',
	'CubeRot',
]


# GROUP
# -----

ID = object() # FIXME: use a better thing
'''
placeholder object that is expanded to the identity element in contexts that
accept short syntax, like the arguments of a group product constructor.

see `Group.short_value()`.
'''

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

	@property
	def ORDER(cls) -> int:
		return cls._order()

	def __bool__(cls) -> bool:
		# the default implementation would call len(cls), which raises
		# OverflowError for lengths that do not fit in an index. use ORDER instead.
		# FIXME: remember to document this quirk and recommend use of ORDER if possible.
		return bool(cls.ORDER)

	def __len__(self) -> int:
		cls = cast(Type['Group'], self)
		return cls._order() # FIXME: maybe translate into a nonexistent method?

	def __getitem__(self: Type[T], index: int) -> T:
		cls = cast(Type['Group'], self)
		order = None
		try:
			order = cls.ORDER
		except ValueError:
			pass

		if not isinstance(index, int):
			raise TypeError(f'element indices must be integers, not {type(index)}')
		if not (index >= 0 and (order == None or index < order)):
			raise IndexError(f'element index {index} out of range')
		return cls._fromindex(index)

	def __iter__(self: Type[T]) -> Iterator[T]:
		cls = cast(Type['Group'], self)
		try:
			return cls._enumerate()
		except NotImplementedError:
			pass
		try:
			indices = range(cls.ORDER)
		except ValueError:
			indices = itertools.count()
		return map(cls._fromindex, indices)

class Group(metaclass=GroupMeta):
	'''
	Base class for finitely generated (finite or infinite) groups.

	This class is hashable, which means all subclasses are expected to be immutable.
	'''

	def __init_subclass__(cls, final=True, **kwargs) -> None:
		super().__init_subclass__(**kwargs)
		if not final:
			return

	# constructing / parsing

	@classmethod
	def short_value(cls, x: Any) -> Self:
		'''
		coerces short value `x` into a group element of type `t`.

		this system is used to allow shorter syntax in cases where the
		expected group type is known by context, such as in group products.

		the default implementation should be fine for most needs:
		 - if the value is already an instance of the expected group, it's
		   returned unchanged.
		 - if the value is the special ID export, then the identity of that
		   group is returned.
		 - otherwise, an element is constructed feeding `x` to the constructor.

		be sure to check out the implementation of your particular Group which
		may be overriden to allow for additional syntax.
		'''
		if isinstance(x, cls):
			return x
		if x is ID:
			return cls.ID
		return cls(x)

	# formatting

	def __str__(self):
		'''
		the default str() implementation just returns `short_repr()`
		'''
		return self.short_repr()

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
			pass
		else:
			return name + ('.ID' if desc is ID else f'({desc})')
		return name + f'[{int(self)}]'

	def short_repr(self) -> str:
		'''
		expresses group element `x` in short syntax understood by `short_value`, see the description
		of that method for the motivation behind this. the default implementation uses `value_repr`
		if implemented, and `__repr__` otherwise.
		'''
		try:
			desc = self.value_repr()
		except NotImplementedError:
			pass
		else:
			return 'ID' if desc is ID else desc
		return repr(self)

	def value_repr(self) -> str:
		'''
		returns the representation of a single value to be passed to the constructor. this
		is used by the default `__repr__` and `short_value` implementations, and for most
		subclasses it should be enough to implement this method. for more advanced cases it
		may be necessary to override `__repr__` and `short_value` instead.

		as an additional feature, if this method returns the `ID` singleton, `__repr__`
		formats `Class.ID` and `short_repr` formats `ID`.
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

		internal method; users should use `Class.ORDER` instead '''

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

	def _pow(self, x: int, order_threshold: Optional[int]=4) -> Self:
		'''
		raises an element to an integer power (may be negative). the default
		implementation uses exponentiation by squaring (together with an optional
		inverse), and it may be overriden if there are more efficient ways to calculate
		powers.

		internal method; users should use `self ** x` notation, which validates for
		integers.
		'''
		# for non-small exponents, round to modulo order of the element
		if order_threshold and abs(x) > order_threshold:
			try:
				order = self.order()
			except NotImplementedError | ValueError:
				pass
			else:
				if abs(x) >= order:
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

	def comm(self, other: Self) -> Self:
		''' obtains a commutator element: equivalent to `self.inv * other.inv * self * other` '''
		return self.inv * other.inv * self * other

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
		assert isinstance(value, int), f'object {repr(value)} is not an int'
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

	@property
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

	# parsing / formatting

	@classmethod
	def short_value(cls, x: Any) -> Self:
		'''
		in addition to the forms listed in `Group.short_value()`, this group
		accepts an additional syntax: if the value is a list, then its items
		are passed to `from_cycles()` to construct the element.
		'''
		if isinstance(x, list):
			return cls.from_cycles(*x)
		return super().short_value(x)

	def value_repr(self):
		repr_cycle = lambda c: '[' + ','.join(map(str, c)) + ']'
		return ', '.join(map(repr_cycle, self.cycles(fixpoints=False)))

	def short_repr(self) -> str:
		return ('[' + self.value_repr() + ']') if self else 'ID'

	def __repr__(self):
		name = type(self).__name__
		return name + (f'.from_cycles({self.value_repr()})' if self else '.ID')

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

	def cycles(self, sort=True, fixpoints=True) -> list[list[int]]:
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
		return list(cycles)

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

	(ideally this would be a subclass of SemidirectProduct, but doing it
	separately allows greater flexibility such as arbitrary number of components)

	note: all groups must be finite except for the first one, which may be infinite
	'''

	PARTS: ClassVar[tuple[Type[Group], ...]]
	''' group for each component of the product (must be defined by child) '''

	# class definition

	_value: tuple

	@property
	def value(self):
		''' underlying value '''
		return self._value

	def __init__(self, *value):
		'''
		construct an element of the group, using the elements provided for each of the parts.

		short syntax is accepted for the arguments, see `Group.short_value()`.
		the amount of arguments provided must match `len(PARTS)`.
		'''
		self._value = tuple( t.short_value(x) for t, x in zip(type(self).PARTS, value, strict=True) )

	def _cmpkey(self):
		return self.value

	# parsing / formatting

	@classmethod
	def short_value(cls, x: Any) -> Self:
		'''
		short syntax for this group works as usual, except that the tuple is spread
		into the constructor's arguments rather than passed as a single argument.
		'''
		if isinstance(x, cls):
			return x
		if x is ID:
			return cls.ID
		assert isinstance(x, tuple), f'short syntax expects a tuple of arguments'
		return cls(*x)

	def value_repr(self):
		return ', '.join(x.short_repr() for x in self.value)

	def short_repr(self) -> str:
		return '(' + self.value_repr() + ')'

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
		return cls(*( g.ID for g in cls.PARTS ))

	def _mul(self, other: Self) -> Self:
		return type(self)(*( a * b for a, b in zip(self, other) ))

	@property
	def inv(self) -> Self:
		return type(self)(*( a.inv for a in self ))

	# bijection

	@classmethod
	def _order(cls):
		return math.prod(map(len, cls.PARTS))

	def _index(self) -> int:
		index = 0
		for t, x in zip(type(self).PARTS, self.value):
			index = index * t.ORDER + int(x)
		return index

	@classmethod
	def _fromindex(cls, index: int):
		result = []
		for t in reversed(cls.PARTS):
			index, x = divmod(index, t.ORDER)
			result.append(t[x])
		return cls(*reversed(result))

	@classmethod
	def _enumerate(cls) -> Iterator[Self]:
		def generator(parts = cls.PARTS, prefix = ()):
			if not parts:
				return (yield cls(*prefix))
			g, *parts = parts
			for x in g:
				yield from generator(parts, prefix + (x,))
		return generator()

	# other operations

	def _pow(self, x: int) -> Self:
		return type(self)(*( a ** x for a in self ))

	def order(self) -> int:
		return math.lcm(*( a.order() for a in self ))


# SEMIDIRECT PRODUCT
# ------------------

# Python's type system seems too limited for this, but we can at least use type aliases to make things clearer
N = H = Group

class SemidirectProduct(Group, final=False):
	'''
	(outer) semidirect product of two groups N ⋊ H, as an `(n, h)` tuple

	implemented bijection matches Python's lexicographical tuple comparison

	note: quotient subgroup H must be finite
	'''

	# FIXME: class properties N and H, maybe

	PARTS: tuple[Type[N], Type[H]]
	''' the 2 groups for the product: normal subgroup N and quotient group H (must be defined by child) '''
	# FIXME: this should be ClassVar, but it's forbidden for type variables...?

	@classmethod
	@abstractmethod
	def semidirect_homomorphism(cls, h: H, n: N) -> N:
		'''
		H → Aut(N) homomorphism that characterises this semidirect product (must be defined by child)

		conceptually this is a mapping from elements of H into functions from N to N,
		but the curried form (taking both H and N) is defined here for simplicity. '''

	@property
	def n(self):
		''' underlying element of the normal subgroup N '''
		return self._value[0]

	@property
	def h(self):
		''' underlying element of the quotient group H '''
		return self._value[1]

	# class definition (this part is very much copied from DirectProduct)

	_value: tuple[N, H]

	@property
	def value(self):
		''' underlying value '''
		return self._value

	def __init__(self, n: N = ID, h: H = ID):
		'''
		construct an element of the group, using the elements provided N and H.

		short syntax is accepted for the arguments, see `Group.short_value()`.
		'''
		self._value = tuple( t.short_value(x) for t, x in zip(type(self).PARTS, (n, h), strict=True) )

	def _cmpkey(self):
		return self.value

	# parsing / formatting (this part is very much copied from DirectProduct)

	@classmethod
	def short_value(cls, x: Any) -> Self:
		'''
		short syntax for this group works as usual, except that the pair is spread
		into the constructor's arguments rather than passed as a single argument.
		'''
		if isinstance(x, cls):
			return x
		if x is ID:
			return cls.ID
		assert isinstance(x, tuple) and len(x) == 2, f'short syntax expects a tuple of 2 arguments'
		return cls(*x)

	def value_repr(self):
		return ', '.join(x.short_repr() for x in self.value)

	def short_repr(self) -> str:
		return '(' + self.value_repr() + ')'

	# pass sequence protocol to underlying tuple (this part is very much copied from DirectProduct)

	def __len__(self, *a, **k):
		return type(self.value).__len__(self.value, *a, **k)
	def __getitem__(self, *a, **k):
		return type(self.value).__getitem__(self.value, *a, **k)
	def __iter__(self, *a, **k):
		return type(self.value).__iter__(self.value, *a, **k)

	# core group operations

	@classmethod
	def _id(cls):
		return cls(*( g.ID for g in cls.PARTS ))

	def _mul(self, other: Self) -> Self:
		phi = type(self).semidirect_homomorphism
		return type(self)( self.n * phi(self.h, other.n), self.h * other.h )

	@property
	def inv(self) -> Self:
		phi = type(self).semidirect_homomorphism
		hinv = self.h.inv
		return type(self)( phi(hinv, self.n.inv), hinv )

	# bijection (this part is very much copied from DirectProduct)

	@classmethod
	def _order(cls):
		return math.prod(map(len, cls.PARTS))

	def _index(self) -> int:
		index = 0
		for t, x in zip(type(self).PARTS, self.value):
			index = index * t.ORDER + int(x)
		return index

	@classmethod
	def _fromindex(cls, index: int):
		result = []
		for t in reversed(cls.PARTS):
			index, x = divmod(index, t.ORDER)
			result.append(t[x])
		return cls(*reversed(result))

	@classmethod
	def _enumerate(cls) -> Iterator[Self]:
		def generator(parts = cls.PARTS, prefix = ()):
			if not parts:
				return (yield cls(*prefix))
			g, *parts = parts
			for x in g:
				yield from generator(parts, prefix + (x,))
		return generator()

	# other operations

	def _pow(self, x: int, order_threshold: Optional[int]=4) -> Self:
		original = (lambda p: lambda: p(x, order_threshold=None))(super()._pow)
		if order_threshold and abs(x) <= order_threshold:
			return original()

		try:
			h_order = self.h.order()
		except NotImplementedError | ValueError:
			return original()
		if abs(x) < h_order:
			return original()

		n_quot, h_quot = super()._pow(h_order)
		assert not h_quot

		q, x = divmod(x, h_order)
		n_mod, h_mod = original()
		return type(self)(n_quot ** q * n_mod, h_mod)

	def order(self) -> int:
		h_order = self.h.order()
		n_quot, h_quot = super()._pow(h_order, order_threshold=None)
		assert not h_quot
		return n_quot.order() * h_order


# WREATH PRODUCT
# --------------

# Python's type system seems too limited for this, but we can at least use type aliases to make things clearer
Bottom = Group
Top = SymmetricGroup

class WreathProduct(SemidirectProduct, final=False):
	'''
	wreath product, with a symmetric (sub)group as top

	note: both bottom and top groups need to be finite
	'''

	BOTTOM: ClassVar[Type[Bottom]]
	''' bottom part of the wreath product (must be defined by child) '''
	TOP: ClassVar[Type[Top]]
	''' top part of the wreath product (must be defined by child) '''

	INVERTED: ClassVar[bool] = True
	'''
	whether to invert the automorphism (phi) derived from the permutation.

	this is set to True by default, meaning the bottom elements refer to
	the PRE permutation arrangement, i.e. bottoms are applied first and
	then permuted. setting it to False causes bottom elements to be applied
	after the permutation, i.e. refer to the POST permutation arrangement.

	this is set to True by default because it is usually easier to reason
	about elements before the permutation occurs, at least in my experience.
	'''

	# core class

	def __init_subclass__(cls, final=True, **kwargs):
		if not final:
			return super().__init_subclass__(final, **kwargs)

		class N(DirectProduct):
			PARTS = (cls.BOTTOM,) * cls.TOP.SIZE
			__name__ = cls.__name__ + '.N'
		cls.N = N
		cls.PARTS = (N, cls.TOP)

		super().__init_subclass__(final, **kwargs)

	@classmethod
	def semidirect_homomorphism(cls, h: H, n: N) -> N:
		h = h.inv if cls.INVERTED else h
		return cls.N(*( n[h(i)] for i in range(len(n)) ))

	# parsing / formatting

	@classmethod
	def short_value(cls, x: Any) -> Self:
		'''
		in addition to the forms listed in `Group.short_value()`, this group
		accepts an additional syntax: if the value is a list, then its items
		are passed to `from_cycles()` to construct the element.
		'''
		if isinstance(x, list):
			return cls.from_cycles(*x)
		return super().short_value(x)

	def value_repr(self):
		def repr_cycle(c: tuple[list[int], list[Bottom]]):
			a = ','.join(map(str, c[0]))
			b = ','.join(x.short_repr() for x in c[1])
			return f'([{a}], [{b}])'
		return ', '.join(map(repr_cycle, self.cycles(fixpoints=False)))

	def short_repr(self) -> str:
		return ('[ ' + self.value_repr() + ' ]') if self else 'ID'

	def __repr__(self):
		name = type(self).__name__
		return name + (f'.from_cycles( {self.value_repr()} )' if self else '.ID')

	# cycle composition / decomposition

	def cycles_iter(self) -> Iterator[ tuple[list[int], list[Bottom]] ]:
		''' like cycles(sort=False), but yields an iterator over the discovered cycles '''
		return ( (c, [self.n[i] for i in c]) for c in self.h.cycles_iter() )

	# signature and code for this method have been copied from SymmetricGroup
	# and should be kept in sync with it
	def cycles(self, sort=True, fixpoints=True) -> list[ tuple[list[int], list[Bottom]] ]:
		'''
		this decomposes the top permutation into cycles (see `SymmetricGroup.cycles()`)
		but also annotates each cycle with the series of bottoms for each of their
		elements, in order.

		for example, where `self.h.cycles()` would return a `[1,3,2]` element,
		this would return `([1,3,2], [self.n[1],self.n[3],self.n[2]])`.

		everything else works the same as `SymmetricGroup.cycles()`, with one exception:
		`fixpoints=False` doesn't filter out *all* fixpoints, only those who have their
		corresponding bottom set to identity.
		'''
		cycles = self.cycles_iter()
		if not fixpoints:
			ID = type(self).BOTTOM.ID
			cycles = filter(lambda c: c[1] != [ID], cycles)
		if sort:
			cycles = sorted(cycles, key=lambda c: len(c[0]), reverse=True)
		return list(cycles)

	@classmethod
	def from_cycles(cls, *cycles: tuple[list[int], list[Bottom]], **kwargs) -> Self:
		'''
		constructs an element from a series of (cycle, bottom_list) pairs;
		see cycles() for more info.

		keyword arguments are passed to `SymmetricGroup.from_cycles()` when creating
		the top element.

		short syntax is accepted for the bottom elements, see `Group.short_value()`.
		'''
		# first of all, create top element (this validates the input)
		top = cls.TOP.from_cycles(*(cycle[0] for cycle in cycles), **kwargs)
		# create bottom element
		bottoms = [ID] * cls.TOP.SIZE
		for cycle in cycles:
			for k, v in zip(*cycle, strict=True):
				bottoms[k] = v
		return cls(tuple(bottoms), top)

	# other operations

	def order(self) -> int:
		INVERTED = type(self).INVERTED
		residue = lambda b: functools.reduce(operator.mul, b[::-1] if INVERTED else b)  # FIXME: verify!!!
		cycle_order = lambda c: len(c[0]) * residue(c[1]).order()
		return math.lcm(*map(cycle_order, self.cycles_iter()))

	# FIXME: override _pow as well


# AUTOMAGICAL GROUP CREATION
# --------------------------

PREFIXES = {
	'S': SymmetricGroup,
	'Z': CyclicGroup,
}

def __getattr__(name: str):
	if (m := re.fullmatch(r'(\D+)(\d+)', name)) and (t := PREFIXES.get(m.group(1))) != None:
		class AutoGroup(t):
			SIZE = int(m.group(2))
		AutoGroup.__name__ = AutoGroup.__qualname__ = name
		globals()[name] = AutoGroup
		return AutoGroup
	mod = __import__(__name__)
	return type(mod).__getattr__(mod, name)


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
		if not self:
			return ID
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
		set_elem(axis + 'p', laxis + '⁻', AXIS ** -1)

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
