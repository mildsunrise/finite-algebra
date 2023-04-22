# finite-algebra

This is (a prototype of) a Python library for finite ~~algebra~~ group theory, which encapsulates elements of a structure as instances of an immutable class, and allow easy and expressive operation with them.

It's designed to be both practical and expressive to use as a calculator (for playing and experiments), and also high-quality and ergonomical enough for use in production code. I've attempted to keep code quality at a minimum, so that it can serve for learning purposes or as base for production code.

Priorities are: flexibility first, readability second, performance third (but we generally care about O(n) complexity).

Disclaimer:

> This code is **not** safe for cryptography; operation time may depend on the values.


## Table of contents

- **[Status](#status)**
  * [Class structure](#class-structure)
  * [Group features](#group-features)
  * [Compatibility](#compatibility)
- **[Getting started](#getting-started)**
  * [Element enumeration / indexing](#element-enumeration---indexing)
  * [Working with elements](#working-with-elements)
  * [The symmetric group](#the-symmetric-group)
  * [Automagical group creation](#automagical-group-creation)
  * [Composing groups using `DirectProduct`](#composing-groups-using--directproduct-)
  * [Implementing custom groups](#implementing-custom-groups)


## Status

### Class structure

Right now there's only basic **finite group theory**. In particular:

 - **`Group`**: Base class & supporting metaclass

 - Common base groups, indexed by `SIZE` ($n$):

   - **`CyclicGroup`**: [Cyclic group](https://en.wikipedia.org/wiki/Cyclic_group) over the integers

   - **`SymmetricGroup`**: [Symmetric group](https://en.wikipedia.org/wiki/Symmetric_group) over $N_n$

 - Constructions of groups from other groups:

   - **`DirectProduct`**: [Direct product](https://en.wikipedia.org/wiki/Direct_product) of an arbitrary amount of groups, with tuple shape

   - **`SemidirectProduct`**: (outer) left [semidirect product](https://en.wikipedia.org/wiki/Semidirect_product) of two groups

   - **`WreathProduct`**: [Wreath product](https://en.wikipedia.org/wiki/Wreath_product) of two groups, with the top one being a (subgroup of a) symmetric group (inherits from `SemidirectProduct`)

 - Common application-specific groups:

   - **`CubeRot`**: Group of (proper) rotations of a cube (isomorphic to $S_4$, inherits from `SymmetricGroup`)

### Group features

The base `Group` class provides a number of features to make elements idiomatic and expressive while reducing the amount of boilerplate required for subclasses, in a way similar to `dataclass`. These are:

 - Operator overloading (`*` for group operation)

 - Default `**` implementation using [exponentiation by squaring](https://en.wikipedia.org/wiki/Exponentiation_by_squaring) (with optional reduction using element order if possible)

 - The class itself implements the [sequence protocol](https://docs.python.org/3/library/functions.html#iter) to enumerate and index elements by naturals

 - `int()` gets the index of an element; `bool()` tests for non-identity

 - Order of the group offered as `ORDER` property on class

 - Identity element offered as `ID` property on class

 - Elements are comparable, hashable and immutable (by default this is provided through the index, but there's a central `_cmpkey` method that is recommended to override)

 - Default behavior for `repr(element)` and `str(element)`

 - "Short syntax" system for concise expression of composite group elements

 - Optional `order()` API for groups that offer an efficient way to calculate the order of an element

 - Convenience shortcuts: `conj`, `conj_by`, `comm` to compute conjugate and commutator elements

Many of these features are introduced in [Getting started](#getting-started) below. Refer to the code of the `Group` and `GroupMeta` classes for more info, especially if you're planning to implement your own group from scratch.

### Compatibility

We're targeting Python 3.7 (for among other things, `__getname__` module support),
but we're not there yet. For now Python 3.10 is needed.

Typing support: 🤡. Yes, there are annotations all over the code and they've helped catch bugs, but they are often incorrect and sometimes deferred to `Any` because I don't think Python's type system is powerful enough to express the abstraction required by this library (in particular, generic type arguments that are resolved at subclass time rather than instance time). If you're a Python wizard and want to fix this, contributions are very welcome.

I don't think I'll have the energy to make this into a more mature project, though.


## Getting started

### Element enumeration / indexing

Each group is a class that ultimately inherits from `Group`. For example, the *symmetric group of 3 elements* is a class named `S3`:

~~~ python
>>> from groups import S3
>>> S3
<class 'groups.S3'>
~~~

Elements of the group are instances of that class. The class is actually a **sequence** of all its elements, so we can iterate over them:

~~~ python
>>> for element in S3: print(repr(element))
S3.ID
S3.from_cycles([1,2])
S3.from_cycles([0,1])
S3.from_cycles([0,1,2])
S3.from_cycles([0,2,1])
S3.from_cycles([0,2])
~~~

And we can use the `ORDER` class property to count them. This is faster than `len(list(S3))` because it computes the number directly:

~~~ python
>>> S3.ORDER
6
~~~

> **Note:** Since it's a sequence, `len(S3)` is also supported. However `ORDER` should be preferred if possible, since for large groups it can exceed the maximum size of a sequence and raise `OverflowError`. Indexing and iteration still work normally in these cases.

We can also obtain an element by its index, as you'd expect:

~~~ python
>>> S3[3]
S3.from_cycles([0,1,2])
~~~

And we can obtain the index of an element by converting to an `int`:

~~~ python
>>> int(S3[3])
3
~~~

This is useful for compact serialization, among other things.

> In technical terms, all `Group` subclasses are required to implement a **bijection** from the group to the naturals. Which particular bijection is implemented (meaning, the order in which the elements are iterated / mapped to indexes) depends on the implementation, but it is guaranteed all elements are mapped to one (unique) index in the sequence.
>
> It is conventional to map the identity element (`S3.ID`) to index 0 (`S3[0]`), but this is again not a guarantee.

### Working with elements

Two elements can be operated with the `*` operator:

~~~ python
>>> a = S3.from_cycles([0,1,2])
>>> b = S3.from_cycles([1,2])
>>> a * b
S3.from_cycles([0,1])
~~~

The [inverse](https://en.wikipedia.org/wiki/Inverse_element) of an element can be obtained through its `inv` property or with `** -1`:

~~~ python
>>> a.inv
S3.from_cycles([0,2,1])
>>> a ** -1
S3.from_cycles([0,2,1])
~~~

The [identity element](https://en.wikipedia.org/wiki/Identity_element) of the group can be obtained through the `ID` property of the class, as you saw above:

~~~ python
>>> S3.ID * a == a
True
>>> a * S3.ID == a
True
~~~

The *truth value* of an element is `False` for the identity and `True` for every other element:

~~~ python
>>> bool(S3.ID)
False
>>> bool(a)
True
~~~

Elements can also be *compared*; this is expected to be consistent with the bijection, so comparing two elements is equivalent to comparing their indexes:

~~~ python
>>> a >= b
True
>>> int(a) >= int(b)
True
~~~

Elements are also hashable, which means you can use them as keys in a set or dictionary:

~~~ python
>>> { S3.ID: 'identity', b: 'my favorite element' }
{S3.ID: 'identity', S3.from_cycles([1,2]): 'my favorite element'}
~~~

Elements can also be exponentiated to an integer; this is often faster than multiplying N instances of the element, as it uses group-specific optimizations if possible:

~~~ python
>>> a ** -22340981
S3.from_cycles([0,1,2])
~~~

Most groups provide an efficient way to compute the [order](https://en.wikipedia.org/wiki/Order_(group_theory)) of an element, through the `order()` method:

~~~ python
>>> a.order()
3
~~~

Other operations include obtaining [conjugate](https://en.wikipedia.org/wiki/Conjugacy_class) and [commutator](https://en.wikipedia.org/wiki/Commutator) elements, see methods `conj`, `conj_by`, `comm` for more info.

### The symmetric group

The [symmetric group over N elements](https://en.wikipedia.org/wiki/Symmetric_group) is a group made of all the permutations of a set of N elements. This is implemented through the `SymmetricGroup` class, which is a **non-final** group class. This means you can't use it directly:

~~~ python
>>> from groups import SymmetricGroup
>>> SymmetricGroup
<class 'groups.SymmetricGroup'>
>>> list(SymmetricGroup)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "groups.py", line 103, in __iter__
    return cls._enumerate()
  File "groups.py", line 578, in _enumerate
    options.append(value.pop())
AttributeError: type object 'SymmetricGroup' has no attribute 'SIZE'
~~~

Instead, you need to subclass it and define `SIZE` (the amount of elements of the set) on your subclass. For example, the symmetric group over 10 elements can be defined as follows:

~~~ python
class MyGroup(SymmetricGroup):
    SIZE = 10
~~~

This also gives you a chance to override methods to customize the default representation of elements, or other properties.

For now though we'll keep working with `S3`, which is a subclass of `SymmetricGroup` with `SIZE = 3` defined by the library (see [automagical group creation](#automagical-group-creation) below).

Elements can be constructed directly by feeding the permutation tuple to the constructor:

~~~ python
>>> S3((0, 1, 2))
S3.ID
>>> S3((1, 0, 2))
S3.from_cycles([0,1])
~~~

And the permutation tuple can be obtained back through the `value` property:

~~~ python
>>> b.value
(0, 2, 1)
~~~

The tuple specifies the permutation in [one-line notation](https://en.wikipedia.org/wiki/Permutation#One-line_notation): each slot has the zero-based index of the slot it's sent to. They're the "raw", natural representation of a permutation. They're fine for computers but not helpful to understand what the permutation *does*; for this it's better to separate the permutation into [disjoint cycles](https://en.wikipedia.org/wiki/Permutation#Cycle_notation):

~~~ python
>>> b.cycles()
[[1, 2], [0]]
~~~

This tells us the permutation swaps slots 1 and 2, while leaving 0 unchanged (fixed). The reverse operation, the `from_cycles()` class method, allows us to construct a permutation from a series of cycles:

~~~ python
>>> S3.from_cycles([1, 2], [0]) == a
True
~~~

We can omit *fixed points* from the list of cycles for brevity, if we wish:

~~~ python
>>> S3.from_cycles([1, 2]) == a
True
~~~

Because this notation is more informative to humans, it's the notation used by `repr()` of an element, as we've already seen.

> This is a common convention for all groups: the class constructor usually accepts a "raw" representation of the element (in this case, a permutation tuple) and does little more than wrap it inside the element. The wrapped value is then often accessible under `.value`. In case alternative, more elaborated ways to express an element are required, they're offered as methods (in this case, `cycles()` and `from_cycles()`). `repr()` of an element tends to use these more informative representations where possible.

Finally, `SymmetricGroup` also exposes some other minor operations of interest, such as `sign()` to get the [sign](https://en.wikipedia.org/wiki/Parity_of_a_permutation) of the permutation. Refer to the class for more info.

### Automagical group creation

The module defines a `__getattr__` that will recognize accesses to
certain name patterns and auto-create the group in the module. This
is what allowed us to write:

~~~ python
from groups import S3
~~~

instead of:

~~~ python
from groups import SymmetricGroup

class S3(SymmetricGroup):
	SIZE = 3
~~~

With the added benefit that the created `S3` lives in this module and
the same class will be reused by everyone.

Right now the only supported pattern is PREFIX + NATURAL, with the
following prefixes:

 - `S<n>`: SymmetricGroup
 - `Z<n>`: CyclicGroup

### Composing groups using `DirectProduct`

TODO

### Implementing custom groups

TODO
