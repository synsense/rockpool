"""
property_arrays.py - Collection of array classes to be used as properties to control
                     access to underlying object (e.g. to enable or disable item assignment).
"""

from typing import Optional, Any, Union, Callable
import numpy as np

from rockpool.utilities.type_handling import ArrayLike

# - Try to import PyTorch functions
try:
    from torch import Tensor, from_numpy

except ModuleNotFoundError:
    # - Define dummy class and function
    class Tensor:
        pass

    def from_numpy(o):
        return o


# - Configure exports
__all__ = ["SetterArray", "ImmutableArray", "RefArray", "RefProperty"]


### --- Customized subclasses of np.ndarray


class SetterArray(np.ndarray):
    """
    SetterArray - np.ndarray subclass that can be used for class properties with
                  extended setter methods. It should be returned by property
                  getter. In case of item assignment it will call the
                  corresponding setter function.
    """

    def __new__(
        cls,
        arraylike: ArrayLike,
        owner: Any,
        name: str,
        dtype: Union[type, str, None] = float,
        custom_setter: Optional[Callable] = None,
    ):
        """
        ___new__ - Customize instance creation. Necessary for custom subclasses
                   of np.ndarray. Create new object as view on existing ndarray
                   or on a new ndarray generated from an array-like object.
                   Then add a reference to the owner of the original array-like
                   and store the name of the object to be able to access the
                   property setter.
        :param arraylike:       Array-like object
        :param owner:           Object that owns `arraylike`
        :param name:            Name of `arraylike
        :param dtype:           If not `None`, force data type of array.
        :param custom_setter:   If not `None` replaces the standard setter method.
                                Will receive the updated values of `arraylike`,
                                `owner` and `name` as arguments.
        :return:
            obj  np.ndarray  Numpy array upon which new instance will be based
        """
        # New class instance is a copy of arraylike (and never a view to original arraylike)
        if dtype is None:
            obj = np.array(arraylike).view(cls)
        else:
            obj = np.array(arraylike).astype(dtype).view(cls)
        # Store reference to original arraylike
        obj._reference = arraylike
        obj._owner = owner
        obj._name = name
        obj._custom_setter = custom_setter
        return obj

    def __array_finalize(self, obj: np.ndarray):
        """
        Used for `np.ndarray` subclasses to include additional elements in instance

        :param np.ndarray obj:  np.ndarray upon which self is based
        """
        # - Store reference to third object as attribute of self
        self._reference = getattr(obj, "_reference")
        self._owner = getattr(obj, "_owner")
        self._name = getattr(obj, "_name")
        self._custom_setter = getattr(obj, "_custom_setter")

    def __setitem__(self, position, value):
        """
        ___setitem___ - Update items of self and of self.reference in the same way.
        """
        super().__setitem__(position, value)
        # - Update data in owner
        if self._custom_setter is None:
            setattr(self._owner, self._name, self.copy())
        else:
            self._custom_setter(self.copy(), self._owner, self._name)

    def copy(self):
        """copy - Return np.ndarray as copy to get original __setitem__ method."""
        array_copy = super().copy()
        return np.array(array_copy)


class ImmutableArray(np.ndarray):
    """
    ImmutableArray - np.ndarray subclass that prevents item assignment.
    """

    def __new__(
        cls,
        arraylike: ArrayLike,
        name: Optional[str] = None,
        custom_error: Optional[str] = None,
    ):
        """
        ___new__ - Customize instance creation. Necessary for custom subclasses
                   of np.ndarray. Create new object as view on existing ndarray
                   or on a new ndarray generated from an array-like object.
                   Optionally add the name of the original object or a custom
                   error message to be displayed when item assignment is
                   attempted.
        :param arraylike:       Array-like object or torch tensor to be copied.
        :param name:            Name to be included in default error message.
        :param custom_error:    If not `None`, message to replace default error message.
        :return:
            obj  np.ndarray  Numpy array upon which new instance will be based
        """
        # New class instance is a copy of arraylike (and never a view to original arraylike)
        obj = np.array(arraylike).view(cls)
        obj._name = str(name) if name is not None else None
        obj._custom_error = str(custom_error) if custom_error is not None else None
        return obj

    def __array_finalize(self, obj: np.ndarray):
        """
        __array_finalize - arguments: to be used for np.ndarray subclasses to include
                           additional elements in instance.
        :param obj:  np.ndarray upon which self is based
        """
        # - Store reference to third object as attribute of self
        self._name = getattr(obj, "_name")
        self._custom_error = getattr(obj, "_custom_error")

    def __setitem__(self, position, value):
        """
        ___setitem___ - Update items of self and of self.reference in the same way.
        """
        if self._custom_error is not None:
            raise AttributeError(self._custom_error)
        else:
            raise AttributeError(
                "{}: Item assignment not possible for this attribute.".format(
                    self._name if self._name is not None else "ImmutableArray"
                )
            )

    def copy(self):
        """copy - Return np.ndarray as copy to get original __setitem__ method."""
        array_copy = super().copy()
        return np.array(array_copy)


class RefArray(np.ndarray):
    """
    RefArray - np.ndarray subclass that is generated from an array-like or torch.Tensor
               and contains a reference to the original array-like or to a third object
               with same shape. Item assignment on a RefArray instance (i.e. refarray[i,j]
               = x) will also change this third object accordingly. Typically this object
               is some original container from which the array-like has been created.
               Therefore the objects in the RefArray are typically copies of those in the
               referenced object.
               This is useful for layers that contain torch tensors with properties
               returning a numpy array. Here, item assignment will also modify the original
               tensor object (as would generally be expected), which is not the case when
               using normal ndarrays.
    """

    def __new__(
        cls,
        arraylike: Union[ArrayLike, Tensor],
        reference: Optional[Union[ArrayLike, Tensor]] = None,
    ):
        """
        ___new__ - Customize instance creation. Necessary for custom subclasses of
                   np.ndarray. Create new object as view on existing ndarray or on a new
                   ndarray generated from an array-like object or tensor. Then add a
                   reference to a third object, with same shape. Typically the original
                   array is some form of copy of the referenced object. Alternatively a
                   reference to the original array-like or tensor can be added. In this
                   case the new instance is always a copy of the array-like and not a
                   reference.
        :param arraylike:  Array-like object or torch tensor to be copied.
        :param reference:  Indexable container with same dimensions as arraylike
                           If None, a reference to arraylike will be added.
        :return:
            obj  np.ndarray  Numpy array upon which new instance will be based
        """
        if reference is not None and tuple(np.shape(arraylike)) != tuple(
            np.shape(reference)
        ):
            raise TypeError(
                "Referenced object and array object need to have same shape"
            )
        # - Convert torch tensor to numpy array on cpu
        arraylike_new = (
            arraylike.cpu().numpy() if isinstance(arraylike, Tensor) else arraylike
        )
        if reference is None:
            # New class instance is a copy of arraylike (and never a view to original arraylike)
            obj = np.array(arraylike_new).view(cls)
            # Store reference to original arraylike
            obj._reference = arraylike
        else:
            # New class instance is a copy of original array-like or a view, if arraylike is np.ndarray
            obj = np.asarray(arraylike_new).view(cls)
            # - Add reference to third object
            obj._reference = reference
        return obj

    def __array_finalize(self, obj: np.ndarray):
        """
        __array_finalize - arguments: to be used for np.ndarray subclasses to include
                           additional elements in instance.
        :param obj:  np.ndarray upon which self is based
        """
        # - Store reference to third object as attribute of self
        self._reference = getattr(obj, "_reference", None)

    def __setitem__(self, position, value):
        """
        ___setitem___ - Update items of self and of self.reference in the same way.
        """
        super().__setitem__(position, value)
        if isinstance(self._reference, Tensor):
            if not isinstance(value, Tensor):
                # - Genrate tensor with new data
                value = from_numpy(np.array(value))
            # - Match dtype and device with self.reference
            value = value.to(self._reference.dtype).to(self._reference.device)
        # - Update data in self.reference
        self._reference[position] = value

    def copy(self):
        """copy - Return np.ndarray as copy to get original __setitem__ method."""
        array_copy = super().copy()
        return np.array(array_copy)


class RefProperty(property):
    """
    RefProperty - The purpose of this class is to provide a decorator @RefProperty
                  to be used instead of @property for objects that require that a copy
                  is returned instead of the original object. The returned object is
                  a RefArray with reference to the original object, allowing item
                  assignment to work.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        # - Change fget so that it returns a RefArray
        fget = self.fct_refarray(fget)
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)

    def fct_refarray(self, fct: Callable):
        """
        Return a function that does the same as fct but convert its return value to a RefArray
        :param Callable fct:  Function whose return value should be converted

        """

        def inner(owner):
            original = fct(owner)
            return RefArray(original)

        return inner
