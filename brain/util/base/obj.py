"""Base Object Utility

    This module provides the :class:`BaseObject` and :class:`BaseObjectList` classes, which serve as the fundamental
    building blocks for handling objects and object lists.
"""

# region Imported Dependencies
import pprint
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Generic, TypeVar, Union

# endregion Imported Dependencies


T = TypeVar('T')
"""
A type variable used in generic classes.

`T` represents the element type of the generic class and is typically used in generic classes like 
:class:`BaseObjectList`.
"""


class BaseObject(ABC):
    """ Base Object

        The Base Object is a principle basic object class that has the common features and functionalities in handling
        an object.

        Attributes
            name:
                A :type:`string` that specifies the class object name.
    """

    def __init__(self, a_name: str = 'Object') -> None:
        """ Base Object

            This is a constructor that create an instance of the BaseObject object.

            Args
                a_name:
                    A :type:`string` that specifies the name of the object.

            Returns
                    The constructor does not return any values.
        """
        self.name: str = a_name

    @abstractmethod
    def to_dict(self) -> dict:
        """ To Dictionary

            This method represent the object as a dictionary. The method should be overridden.

            Returns
                dic:
                    A dictionary that contains the object elements.
        """
        NotImplementedError("Subclasses must implement `to_dict`")

    def to_str(self) -> str:
        """ To String

            This method represent the object as a string.

            Returns
                message:
                    A :type:`string` as the object representative.
        """
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """ Represent Instance

            This method represents the object of the class as a string.

            Returns
                message:
                    This method returns a :type:`string` as the representation of the class object.
        """
        return self.to_str()

    def copy(self) -> 'BaseObject':
        """ Copy Instance

            This method copies the object deeply.

            Returns
                The method returns the duplicated object of the class.
        """
        return deepcopy(self)


class BaseObjectList(Generic[T], ABC):
    """Base Object List

    The `BaseObjectList` class represents a list of objects of type `T`.

    Attributes:
        name (str):
            A :type:`string` that specifies the name of the `BaseObjectList` instance.
        _max_size (int):
            An integer representing the maximum size of the list (default is -1, indicating no size limit).
        _items (List[T]):
            A list of objects of type `T` contained within the `BaseObjectList`.
    """

    def __init__(self, a_name: str = 'Objects', a_max_size: int = -1, a_items: List[T] = None):
        """
        Constructor for the `BaseObjectList` class.

        Args:
            a_name (str, optional):
                A :type:`string` that specifies the name of the `BaseObjectList` instance (default is 'Objects').
            a_max_size (int, optional):
                An :type:`int` representing the maximum size of the list (default is -1, indicating no size limit).
            a_items (List[T], optional):
                A list of objects of type :class:`T` to initialize the `BaseObjectList` (default is None).

        Returns:
            None: The constructor does not return any values.
        """
        self.name: str = a_name
        self._max_size: int = a_max_size
        self._items: List[T] = []

        if a_items is not None:
            self.append(a_item=a_items)

    def to_dict(self) -> List[dict]:
        """
        Convert the `BaseObjectList` to a list of dictionaries.

        This method iterates through the objects in the `BaseObjectList` and converts each object to a dictionary.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary represents an object in the `BaseObjectList`.
        """
        dict_items = []
        for item in self._items:
            dict_items.append(item.to_dict())
        return dict_items

    def to_str(self) -> str:
        """
        Convert the `BaseObjectList` to a formatted string.

        This method converts the `BaseObjectList` into a human-readable string representation by
        using the :class:`pprint.pformat` function on the result of `to_dict`.

        Returns:
            str: A formatted string representing the `BaseObjectList`.
        """
        return pprint.pformat(self.to_dict())

    def __repr__(self) -> str:
        """
        Return a string representation of the `BaseObjectList` object.

        This method returns a string representation of the `BaseObjectList` by calling the `to_str` method.

        Returns:
            str: A string representing the `BaseObjectList` object.
        """
        return self.to_str()

    @property
    def items(self) -> List[T]:
        """
        Get the list of items in the `BaseObjectList`.

        This property provides access to the list of items contained within the `BaseObjectList`.

        Returns:
            List[T]: A :class:`list` of objects of type :class:`T` within the `BaseObjectList`.
        """
        return self._items

    def __getitem__(self, a_index: int) -> T:
        return self._items[a_index]

    def __setitem__(self, a_index: int, a_item: T):
        """
           Get an item from the `BaseObjectList` by index.

           This method allows retrieving an item from the `BaseObjectList` by its index.

           Args:
               a_index (:type:int): The index of the item to retrieve.

           Returns:
               T: The item at the specified index.
           """
        self._items[a_index] = a_item

    def append(self, a_item: Union[T, List[T]]):
        """
            Append an item or a list of items to the `BaseObjectList`.

            This method appends an individual item or a list of items to the `BaseObjectList`.

            Args:
                a_item (Union[T, List[T]]): An item or a list of items to append.

            Returns:
                None
        """
        if isinstance(a_item, list):
            for item in a_item:
                self._append_item(item)
        else:
            self._append_item(a_item)

    def _append_item(self, a_item: T) -> None:
        """
            Append an item to the `BaseObjectList` (Internal).

            This internal method appends an item to the `BaseObjectList`, handling size constraints if `_max_size` is set.

            Args:
                a_item (T): The item to append.

            Returns:
                None
        """
        if self._max_size != -1:
            self._items.pop(0) if len(self) >= self._max_size else None
        self._items.append(a_item)

    def __delitem__(self, a_index: int):
        """
            Delete an item from the `BaseObjectList` by index.

            This method allows deleting an item from the `BaseObjectList` by its index.

            Args:
                a_index (int): The index of the item to delete.

            Returns:
                None
        """
        del self._items[a_index]

    def copy(self) -> 'BaseObjectList[T]':
        """
            Create a deep copy of the `BaseObjectList`.

            This method creates a deep copy of the `BaseObjectList`, including a copy of all contained items.

            Returns:
                BaseObjectList[T]: A duplicated instance of the class.
        """
        return deepcopy(self)

    def __len__(self) -> int:
        """
            Get the number of items in the `BaseObjectList`.

            This method returns the number of items contained within the `BaseObjectList`.

            Returns:
                int: The number of items.
        """
        return len(self._items)
