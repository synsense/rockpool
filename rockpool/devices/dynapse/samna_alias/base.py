"""
Dynap-SE2 samna alias implementations base class

All se2 alias modules must extend ``SamnaAlias`` class.
It provides from_samna to rockpool and from rockpool to samna conversion utilities.
``SamnaAlias`` methodology heavily depends on serializations of the objects from both sides
The workflow is samna.to_json -> alias.from_json -> play and manipulate inside rockpool -> alias.to_json -> samna.from_json

* Non User Facing *
"""

from __future__ import annotations
from abc import abstractclassmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import json
import logging

SAMNA_AVAILABLE = False

try:
    import samna

    SAMNA_AVAILABLE = True
except:
    samna = Any
    logging.warning(
        "samna installation not found in the environtment! You can still work with simulator to full extend but the objects cannot be converted to device configuration."
    )

    SAMNA_AVAILABLE = False

__all__ = ["SamnaAlias"]


@dataclass
class SamnaAlias:
    """
    SamnaAlias embodies the common samna object utilities that are used by fake samna objects like
    snake to camel case conversion and json constructor
    """

    def samna_object(self, cls: Any) -> Any:
        """
        samna_object converts the samna alias object to a real samna object

        i.e.
        event = NormalGridEvent()
        event.samna_object(samna.dynapse2.NormalGridEvent)

        :param cls: the samna class
        :type cls: Any
        :return: the samna object
        :rtype: Any
        """
        if SAMNA_AVAILABLE:
            obj = cls()
            obj.from_json(self.to_json())
            return obj
        else:
            raise ModuleNotFoundError(
                "samna installation is not found in the environment!"
            )

    def save(self, path: Optional[str] = None) -> None:
        """
        save the file as a json

        :param path: the path to save, default to __class__.__name__.json
        :type path: str
        """
        if path is None:
            path = self.__class__.__name__ + ".json"

        with open(path, "w") as f:
            f.write(self.to_json())

    @abstractclassmethod
    def from_samna(cls, obj: Any) -> SamnaAlias:
        """
        from_samna is a abstract class method, which should be implemented individually

        :param obj: the reciprocal samna object
        :type obj: Any
        :return: the samna alias of the actual samna object
        :rtype: SamnaAlias
        """
        raise NotImplementedError("This method needs per class implementation!")

    @abstractclassmethod
    def to_samna(cls) -> Any:
        """
        to_samna is a abstract class method, which should be implemented individually
        it returns a samna object created by the same data-structure with the alias object
        """
        raise NotImplementedError("This method needs per class implementation!")

    # --- JSON Converter Utils --- #

    def to_json(self) -> str:
        """to_json creates a proper & samna-compatible json string from the samna alias object"""
        return json.dumps({"value0": self.json_wrapper()}, indent="    ")

    def json_wrapper(self) -> str:
        """
        json_wrapper returns a dictionary which is ready to be converted to a json string.
        wrapper stands in an intermediate place in between json and object dictionary
        """
        return self.ctor

    @property
    def ctor(self) -> Dict[str, Any]:
        """
        ctor creates a valid constructor to be able to update a samna object using `from_json()` method
        convert the snake_case named variables to camelCase to be compatible with C++ methods

        :return: a dictionary of the object datastructure
        :rtype: Dict[str, Any]
        """
        return dict(
            map(
                lambda kv: (self.snake_to_camel(kv[0]), kv[1]),
                self.__dict__.items(),
            )
        )

    @staticmethod
    def snake_to_camel(name: str) -> str:
        """
        snake_to_camel converts a snake_case variable name to camelCase variable name

        :param name: the snake_case formatted variable name
        :type name: str
        :return: a camelCase formatted variable name
        :rtype: str
        """

        # __attr_name -> __attrName
        for i, c in enumerate(name):
            if c.isalpha():
                first_index = i
                break

        # Split the rest of the words
        split_list = name[first_index:].split("_")
        name = name[:first_index] + split_list[0]
        name += "".join(word.title() for word in split_list[1:])
        return name

    @staticmethod
    def jlist_alias(__list: List[SamnaAlias]) -> Dict[str, Dict[str, Any]]:
        """
        jlist_alias process the list of samna objects and convert them to a dictionary of json wrappers

        :param __list: a list of samna objects
        :type __list: _type_
        :return: a dictionary of the dictionaries of the object datastructures inside the samna object list
        :rtype: Dict[str, Dict[str, Any]]
        """
        return {f"value{i}": alias.json_wrapper() for (i, alias) in enumerate(__list)}

    @staticmethod
    def jlist_regular(__list: List[Any]) -> Dict[str, Any]:
        """
        jlist_regular process the list of any standard python objects like str, bool, int or so
        then convert them to a dictionary which could be converted to json strings

        :param __list: a list of python object
        :type __list: List[Any]
        :return: a dictionary with enumerated value keys
        :rtype: Dict[str, Any]
        """
        return {f"value{i}": regular for (i, regular) in enumerate(__list)}

    @staticmethod
    def jdict_alias(__dict: Dict[str, SamnaAlias]) -> List[Dict[str, Any]]:
        """
        jdict_alias process the dictionary of samna alias objects (usually parameter maps),
        then convert them into a list of dictionaries of samna compatible entries

        :param __dict: the dictionary of samna alias objects
        :type __dict: Dict[str, SamnaAlias]
        :return: a list of dictionaries
        :rtype: List[Dict[str, Any]]
        """
        return list(
            map(
                lambda __kv: {"key": __kv[0], "value": __kv[1].json_wrapper()},
                __dict.items(),
            )
        )

    @staticmethod
    def jlist_dict_alias(
        __list_dict: List[Dict[str, SamnaAlias]]
    ) -> Dict[List[Dict[str, Any]]]:
        """
        jlist_dict_alias processes the list of samna object dictionaries (usually parameter map lists)
        then convert them into a dictionaries of lists of dictionaries of samna compatible entries

        :param __list_dict: list of samna object dictionaries
        :type __list_dict: List[Dict[str, SamnaAlias]]
        :return: the dictionaries of samna compatible entries
        :rtype: Dict[List[Dict[str, Any]]]
        """
        return SamnaAlias.jlist_regular(
            [SamnaAlias.jdict_alias(d) for d in __list_dict]
        )
