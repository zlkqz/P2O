

import collections
from dataclasses import (
    dataclass,
    field,
    fields,
)
from typing import Any


@dataclass
class BaseConfig(collections.abc.Mapping):
    """The BaseConfig provides omegaconf DictConfig-like interface for a dataclass config.

    The BaseConfig class implements the Mapping Abstract Base Class.
    This allows instances of this class to be used like dictionaries.
    """

    extra: dict[str, Any] = field(default_factory=dict)

    def __setattr__(self, name: str, value):
        if hasattr(self, "_frozen_fields") and name in self._frozen_fields and name in self.__dict__:
            from dataclasses import FrozenInstanceError

            raise FrozenInstanceError(f"Field '{name}' is frozen and cannot be modified")
        super().__setattr__(name, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value associated with the given key. If the key does not exist, return the default value.

        Args:
            key (str): The attribute name to retrieve.
            default (Any, optional): The value to return if the attribute does not exist. Defaults to None.

        Returns:
            Any: The value of the attribute or the default value.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str):
        """Implement the [] operator for the class. Allows accessing attributes like dictionary items.

        Args:
            key (str): The attribute name to retrieve.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
            TypeError: If the key type is not string
        """
        return getattr(self, key)

    def __iter__(self):
        """Implement the iterator protocol. Allows iterating over the attribute names of the instance.

        Yields:
            str: The name of each field in the dataclass.
        """
        for f in fields(self):
            yield f.name

    def __len__(self):
        """
        Return the number of fields in the dataclass.

        Returns:
            int: The number of fields in the dataclass.
        """
        return len(fields(self))
