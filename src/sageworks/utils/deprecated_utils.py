"""Chem/RDKIT/Mordred utilities for Sageworks"""

import logging
from typing import Optional, Union, Callable, Type
from functools import wraps

log = logging.getLogger("sageworks")


def deprecated(version: Optional[str] = None) -> Callable:
    """Decorator to mark classes or functions as deprecated, logging a warning on use.

    Args:
        version (str, optional): The version in which the class or function is deprecated.
    """

    def decorator(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
        message = f"{cls_or_func.__name__} is deprecated"
        if version:
            message += f" and will be removed in version {version}."

        if isinstance(cls_or_func, type):
            # Class decorator
            original_init = cls_or_func.__init__

            @wraps(original_init)
            def new_init(self, *args, **kwargs):
                log.warning(message)
                original_init(self, *args, **kwargs)

            cls_or_func.__init__ = new_init
            return cls_or_func
        else:
            # Function/method decorator
            @wraps(cls_or_func)
            def wrapper(*args, **kwargs):
                log.warning(message)
                return cls_or_func(*args, **kwargs)

            return wrapper

    return decorator


if __name__ == "__main__":
    # Example usage of the deprecated decorator
    @deprecated(version="0.9")
    class OldClass:
        def __init__(self):
            print("OldClass initialized")

    # Using OldClass will log a deprecation warning
    instance = OldClass()

    class MyClass:
        @deprecated(version="0.9")
        def old_method(self):
            print("This is an old method.")

    # Using old_method will log a deprecation warning
    obj = MyClass()
    obj.old_method()
