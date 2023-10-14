"""Helper functions for tracing function/method calls"""

import inspect
import logging
from sageworks.utils.sageworks_logging import logging_setup

logging_setup()
log = logging.getLogger("sageworks")


def trace_calls(func):
    def wrapper(*args, **kwargs):
        callers = inspect.stack()
        log.info(f"{func.__name__} was called by {callers[1].function}")
        return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    """Exercise the trace call decorator"""

    class MyClass:
        @trace_calls
        def my_method(self, param):
            print(f"Doing something with {param}")

    # Usage
    obj = MyClass()
    obj.my_method("test")
