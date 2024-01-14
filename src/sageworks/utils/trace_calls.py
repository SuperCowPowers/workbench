"""Helper functions for tracing function/method calls"""

import inspect
import logging

# SageWorks Logger
import sageworks  # noqa: F401 (we need to import this to set up the logger)

log = logging.getLogger("sageworks")

# Define ANSI color codes for blue text and class name color
BLUE = "\x1b[94m"
GREEN = "\x1b[92m"
CYAN = "\x1b[96m"
RESET = "\x1b[0m"

exclude_classes = ["Thread", "WSGIRequestHandler", "ThreadedWSGIServer", "Flask", "_NewThreadStartupWithTrace"]


def trace_calls(func):
    def get_call_stack():
        callers = inspect.stack()
        stack = []
        for caller in callers[1:]:
            caller_func = caller.function
            if caller_func != "wrapper" and caller_func != "<module>":
                if "self" in caller.frame.f_locals:
                    class_name = caller.frame.f_locals["self"].__class__.__name__
                    if class_name not in exclude_classes:
                        stack.append(f"{GREEN}{class_name}{RESET}.{caller_func}")
                else:
                    stack.append(caller_func)
            if caller_func == "<module>":
                break
        return stack

    def wrapper(*args, **kwargs):
        if "self" in inspect.signature(func).parameters:
            class_name = args[0].__class__.__name__
            func_name_colored = f"{BLUE}{class_name}.{func.__name__}{RESET}"  # Add color and class name to func name
        else:
            func_name_colored = f"{BLUE}{func.__name__}{RESET}"  # Add color to func name
        call_stack = get_call_stack()
        call_stack.reverse()  # Reverse the stack to show the calling order.
        call_chain = " -> ".join(call_stack)
        log.trace(f"{func_name_colored} called by {call_chain}")
        return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    """Exercise the trace call decorator"""

    class MyClass:
        def my_method(self):
            self.inner_method()

        @trace_calls
        def inner_method(self):
            print("Hello World")

    # Usage
    obj = MyClass()
    obj.my_method()
