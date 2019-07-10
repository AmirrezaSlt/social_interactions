import time
import math
import logging
from functools import wraps
from env import logger


def timer(orig_func):
    """Measure runtime of functions"""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = orig_func(*args, **kwargs)
        t = time.time() - t0
        print('{} ran in: {} sec'.format(orig_func.__name__, t))
        return result
    return wrapper


def debug(func):
    """Print input and output of all functions"""
    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


def slow_down(_func=None, *, rate=1):
    """create a delay before calling the function"""
    def decorator_slow_down(func):
        @wraps(func)
        def wrapper_slow_down(*args, **kwargs):
            time.sleep(rate)
            return func(*args, **kwargs)
        return wrapper_slow_down

    if _func is None:
        return decorator_slow_down
    else:
        return decorator_slow_down(_func)


def exception(logger):
    """Exception logger"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                err = "There was an exception in  "
                err += func.__name__
                logger.exception(err)
            raise
        return wrapper
    return decorator


