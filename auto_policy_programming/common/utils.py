import signal

import functools


def timeout(seconds=5, default=None):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            result = func(*args, **kwargs)

            signal.alarm(0)

            return result

        return wrapper

    return decorator