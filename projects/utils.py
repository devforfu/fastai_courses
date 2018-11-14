import time
from timeit import default_timer


class Timer:
    """Simple util to measure execution time.

    Examples
    --------
    >>> import time
    >>> with Timer() as timer:
    ...     time.sleep(1)
    >>> print(timer)
    00:00:01
    """
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = default_timer() - self.start

    def __float__(self):
        return self.elapsed

    def __str__(self):
        return self.verbose()

    def verbose(self):
        if self.elapsed is None:
            return '<not-measured>'
        return time.strftime('%H:%M:%S', time.gmtime(self.elapsed))
