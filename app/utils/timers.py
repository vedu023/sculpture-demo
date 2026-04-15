import time


class Timer:
    """Context manager that measures elapsed time in milliseconds."""

    def __init__(self):
        self._start: float | None = None
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
