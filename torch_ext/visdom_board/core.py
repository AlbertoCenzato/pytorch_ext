from typing import Optional, Callable, Any

import visdom


def check_connection(function: Callable) -> Callable:

    def placeholder(self, *args, **kwargs) -> Any:
        if self._vis:
            return function(self, *args, **kwargs)
        else:
            return

    return placeholder


class VisObject:

    def __init__(self, vis: visdom.Visdom, env: Optional[str] = None):
        self._vis = vis
        self._win = None
        self._env = env

    def check_connection(self) -> bool:
        return not (self._vis is None)

    def close(self):
        if not self.check_connection() or self._win is None:
            return

        self._vis.close(win=self._win, env=self._env)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
