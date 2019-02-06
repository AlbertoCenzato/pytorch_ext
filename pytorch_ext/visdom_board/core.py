from typing import Optional

import visdom


class VisObject:

    def __init__(self, vis: visdom.Visdom, env: Optional[str] = None):
        self._vis = vis
        self._win = None
        self._env = env

    def check_connection(self) -> bool:
        return not (self._vis is None)

    # FIXME: this method works only in debug mode, don't know why...
    def close(self):
        if not self.check_connection() or self._win is None:
            return

        self._vis.close(win=self._win, env=self._env)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
