from typing import Optional
import time

import torch
from torch import nn

import visdom

from .plotting import LinePlot, LineStdPlot, HistogramPlot, RibbonPlot, colorscale
from .misc import OutputConsole, ImageWindow, ProgressBar, VideoWindow
from .net_inspector import NetInspector


class VisdomManager:

    MAX_CONNECTION_ATTEMPTS = 10
    DEFAULT_ENV = 'main'

    def __init__(self):
        attempts = 0
        self.vis = None
        while attempts < VisdomManager.MAX_CONNECTION_ATTEMPTS:
            try:
                self.vis = visdom.Visdom(raise_exceptions=True)
            except ConnectionError:
                attempts += 1
                backoff_time = attempts * 0.1
                print('Connection attempt to visdom server failed, waiting {:.0f}ms'.format(backoff_time*1000))
                time.sleep(backoff_time)
                continue
            except NameError:
                print('VisdomBoard WARNING: visdom package not found.')
                attempts = VisdomManager.MAX_CONNECTION_ATTEMPTS   # fail as if unable to connect
            break

        self._connection_available = attempts < VisdomManager.MAX_CONNECTION_ATTEMPTS
        self._current_env = VisdomManager.DEFAULT_ENV

    def is_connection_available(self) -> bool:
        return self._connection_available

    def get_line_plot(self, env: Optional[str]=None, title='', 
                      xaxis='', yaxis='') -> LinePlot:
        if env is None:
            env = self._current_env
        return LinePlot(self.vis, env, title, xaxis, yaxis)

    def get_output_console(self, env: Optional[str]=None) -> OutputConsole:
        if env is None:
            env = self._current_env
        return OutputConsole(self.vis, env)

    def get_histograms(self, title: str='', xlabel: str='', ylabel: str='', 
                       env: Optional[str]=None) -> HistogramPlot:
        if env is None:
            env = self._current_env
        return HistogramPlot(self.vis, env, title, xlabel, ylabel)

    def get_ribbon(self, title: str='', xlabel: str='', ylabel: str='', 
                   env: Optional[str]=None) -> RibbonPlot:
        if env is None:
            env = self._current_env
        return RibbonPlot(self.vis, env, title, xlabel, ylabel)

    def get_line_std_plot(self, title: str='', xlabel: str='', ylabel: str='', 
                          total_traces: int=len(colorscale), env: Optional[str]=None) -> LineStdPlot:
        if env is None:
            env = self._current_env
        return LineStdPlot(self.vis, env, title, xlabel, ylabel, total_traces)

    def get_image_window(self, env: Optional[str]=None) -> ImageWindow:
        if env is None:
            env = self._current_env
        return ImageWindow(self.vis, env)

    def get_video_window(self, env: Optional[str]=None) -> VideoWindow:
        if env is None:
            env = self._current_env
        return VideoWindow(self.vis, env)

    def get_progress_bar(self, env: Optional[str]=None, title: str='') -> ProgressBar:
        if env is None:
            env = self._current_env
        return ProgressBar(self.vis, env, title)

    def get_net_inspector(self, model: nn.Module, test_tensor: torch.Tensor) -> NetInspector:
        return NetInspector(self.vis, model, test_tensor)

    def environment(self, env: str):
        self._current_env = env
        return self

    def close(self, env: str) -> None:
        self.vis.close(win=None, env=env)

    def close_all(self) -> None:
        if not self.is_connection_available():
            return

        env_list = self.vis.get_env_list()
        for env in env_list:
            self.close(env)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._current_env = VisdomManager.DEFAULT_ENV


_visdom_manager = VisdomManager()


def get_visdom_manager():
    return _visdom_manager
