from typing import Optional, Callable, Any
import time

import torch

import visdom

from .plotting import LinePlot, LineStdPlot, HistogramPlot, RibbonPlot, colorscale
from .misc import OutputConsole, ImageWindow, VideoWindow
from .net_inspector import NetInspector


def environment_context(function: Callable) -> Callable:

    def placeholder(self, env, *args, **kwargs) -> Any:
        if env is None:
            env = self._current_env

        return function(self, env, *args, **kwargs)

    return placeholder


class VisdomManager:

    MAX_CONNECTION_ATTEMPTS = 5
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

    @environment_context
    def get_line_plot(self, env: Optional[str]=None, title='', 
                      xaxis='', yaxis='') -> LinePlot:
        # if env is None:
        #     env = self._current_env
        return LinePlot(self.vis, env, title, xaxis, yaxis)

    @environment_context
    def get_output_console(self, env: Optional[str]=None) -> OutputConsole:
        # if env is None:
        #     env = self._current_env
        return OutputConsole(self.vis, env)

    @environment_context
    def get_histograms(self, env: Optional[str] = None, title: str = '',
                       xlabel: str = '', ylabel: str = '') -> HistogramPlot:
        # if env is None:
        #     env = self._current_env
        return HistogramPlot(self.vis, env, title, xlabel, ylabel)

    @environment_context
    def get_ribbon(self, env: Optional[str] = None, title: str = '', xlabel: str = '', ylabel: str = '') -> RibbonPlot:
        # if env is None:
        #     env = self._current_env
        return RibbonPlot(self.vis, env, title, xlabel, ylabel)

    @environment_context
    def get_line_std_plot(self, env: Optional[str] = None, title: str = '', xlabel: str = '',
                          ylabel: str = '', total_traces: int = len(colorscale)) -> LineStdPlot:
        # if env is None:
        #     env = self._current_env
        return LineStdPlot(self.vis, env, title, xlabel, ylabel, total_traces)

    @environment_context
    def get_image_window(self, env: Optional[str]=None) -> ImageWindow:
        # if env is None:
        #     env = self._current_env
        return ImageWindow(self.vis, env)

    @environment_context
    def get_video_window(self, env: Optional[str]=None) -> VideoWindow:
        # if env is None:
        #     env = self._current_env
        return VideoWindow(self.vis, env)

    def get_net_inspector(self, model: torch.nn.Module, test_tensor: torch.Tensor) -> NetInspector:
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
