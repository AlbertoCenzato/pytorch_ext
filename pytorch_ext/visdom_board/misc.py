import os
from typing import Union, List, Optional, Tuple

import numpy as np
from PIL import Image

import tempfile
import subprocess

import torch
import torchvision
import visdom

import plotly.graph_objs as go

from .core import VisObject
from .html_utils import html_progress_bar


# typedef
Tensors = Union[torch.Tensor, List[torch.Tensor]]


class OutputConsole(VisObject):

    def __init__(self, vis: visdom.Visdom, env: Optional[str]=None):
        super(OutputConsole, self).__init__(vis, env)

    def print(self, text: str) -> None:
        if not self.check_connection():     # TODO: find a way to move this check into the base class
            return

        if self._win is None:
            self._win = self._vis.text(text, env=self._env)
        else:
            self._vis.text(text, win=self._win, append=True, env=self._env)

    def clear_console(self) -> None:
        if not self.check_connection():
            return 

        if self._win is None:
            self._win = self._vis.text('', env=self._env)
        else:
            self._vis.text('', win=self._win, append=False, env=self._env)


class ProgressBar(VisObject):

    def __init__(self, vis: visdom.Visdom, env: Optional[str]=None, title: str=''):
        super(ProgressBar, self).__init__(vis, env)
        self._title = title

    def update(self, current_index: int, total: int):
        if not self.check_connection():
            return
        
        if self._win is None:
            self._win = self._vis.text(html_progress_bar(self._title, current_index, total), env=self._env, append=False)
        else:
            self._vis.text(html_progress_bar(self._title, current_index, total), win=self._win, env=self._env, append=False)


class ImageWindow(VisObject):

    def __init__(self, vis: visdom.Visdom, env: str=None):
        super(ImageWindow, self).__init__(vis, env)

    def imshow(self, image: np.array, caption: str='') -> None:
        if not self.check_connection():
            return

        pillow_image = Image.fromarray(image)
        img_width  = pillow_image.width
        img_height = pillow_image.height
        scale_factor = 10

        layout = go.Layout(
            xaxis=go.layout.XAxis(
                visible=False,
                range=[0, img_width*scale_factor]),
            yaxis=go.layout.YAxis(
                visible=False,
                range=[0, img_height*scale_factor],
                scaleanchor='x'),  # the scaleanchor attribute ensures that the aspect ratio stays constant
            width=img_width*scale_factor,
            height=img_height*scale_factor,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            images=[go.layout.Image(
                x=0,
                sizex=img_width*scale_factor,
                y=img_height*scale_factor,
                sizey=img_height*scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                #sizing="stretch",
                source=pillow_image)]
        )
        # we add a scatter trace with data points in opposite corners to give the Autoscale feature a reference point
        fig = go.Figure(data=[{
            'x': [0, img_width*scale_factor], 
            'y': [0, img_height*scale_factor], 
            'mode': 'markers',
            'marker': {'opacity': 0}}], layout=layout)

        if self._win is None:
            self._win = self._vis.plotlyplot(fig, env=self._env)
        else:
            self._vis.plotlyplot(fig, win=self._win, env=self._env)


def video_encode(tensor: torch.Tensor, fps: int) -> Tuple[str, tempfile.TemporaryDirectory]:
    L = tensor.size(0)
    H = tensor.size(1)
    W = tensor.size(2)

    temporary_mp4_dir = tempfile.TemporaryDirectory()
    videofile_path = os.path.join(temporary_mp4_dir.name, 'video.mp4')

    dir = tempfile.TemporaryDirectory()
    # save tensors to png files
    file_prefix = 'frame_'
    file_extension = '.png'
    for t in range(L):
        file_name = ''.join([file_prefix, '{:04d}', file_extension]).format(t)
        file_path = os.path.join(dir.name, file_name)
        torchvision.utils.save_image(tensor[t, :], file_path, normalize=True)

    ffmpeg_file_name_template = ''.join([file_prefix, '%4d', file_extension])
    ffmpeg_file_path = os.path.join(dir.name, ffmpeg_file_name_template)

    command = ['ffmpeg',
               '-loglevel', 'error',
               '-r', '{}'.format(fps),  # frames per second
               '-i', ffmpeg_file_path,
               '-pix_fmt', 'yuv420p',
               '-an',  # Tells FFMPEG not to expect any audio
               '-vcodec', 'h264',
               '-f', 'mp4',
               '-y']  # overwrite
    if H < 240:
        command.append('-vf')
        command.append('scale=-1:240')
    command.append(videofile_path)

    proc = subprocess.run(command, shell=True)
    if proc.returncode != 0:
        videofile_path = ''

    return videofile_path, temporary_mp4_dir


class VideoWindow(VisObject):

    def __init__(self, vis: visdom.Visdom, env: str=None, fps=10):
        super(VideoWindow, self).__init__(vis, env)
        self.fps = fps
        self.opts = dict(fps=fps)

    def play_video(self, video: torch.Tensor) -> None:
        assert video.dim() == 3

        tmp_mp4_file, directory = video_encode(video, self.fps)

        if not tmp_mp4_file:
            self._win = self._vis.text('An error occured while producing the video!')
        else:
            if self._win is None:
                self._win = self._vis.video(videofile=tmp_mp4_file, env=self._env, opts=self.opts)
            else:
                self._vis.video(videofile=tmp_mp4_file, win=self._win, env=self._env, opts=self.opts)



