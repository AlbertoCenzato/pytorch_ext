import os
from typing import Union, List, Optional, Tuple

import numpy as np

import tempfile
import subprocess

import torch
import torchvision.utils
import visdom

from .core import VisObject, check_connection


# typedef
Tensors = Union[torch.Tensor, List[torch.Tensor]]


class OutputConsole(VisObject):

    def __init__(self, vis: visdom.Visdom, env: Optional[str]=None):
        super(OutputConsole, self).__init__(vis, env)

    @check_connection
    def print(self, text: str) -> None:
        # if not self.check_connection():     # TODO: find a way to move this check into the base class
        #     return

        if self._win is None:
            self._win = self._vis.text(text, env=self._env)
        else:
            self._vis.text(text, win=self._win, append=True, env=self._env)

    @check_connection
    def clear_console(self) -> None:
        # if not self.check_connection():
        #     return

        if self._win is None:
            self._win = self._vis.text('', env=self._env)
        else:
            self._vis.text('', win=self._win, append=False, env=self._env)


class ImageWindow(VisObject):

    def __init__(self, vis: visdom.Visdom, env: str=None):
        super(ImageWindow, self).__init__(vis, env)

    @check_connection
    def imshow(self, image: Union[np.array, torch.Tensor], caption: str='') -> None:
        # if not self.check_connection():
        #     return

        options = {'caption': caption}

        if self._win:
            self._vis.image(image, opts=options, win=self._win, env=self._env)
        else:
            self._win = self._vis.image(image, opts=options, env=self._env)


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

    @check_connection
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



