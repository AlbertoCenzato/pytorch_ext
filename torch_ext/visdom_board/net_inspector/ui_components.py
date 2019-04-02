from typing import Callable, Optional, List, Any
import math

import torch

from ..property import Button, DropdownList, PropertyContainer, Checkbox
from ..misc import ImageWindow

from .common import *


BATCH_SELECT_TXT = 'Select batch element'
FRAME_SELECT_TXT = 'Select input frame'
CHANN_SELECT_TXT = 'Select channel'


class CNNSubmoduleButton(Button):
    """
    Button associated to a CNN submodule
    """

    def __init__(self, uid: UID, init_value: str, on_update: Callable, data: torch.Tensor, vis):
        """
        :param uid:
        :param init_value: text shown in the button
        :param on_update: update event callback
        :param data: torch.Tensor containing the activations of the output layer of
                     the submodule associated to this Button
        """
        super(CNNSubmoduleButton, self).__init__(uid, init_value, on_update, data)
        self.data = self.check_data(data)
        self.batch_select_uid = uid + '.batch_select'
        self.chann_select_uid = uid + '.channel_select'
        self.active_windows = []
        self._vis = vis

        self.init_children()

    def init_children(self):
        batch_size = self.data.size(0)
        channels   = self.data.size(-3)

        # dropdown list to choose batch element
        batch_elem_list = ['All'] + ['Sample {}'.format(i) for i in range(batch_size)]
        batch_elem_select = DropdownList(uid=self.batch_select_uid, name=BATCH_SELECT_TXT,
                                         values=batch_elem_list, init_value=1, on_update=self.update_shown_activations)

        # dropdown list to choose channel
        channels_list = ['All'] + ['Channel {}'.format(i) for i in range(channels)]
        channel_select = DropdownList(uid=self.chann_select_uid, name=CHANN_SELECT_TXT,
                                      values=channels_list, init_value=1, on_update=self.update_shown_activations)

        self.add_child(batch_elem_select)
        self.add_child(channel_select)

    def update_shown_activations(self, dropdown_list: Optional[DropdownList] = None,
                                 old_value: Optional[str] = None) -> None:
        """
        Updates activations based on the children dropdown list selections. This method is called automatically
        when a child dropdown list is updated.
        :param dropdown_list: placeholder param to match on_update function signature, can be leaved empty
        :param old_value: placeholder param to match on_update function signature, can be leaved empty
        """
        sample_idx = int(self.children[self.batch_select_uid].value['value']) - 1
        channel_idx = int(self.children[self.chann_select_uid].value['value']) - 1

        activations = self.data
        samples = activations.unbind(dim=0) if sample_idx == -1 else [activations[sample_idx, :]]

        # split per channel
        if channel_idx < 0:
            channels = []
            for tensor in samples:
                for t in tensor.split(1, dim=0):
                    channels.append(t)
        else:
            channels = [tensor[channel_idx, :] for tensor in samples]

        self.close_active_windows()

        for image in channels:
            image_win = ImageWindow(self._vis, NET_INSPECTOR_ENV)
            self.active_windows.append(image_win)
            image_win.imshow(image, 'activation')

    def close_active_windows(self):
        self.close()

    def close(self):
        for win in self.active_windows:
            win.close()
        self.active_windows = []

    def check_data(self, data: torch.Tensor) -> torch.Tensor:
        dimensions = len(data.size())
        if dimensions != 2 and dimensions != 4:
            raise ValueError('data must have either shape (height, width) or (batch, channels, height, width)')

        if dimensions == 2:
            old_shape = data.size()
            height = int(math.sqrt(old_shape[1]))
            if old_shape[1] % height != 0:
                new_shape = (old_shape[0], 1, 16, -1)
            else:
                new_shape = (old_shape[0], 1, height, height)
            reshaped_data = data.view(new_shape)
        else:
            reshaped_data = data

        return reshaped_data


class RNNSubmoduleButton(CNNSubmoduleButton):
    """
    Button associated to a recurrent submodule
    """

    def init_children(self):
        super(RNNSubmoduleButton, self).init_children()
        sequence_len = self.data.size(1)

        # dropdown list to choose frame
        frame_list = ['All'] + ['Frame {}'.format(i) for i in range(sequence_len)]
        frame_select = DropdownList(uid=self.frame_select_uid, name=FRAME_SELECT_TXT,
                                    values=frame_list, init_value=1, on_update=self.update_shown_activations)

        self.add_child(frame_select)

    def update_shown_activations(self, dropdown_list: Optional[DropdownList]=None,
                                 old_value: Optional[str]=None) -> None:
        """
        Updates activations based on the children dropdown list selections. This method is called automatically
        when a child dropdown list is updated.
        :param dropdown_list: placeholder param to match on_update function signature, can be leaved empty
        :param old_value: placeholder param to match on_update function signature, can be leaved empty
        """
        sample_idx  = int(self.children[self.batch_select_uid].value['value']) - 1
        frame_idx   = int(self.children[self.frame_select_uid].value['value']) - 1
        channel_idx = int(self.children[self.chann_select_uid].value['value']) - 1

        activations = self.data
        samples = activations.unbind(dim=0) if sample_idx == -1 else [activations[sample_idx, :]]

        # split per frame
        if frame_idx < 0:
            frames = []
            for tensor in samples:
                for t in tensor.unbind(dim=0):
                    frames.append(t)
        else:
            frames = [tensor[frame_idx, :] for tensor in samples]

        # split per channel
        if channel_idx < 0:
            channels = []
            for tensor in frames:
                for t in tensor.split(1, dim=0):
                    channels.append(t)
        else:
            channels = [tensor[channel_idx, :] for tensor in frames]

        self.close_active_windows()

        for image in channels:
            image_win = ImageWindow(self._vis, NET_INSPECTOR_ENV)
            self.active_windows.append(image_win)
            image_win.imshow(image, 'activation')

    def check_data(self, data: torch.Tensor) -> torch.Tensor:
        dimensions = len(data.size())
        if dimensions != 3 and dimensions != 5:
            raise ValueError('data must have either shape (batch, sequence_len, features) or (batch, sequence_len, channels, height, width)')

        if dimensions == 3:
            old_shape = data.size()
            height = int(math.sqrt(old_shape[2]))
            if old_shape[2] % height != 0:
                new_shape = (old_shape[0], old_shape[1], 1, 16, -1)
            else:
                new_shape = (old_shape[0], old_shape[1], 1, height, height)
            reshaped_data = data.view(new_shape)
        else:
            reshaped_data = data

        return reshaped_data


class MultipleChoice(PropertyContainer):

    def __init__(self, uid: UID, choices: List[str], on_update: Optional[Callable]=None, data: Optional[Any]=None):
        super(MultipleChoice, self).__init__(uid, data=data)
        self._init_children(choices)
        self._selection = 0
        self._on_update = on_update

    def _init_children(self, choices: List[str]) -> None:
        for i, choice in enumerate(choices):
            init_value = (i == 0)
            checkbox = Checkbox(
                uid='{}.{}'.format(self.uid, choice),
                name=choice,
                init_value=init_value,
                on_update=self.callback,
                data=i
            )
            self.add_child(checkbox)

    def callback(self, checkbox: Checkbox, old_value: bool) -> None:
        for i, child in enumerate(self.children.values()):
            child.value['value'] = i == checkbox.data

        old_selection = self.get_selection()
        self._selection = checkbox.data

        self._on_update(self, old_selection)

    def get_selection(self):
        return self._selection
