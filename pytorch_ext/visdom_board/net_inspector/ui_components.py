from typing import Callable, Optional

import torch

from ..property import Button, DropdownList
from ..misc import ImageWindow

from .common import *


class CNNSubmoduleButton(Button):
    """
    Button associated to a CNN submodule
    """

    BATCH_SELECT_TXT = 'Select batch element'
    CHANN_SELECT_TXT = 'Select channel'

    def __init__(self, uid: UID, init_value: str, on_update: Callable, data: torch.Tensor, vis):
        """
        :param uid:
        :param init_value: text shown in the button
        :param on_update: update event callback
        :param data: torch.Tensor containing the activations of the output layer of
                     the submodule associated to this Button
        """
        super(RNNSubmoduleButton, self).__init__(uid, init_value, on_update, data)
        self.batch_select_uid = uid + ' batch_select'
        self.chann_select_uid = uid + ' channel_select'
        self.active_windows = []
        self._vis = vis

        self.init_children()

    def init_children(self):
        batch_size = self.data.size(0)
        channels   = self.data.size(1)

        # dropdown list to choose batch element
        batch_elem_list = ['All'] + ['Sample {}'.format(i) for i in range(batch_size)]
        batch_elem_select = DropdownList(uid=self.batch_select_uid, name=RNNSubmoduleButton.BATCH_SELECT_TXT,
                                         values=batch_elem_list, init_value=1, on_update=self.update_shown_activations)

        # dropdown list to choose channel
        channels_list = ['All'] + ['Channel {}'.format(i) for i in range(channels)]
        channel_select = DropdownList(uid=self.chann_select_uid, name=RNNSubmoduleButton.CHANN_SELECT_TXT,
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
        sample_idx  = int(self.children[self.batch_select_uid].value['value']) - 1
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


class RNNSubmoduleButton(Button):
    """
    Button associated to a recurrent submodule
    """

    BATCH_SELECT_TXT = 'Select batch element'
    FRAME_SELECT_TXT = 'Select input frame'
    CHANN_SELECT_TXT = 'Select channel'

    def __init__(self, uid: UID, init_value: str, on_update: Callable, data: torch.Tensor, vis):
        """
        :param uid:
        :param init_value: text shown in the button
        :param on_update: update event callback
        :param data: torch.Tensor containing the activations of the output layer of
                     the submodule associated to this Button
        """
        super(RNNSubmoduleButton, self).__init__(uid, init_value, on_update, data)
        self.batch_select_uid = uid + ' batch_select'
        self.frame_select_uid = uid + ' frame_select'
        self.chann_select_uid = uid + ' channel_select'
        self.active_windows = []
        self._vis = vis

        self.init_children()

    def init_children(self):
        batch_size   = self.data.size(0)
        sequence_len = self.data.size(1)
        channels     = self.data.size(2)

        # dropdown list to choose batch element
        batch_elem_list = ['All'] + ['Sample {}'.format(i) for i in range(batch_size)]
        batch_elem_select = DropdownList(uid=self.batch_select_uid, name=RNNSubmoduleButton.BATCH_SELECT_TXT,
                                         values=batch_elem_list, init_value=1, on_update=self.update_shown_activations)

        # dropdown list to choose frame
        frame_list = ['All'] + ['Frame {}'.format(i) for i in range(sequence_len)]
        frame_select = DropdownList(uid=self.frame_select_uid, name=RNNSubmoduleButton.FRAME_SELECT_TXT,
                                    values=frame_list, init_value=1, on_update=self.update_shown_activations)

        # dropdown list to choose channel
        channels_list = ['All'] + ['Channel {}'.format(i) for i in range(channels)]
        channel_select = DropdownList(uid=self.chann_select_uid, name=RNNSubmoduleButton.CHANN_SELECT_TXT,
                                      values=channels_list, init_value=1, on_update=self.update_shown_activations)

        self.add_child(batch_elem_select)
        self.add_child(frame_select)
        self.add_child(channel_select)

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

    def close_active_windows(self):
        self.close()

    def close(self):
        for win in self.active_windows:
            win.close()
        self.active_windows = []
