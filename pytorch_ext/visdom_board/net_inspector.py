import torch
from torch import nn

from visdom import Visdom

from .misc import ImageWindow
from .property import PropertiesManager, DropdownList, Button

from ..cnn_vis import NetQuery, SubmodulesTree


class NetInspector:

    ENV = 'Network Inspector'

    def __init__(self, vis: Visdom, model: nn.Module, test_tensor: torch.Tensor):
        self.test_tensor = test_tensor
        self._vis = vis
        self._activations_win = []

        self._vis.close(env=NetInspector.ENV)
        self.properties_manager = PropertiesManager(vis, NetInspector.ENV)
        self.model = SubmodulesTree(model)
        self.net_query = NetQuery(self.model)
        self.current_modules = set()

        self._init_properties()

    def _init_properties(self) -> None:
        """
        Initializes the UI.
        """
        root_uid = self.model.root()
        for uid in self.model.children(root_uid):
            self._build_rnn_module_button(uid)

        self.properties_manager.update_property_win()

    def on_click(self, button: Button, module_uid: str) -> None:
        button_state = button.state
        if button_state == Button.State.RELEASED:  # remove children
            for uid in button.children:
                self.properties_manager.remove(uid)

            for win in button.data['active_windows']:
                win.close()
            button.data['active_windows'] = []
        else:
            for child_uid in self.model.children(module_uid):
                self._build_rnn_module_button(child_uid)

        self.properties_manager.update_property_win()

    def _build_rnn_module_button(self, module_uid: str) -> None:

        def child_callback(dropdown_list: DropdownList, old_value: str) -> None:
            """
            notify parent of the update
            """
            parent: Button = dropdown_list.data
            batch_elem_select    = parent.children['Select batch element']
            sequence_elem_select = parent.children['Select input frame']
            channel_select       = parent.children['Select channel']

            sample_idx  = int(batch_elem_select.value['value'])    - 1
            frame_idx   = int(sequence_elem_select.value['value']) - 1
            channel_idx = int(channel_select.value['value'])       - 1

            activations = parent.data['activations']
            samples  = activations.unbind(dim=0) if sample_idx == -1 else [activations[sample_idx, :]]

            # split per frame
            if frame_idx < 0:
                frames = []
                for tensor in samples:
                    for t in tensor.unbind(dim=0):
                        frames.append(t)
            else: 
                frames =[tensor[frame_idx, :] for tensor in samples]

            #split per channel
            if channel_idx < 0:
                channels = []
                for tensor in frames:
                    for t in tensor.split(1, dim=0):
                        channels.append(t)
            else: 
                channels = [tensor[channel_idx, :] for tensor in frames]

            for win in parent.data['active_windows']:
                win.close()
            parent.data['active_windows'] = []

            for image in channels:
                image_win = ImageWindow(self._vis, NetInspector.ENV)
                parent.data['active_windows'].append(image_win)
                image_win.imshow(image, 'activation')

        button = Button(module_uid, self.on_click)
        self.properties_manager.add(module_uid, button)

        activations = self.net_query.get_activations(self.test_tensor, module_uid)
        button.data = {'activations': activations, 'active_windows': []}
        if activations is None:
            return

        batch_size   = activations.size(0)
        sequence_len = activations.size(1)
        channels     = activations.size(2)

        # dropdown list to choose batch element
        batch_elem_list = ['All'] + ['Sample {}'.format(i) for i in range(batch_size)]
        batch_elem_select = DropdownList('Select batch element', 1, batch_elem_list, child_callback)
        batch_elem_select.data = button

        # dropdown list to choose frame
        sequence_elem_list = ['All'] + ['Frame {}'.format(i) for i in range(sequence_len)]
        sequence_elem_select = DropdownList('Select input frame', 1, sequence_elem_list, child_callback)
        sequence_elem_select.data = button

        # dropdown list to choose channel
        channels_list = ['All'] + ['Channel {}'.format(i) for i in range(channels)]
        channel_select = DropdownList('Select channel', 1, channels_list, child_callback)
        channel_select.data = button

        button.children['Select batch element'] = batch_elem_select
        button.children['Select input frame']   = sequence_elem_select
        button.children['Select channel']       = channel_select

        self.properties_manager.add(module_uid + '({})'.format('batch_elem_list'), batch_elem_select)
        self.properties_manager.add(module_uid + '({})'.format('sequence_elem_list'), sequence_elem_select)
        self.properties_manager.add(module_uid + '({})'.format('channels_list'), channel_select)

    def __del__(self):
        self.properties_manager.close()
        self._vis.close(env=NetInspector.ENV)
