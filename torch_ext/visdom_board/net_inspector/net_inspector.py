from typing import Union, List

import torch
from torch import nn

from visdom import Visdom

from ..property import PropertiesManager, Button
from .ui_components import CNNSubmoduleButton, RNNSubmoduleButton, MultipleChoice

from torch_ext.cnn_vis import NetQuery, SubmodulesTree

from .common import *


MULTICHOICE_UID = 'inspection_mode_selection'


class NetInspector:

    def __init__(self, vis: Visdom, model: nn.Module, test_tensor: torch.Tensor):
        self.test_tensor = test_tensor
        self._vis = vis
        self._activations_win = []

        self._vis.close(env=NET_INSPECTOR_ENV)
        self.properties_manager = PropertiesManager(vis, NET_INSPECTOR_ENV)
        self.model = SubmodulesTree(model)
        self.net_query = NetQuery(self.model)
        self._modes = ['weights', 'gradients', 'activations']

        self._init_properties()

    def _init_properties(self) -> None:
        """
        Initializes the UI.
        """
        root_uid = self.model.root()

        multichoice = MultipleChoice(MULTICHOICE_UID, self._modes)
        self.properties_manager.add(multichoice)

        for child_uid in self.model.children(root_uid):
            button = self._build_submodule_button(child_uid)
            self.properties_manager.add(button)

        self.properties_manager.update_property_win()

    def _button_on_click(self, button: Button, module_uid: str) -> None:
        button_state = button.state
        if button_state == Button.State.RELEASED:  # remove children
            button.remove_all_children()
            button.close()

            if hasattr(button, 'init_children'):
                button.init_children()
        else:
            for child_uid in self.model.children(module_uid):
                child_button = self._build_submodule_button(child_uid)
                button.add_child(child_button)

        self.properties_manager.update_property_win()

    def _build_submodule_button(self, module_uid: str) -> Button:
        multichoice = self.properties_manager.get(MULTICHOICE_UID)
        mode = self._modes[multichoice.get_selection()]

        data = self._query_network(mode, module_uid)

        if data is None:
            button = Button(module_uid, module_uid, self._button_on_click)
        else:
            dimensions = len(data.size())
            if dimensions == 5 or dimensions == 3:
                button = RNNSubmoduleButton(module_uid, module_uid, self._button_on_click, data, self._vis)
            elif dimensions == 4 or dimensions == 2:
                button = CNNSubmoduleButton(module_uid, module_uid, self._button_on_click, data, self._vis)
            else:
                raise NotImplementedError('Tensors with a number of dimensions < 2 or > 5 are not supported')

            button.init_children()

        return button

    def _query_network(self, mode: str, module_uid: UID) -> Union[torch.Tensor, List[torch.Tensor]]:
        if mode == 'activations':
            data = self.net_query.get_activations(self.test_tensor, module_uid)
        elif mode == 'gradients':
            raise NotImplementedError()
        elif mode == 'weights':
            raise NotImplementedError()  # TODO: deal with data as list
            data = self.net_query.get_weights(module_uid)
        else:
            ValueError

        return data

    def close(self):
        self.properties_manager.close()
        self._vis.close(env=NET_INSPECTOR_ENV)
