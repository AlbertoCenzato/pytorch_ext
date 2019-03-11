from typing import List

import math

import torch
from torch import nn

from visdom import Visdom

from ..property import PropertiesManager, Button
from .ui_components import CNNSubmoduleButton, RNNSubmoduleButton

from pytorch_ext.cnn_vis import NetQuery, SubmodulesTree

from .common import *


class NetInspector:

    def __init__(self, vis: Visdom, model: nn.Module, test_tensor: torch.Tensor):
        self.test_tensor = test_tensor
        self._vis = vis
        self._activations_win = []

        self._vis.close(env=NET_INSPECTOR_ENV)
        self.properties_manager = PropertiesManager(vis, NET_INSPECTOR_ENV)
        self.model = SubmodulesTree(model)
        self.net_query = NetQuery(self.model)

        self._init_properties()

    def _init_properties(self) -> None:
        """
        Initializes the UI.
        """
        root_uid = self.model.root()
        for uid in self.model.children(root_uid):
            self._build_submodule_button(uid)

        self.properties_manager.update_property_win()

    def _get_children_list(self, module_uid: UID, ignore_builtin_collections=True) -> List[UID]:
        children_list = []
        for child_uid in self.model.children(module_uid):
            if ignore_builtin_collections:
                child_module = self.model.get(child_uid)
                if isinstance(child_module, torch.nn.Sequential) or isinstance(child_module, torch.nn.ModuleList) or \
                   isinstance(child_module, torch.nn.ModuleDict):
                    for subchild_uid in self._get_children_list(child_uid, ignore_builtin_collections):
                        children_list.append(subchild_uid)
            children_list.append(child_uid)

        return children_list

    def _submodule_button_on_click(self, button: Button, module_uid: str) -> None:
        button_state = button.state
        if button_state == Button.State.RELEASED:  # remove children
            for uid in button.children:
                self.properties_manager.remove(uid)
            button.children = dict()

            if hasattr(button, 'close_active_windows'):
                button.close_active_windows()
                button.init_children()
                for child in button.children.values():
                    self.properties_manager.add(child)
        else:
            children_list = self._get_children_list(module_uid)
            for child_uid in children_list:
                child_button = self._build_submodule_button(child_uid)
                button.children[child_uid] = child_button

        self.properties_manager.update_property_win()

    def _build_submodule_button(self, module_uid: str) -> Button:
        activations = self.net_query.get_activations(self.test_tensor, module_uid)
        if activations is None:
            button = Button(module_uid, module_uid, self._submodule_button_on_click)
        else:
            dimensions = len(activations.size())
            if dimensions == 5:
                button = RNNSubmoduleButton(module_uid, module_uid, self._submodule_button_on_click, activations, self._vis)
                button.init_children()
            elif dimensions == 4:
                button = CNNSubmoduleButton(module_uid, module_uid, self._submodule_button_on_click, activations, self._vis)
                button.init_children()
            elif dimensions == 3:
                old_shape = activations.size()
                height = int(math.sqrt(old_shape[2]))
                if old_shape[2] % height != 0:
                    raise NotImplementedError('Only power of two feature sizes are supported for 3D output tensors')
                new_shape = (old_shape[0], old_shape[1], 1, height, height)
                activations = activations.view(new_shape)
                button = RNNSubmoduleButton(module_uid, module_uid, self._submodule_button_on_click, activations, self._vis)
                button.init_children()
            elif dimensions == 2:
                old_shape = activations.size()
                height = int(math.sqrt(old_shape[1]))
                if old_shape[1] % height != 0:
                    raise NotImplementedError('Only power of two feature sizes are supported for 2D output tensors')
                new_shape = (old_shape[0], 1, height, height)
                activations = activations.view(new_shape)
                button = CNNSubmoduleButton(module_uid, module_uid, self._submodule_button_on_click, activations, self._vis)
                button.init_children()
            else:
                raise NotImplementedError('Tensors with a number of dimensions < 2 or > 5 are not supported')
        self.properties_manager.add(button)

        return button

    def close(self):
        self.properties_manager.close()
        self._vis.close(env=NET_INSPECTOR_ENV)