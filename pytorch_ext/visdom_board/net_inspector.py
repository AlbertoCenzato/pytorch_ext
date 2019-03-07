import torch
from torch import nn

from visdom import Visdom

from .misc import ImageWindow
from .property import PropertiesManager, DropdownList, Button

from ..cnn_vis import NetVis, SubmodulesTree


class NetInspector:

    ENV = 'Network Inspector'

    def __init__(self, vis: Visdom, model: nn.Module, test_tensor: torch.Tensor):
        self.test_tensor = test_tensor
        self._vis = vis
        self._activations_win = []

        self._vis.close(env=NetInspector.ENV)
        self.properties_manager = PropertiesManager(vis, NetInspector.ENV)
        self.model = SubmodulesTree(model)
        self.net_vis = NetVis(self.model)
        self.current_modules = set()

        self._init_properties()

    def _init_properties(self) -> None:
        root_uid = self.model.root()
        for uid in self.model.children(root_uid):
            button = Button(uid, self._inspect)
            self.properties_manager.add(uid, button)

        self.properties_manager.update_property_win()

    def _inspect(self, property_value: dict, module_uid: str, button_state: Button.State) -> None:
        if button_state == Button.State.RELEASED:  # remove children
            to_remove = []
            for uid in self.properties_manager:
                if SubmodulesTree.is_parent(module_uid, uid):
                    to_remove.append(uid)
            for uid in to_remove:
                #self.current_modules.remove(key)
                self.properties_manager.remove(uid)
        else:
            for child_uid in self.model.children(module_uid):
                button = Button(child_uid, self._inspect)
                self.properties_manager.add(child_uid, button)

            self._show_layers_activations(module_uid)

        self.properties_manager.update_property_win()

    def _show_layers_activations(self, layer_uid: str) -> None:
        activations = self.net_vis.get_activations(self.test_tensor, layer_uid)

        for win in self._activations_win:
            win.close()

        for layer, activ in activations:
            win = ImageWindow(self._vis, NetInspector.ENV)
            image = activ  # (255*activ).to(torch.uint8)
            win.imshow(image, layer)
            print(layer)
            self._activations_win.append(win)

    def __del__(self):
        self.properties_manager.close()
        self._vis.close(env=NetInspector.ENV)
