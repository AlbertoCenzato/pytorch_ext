from typing import List, Callable, Tuple

import torch
from torch import nn

from .submodules_tree import SubmodulesTree


class NetVis:

    def __init__(self, model_graph: SubmodulesTree):
        self.model_graph = model_graph
        self.modules: List[Tuple[str, nn.Module]] = []

    def get_modules(self, input: torch.Tensor, module_name: str='') -> List[Tuple[str, nn.Module]]:
        """
        This function calls self.model(input) and traces all the traversed modules
        then returns them in topological order
        :return:
        """
        if len(self.modules) == 0:
            self._store_modules_topological(input)

        if module_name == '':
            return self.modules
        return [module for module in self.modules if module_name in module[0]]

    def get_activations(self, input: torch.Tensor, module_name: str='') -> List[Tuple[str, torch.Tensor]]:
        activations = []

        def store_activations(module, input, output):
            module_name = module.__class__.__name__
            if not torch.is_tensor(output):
                output = output[0]
            batch_elem = output.unbind(dim=0)
            for i, activ in enumerate(batch_elem):
                channels = activ.chunk(activ.size(0))
                for j, activation in enumerate(channels):
                    name = module_name + ' sample_{} ch_{}'.format(i, j)
                    activations.append((name, activation))

        self._apply_hook(store_activations, input, module_name)

        return activations

    def get_weights(self, input: torch.Tensor, module_name: str=''):
        if len(self.modules) == 0:
            self._store_modules_topological(input)

        return [(module[0], module[1].parameters(recurse=False)) for module in self.modules if module_name in module[0]]

    def _apply_hook(self, hook: Callable, input: torch.Tensor, module_uid: str= '') -> None:
        handles = []

        def register_hook(module) -> None:
            if module_uid == module.uid:
                hook_handle = module.register_forward_hook(hook)
                handles.append(hook_handle)

        self.model_graph.model.apply(register_hook)

        self.model_graph.model(input)

        for handle in handles:
            handle.remove()

    def _store_modules_topological(self, input: torch.Tensor) -> None:
        """
        Stores the model submodules in topological order, i.e. from input to
        output following the computation order
        :param input:
        :return:
        """

        def store_module(module, input, output):
            named_module = (module.__class__.__name__, module)
            self.modules.append(named_module)

        self._apply_hook(store_module, input)
