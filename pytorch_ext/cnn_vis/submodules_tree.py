from typing import List, Optional, Dict

from torch import nn


class SubmodulesTree:
    """
    This class wraps a model and has the notion of module-submodule hierarchies.
    In this way it is possible to navigate the submodules tree using meaningful
    UIDs to identify each submodule. Every submodule UID has this recursive
    definition: parent_uid.submodule_class_name_[0-9]+ where the trailing digits
    are used to discriminate submodules of the same type with having the same parent
    """

    class Node:

        def __init__(self, module: nn.Module, parent):
            self.module = module
            self.parent: Optional[SubmodulesTree.Node] = parent
            self.uid = str(self)  # use object pointer as uid until a more meaningful uid is given
            self.children: List[SubmodulesTree.Node] = []

            if self.parent:
                self.parent.children.append(self)

    def __init__(self, model: nn.Module):
        self.model = model
        self.root_node: Optional[SubmodulesTree.Node] = None
        self._nodes: Dict[str, SubmodulesTree.Node] = dict()

        self._build_graph()

    @staticmethod
    def is_parent(parent: str, child: str) -> bool:
        return child.startswith(parent + '.')

    def root(self) -> str:
        return self.root_node.uid

    def children(self, uid: str) -> List[str]:
        if uid in self._nodes:
            return [child.uid for child in self._nodes[uid].children]

        return []

    def parent(self, uid: str) -> Optional[str]:
        if uid in self._nodes:
            parent = self._nodes[uid].parent
            if parent:
                return parent.uid

    def get(self, uid: str) -> Optional[nn.Module]:
        if uid in self._nodes:
            return self._nodes[uid]

    def _assign_uid(self, node: Node) -> None:
        base_uid = node.module.__class__.__name__
        if node.parent:
            base_uid = node.parent.uid + '.' + base_uid

        counter = 0
        uid = base_uid + '_' + str(counter)
        while uid in self._nodes:
            counter += 1
            uid = base_uid + '_' + str(counter)

        node.uid = uid
        node.module.uid = uid

    def _build_graph(self) -> None:
        self.root_node = SubmodulesTree.Node(self.model, None)
        self._assign_uid(self.root_node)
        self._nodes = {self.root_node.uid: self.root_node}
        self._build_graph_recurse(self.root_node)

    def _build_graph_recurse(self, node: Node) -> None:
        for child in node.module.children():
            child_node = SubmodulesTree.Node(child, node)
            self._assign_uid(child_node)
            self._nodes[child_node.uid] = child_node
            self._build_graph_recurse(child_node)
