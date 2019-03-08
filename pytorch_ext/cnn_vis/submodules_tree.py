from typing import List, Optional, Dict

from torch import nn


UID = str


# TODO: add the option to navigate the tree as a computational graph.
#       Every node should have 'prev' and 'next' attributes respectively
#       indicating which is the predecessor and successor node in the computation
class SubmodulesTree:
    """
    This class wraps a model and has the notion of module-submodule hierarchies.
    In this way it is possible to navigate the submodules tree using meaningful
    UIDs to identify each submodule. Every submodule UID has this recursive
    definition: parent_uid.submodule_class_name_[0-9]+ where the trailing digits
    are used to discriminate submodules of the same type with having the same parent
    """

    class _Node:
        """
        A tree node. This is intended to be a PRIVATE inner class of SubmodulesTree.
        Stores the module, has a UID to be identified. It can be used by the tree
        to navigate either 'up' or 'down'.
        """

        def __init__(self, module: nn.Module, parent):
            """
            Builds a node. UID defaults to the object address in memory;
            children is initialized as an empty list. Self is automatically
            added as child in parent.children list.
            :param module: the module stored by the node
            :param parent: the parent Node
            """
            self.module = module
            self.parent: Optional[SubmodulesTree._Node] = parent
            self.uid = UID(self)  # use object pointer as uid until a more meaningful uid is given
            self.children: List[SubmodulesTree._Node] = []

            if self.parent:
                self.parent.children.append(self)

    def __init__(self, model: nn.Module):
        """
        Builds a SubmoduleTree based on 'model'. An UID is recursively added
        to each nn.Module in 'model' and is accessible trough the .uid attribute.
        :param model: a PyTorch torch.nn.Module
        """
        self.model = model
        self.root_node: Optional[SubmodulesTree._Node] = None
        self._nodes: Dict[UID, SubmodulesTree._Node] = dict()

        self._build_graph()

    @staticmethod
    def is_parent(parent: UID, child: UID) -> bool:
        """
        Determines if 'parent' and 'child' have a parent-child relationship
        based on their UIDs.
        :param parent:
        :param child:
        :return:
        """
        return child.startswith(parent + '.')

    def root(self) -> UID:
        """
        :return: UID of the root module
        """
        return self.root_node.uid

    def children(self, uid: UID) -> List[UID]:
        """
        :param uid: uid of the parent module.
        :return: list of UIDs of submodules. Returns None
                 if uid is not a module's UID for this tree.
        """
        if uid in self._nodes:
            return [child.uid for child in self._nodes[uid].children]

        return []

    def parent(self, uid: UID) -> Optional[UID]:
        """
        :param uid: UID of the child submodule.
        :return: UID of the parent module. Return None if uid == self.root()
                 or uid is not a module's UID for this tree.
        """
        if uid in self._nodes:
            parent = self._nodes[uid].parent
            if parent:
                return parent.uid

    def get(self, uid: UID) -> Optional[nn.Module]:
        """
        :param uid:
        :return: the module associated with UID or None if uid is not
                 a module's UID for this tree.
        """
        if uid in self._nodes:
            return self._nodes[uid].module

    def _assign_uid(self, node: _Node) -> None:
        """
        Assigns a readable uid to node. Each UID is recursively composed as:
        <parent.uid>.<node_name>_[0-9]+ where the trailing digits are used to
        discriminate siblings of the same type.
        :param node:
        """
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
        self.root_node = SubmodulesTree._Node(self.model, None)
        self._assign_uid(self.root_node)
        self._nodes = {self.root_node.uid: self.root_node}
        self._build_graph_recurse(self.root_node)

    def _build_graph_recurse(self, node: _Node) -> None:
        for child in node.module.children():
            child_node = SubmodulesTree._Node(child, node)
            self._assign_uid(child_node)
            self._nodes[child_node.uid] = child_node
            self._build_graph_recurse(child_node)
