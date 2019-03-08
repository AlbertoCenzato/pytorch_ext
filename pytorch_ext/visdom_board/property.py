import enum
from typing import Callable, List, Dict, Any, Optional

from visdom import Visdom

from .core import VisObject


class Property:

    def __init__(self, property_type: str, name: str, init_value: int, on_update: Optional[Callable]=None):
        """

        :param property_type:
        :param name:
        :param init_value:
        :param on_update: function to be called on property update. The callback receives
                          the property as first argument and its old value as second argument.
        """
        self.value = dict(
            type=property_type,
            name=name,
            value=init_value
        )

        if on_update:
            self.on_update = on_update
        else:
            def noop(prop, old_val):
                pass

            self.on_update = noop

        # FIXME: Property does not have the concept of a uid or a name so this list should contain
        #        Properties not strings. But then how can Property manager access to Properties using their uids?
        self.children: Dict[str, Property] = {}
        self.data: Any = None

    def handle(self, event):
        if event['event_type'] != 'PropertyUpdate':
            return

        old_value = self.value['value']
        self.value['value'] = event['value']
        self.on_update(self, old_value)


class DropdownList(Property):

    def __init__(self, name: str, init_value: int, values: List[str], on_update: Callable):
        super(DropdownList, self).__init__('select', name, init_value, on_update)
        self.value['values'] = values


class Button(Property):

    class State(enum.Enum):
        RELEASED = 0
        PRESSED  = 1

    def __init__(self, init_value: int, on_update: Callable):
        super(Button, self).__init__('button', '', init_value, on_update)
        self.state = Button.State.RELEASED

    def handle(self, event):
        if event['event_type'] != 'PropertyUpdate':
            return

        self.state = Button.State.RELEASED if self.state == Button.State.PRESSED else Button.State.PRESSED

        old_value = self.value['value']
        self.on_update(self, old_value)


class PropertiesManager(VisObject):
    """
    VisObject that manages the properties window.
    """

    def __init__(self, vis: Visdom, env: str='main'):
        super(PropertiesManager, self).__init__(vis, env)
        self._properties: Dict[str, Property] = dict()
        self._cached_properties_list: List[Property] = []

        self._win = vis.properties(self._cached_properties_list, env=env)

        self._vis.register_event_handler(self.dispatcher, self._win)

    def update_property_win(self) -> None:
        """
        Refreshes the UI
        """
        self._cached_properties_list = list(self._properties.values())
        properties = [prop.value for prop in self._cached_properties_list]
        self._vis.properties(properties, win=self._win, env=self._env)

    def dispatcher(self, event: dict) -> None:
        """
        Dispatches the event raised by visdom server on PropertiesManager window
        to the correct property.
        :param event: visdom event
        """
        if event['event_type'] != 'PropertyUpdate':
            return

        prop = self._cached_properties_list[event['propertyId']]
        prop.handle(event)

        self.update_property_win()

    def add(self, property_id: str, property: Property) -> None:
        """
        Add a property to the properties window. To show it call PropertyManager.update_property_win()
        :param property_id: ID associated with 'property'. Every property must have a unique ID.
        :param property:
        """
        self._properties[property_id] = property

    def remove(self, name: str) -> Property:
        """
        Recusively removes the property associated with 'name' and all its children.
        :param name:
        :return: the removed property
        """
        prop = self._properties[name]
        if prop.children:
            for child in prop.children:
                self.remove(child)
        del self._properties[name]
        return prop

    def get(self, name: str) -> Property:
        return self._properties[name]

    def __contains__(self, item) -> bool:
        return item in self._properties

    def __iter__(self):
        return iter(self._properties)
