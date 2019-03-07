import enum
from typing import Callable, List, Dict

from visdom import Visdom

from .core import VisObject


class Property:

    def __init__(self, type: str, name: str, init_value: int, on_update: Callable):
        self.value = dict(
            type=type,
            name=name,
            value=init_value
        )
        self.on_update = on_update

    def handle(self, event):
        if event['event_type'] != 'PropertyUpdate':
            return

        old_value = self.value['value']
        self.value['value'] = event['value']
        self.on_update(self.value, old_value)


class DropdownList(Property):

    def __init__(self, name: str, init_value: int, values: List[str], on_update: Callable):
        super(DropdownList, self).__init__('select', name, init_value, on_update)
        self.value['values'] = values


class Button(Property):

    class State(enum.Enum):
        RELEASED = 0
        PRESSED  = 1

    def __init__(self, init_value: str, on_update: Callable):
        super(Button, self).__init__('button', '', init_value, on_update)
        self.state = Button.State.RELEASED

    def handle(self, event):
        if event['event_type'] != 'PropertyUpdate':
            return

        self.state = Button.State.RELEASED if self.state == Button.State.PRESSED else Button.State.PRESSED

        old_value = self.value['value']
        self.on_update(self.value, old_value, self.state)


class PropertiesManager(VisObject):

    def __init__(self, vis: Visdom, env: str='main'):
        super(PropertiesManager, self).__init__(vis, env)
        self._properties: Dict[str, Property] = dict()
        self._cached_properties_list: List[Property] = []

        self._win = vis.properties(self._cached_properties_list, env=env)

        self._vis.register_event_handler(self.dispatcher, self._win)

    def update_property_win(self) -> None:
        self._cached_properties_list = list(self._properties.values())
        properties = [property.value for property in self._cached_properties_list]
        self._vis.properties(properties, win=self._win, env=self._env)

    def dispatcher(self, event: dict) -> None:
        print('dispatcher call')

        if event['event_type'] != 'PropertyUpdate':
            return

        property = self._cached_properties_list[event['propertyId']]
        property.handle(event)

        self.update_property_win()

    def add(self, name: str, property: Property) -> None:
        self._properties[name] = property

    def remove(self, name) -> Property:
        property = self._properties[name]
        del self._properties[name]
        return property

    def __contains__(self, item) -> bool:
        return item in self._properties

    def __iter__(self):
        return iter(self._properties)
