import enum
from typing import Callable, List, Dict, Any, Optional

from visdom import Visdom

from .core import VisObject


UID = str


class Property:

    class Type(enum.Enum):
        text     = enum.auto()
        number   = enum.auto()
        button   = enum.auto()
        checkbox = enum.auto()
        select   = enum.auto()

    def __init__(self, uid: UID, property_type: enum.Enum, name: str, init_value: str,
                 on_update: Optional[Callable]=None, data: Optional[Any]=None):
        """
        Represents a property to be shown in visdom property window. in addition to property type, property name and
        property value this class has a UID, a callback to be called when a visdom event associated to this property is
        raised and a data field to suit any additional need.
        :param property_type: one of Property.Type
        :param name: property name to display
        :param init_value: property value
        :param on_update: function to be called on property update. The callback receives
                          the property as first argument and its old value as second argument.
        """
        self.uid = uid
        self.value = dict(
            type=property_type.name,
            name=name,
            value=init_value
        )

        if on_update:
            self.on_update = on_update
        else:
            def noop(prop: Property, old_val: str):
                pass

            self.on_update = noop

        self.children: Dict[UID, Property] = {}
        self.data = data

    def handle(self, event) -> None:
        """
        Updates the property value and calls on_update callback
        :param event:
        """
        if event['event_type'] != 'PropertyUpdate':
            return

        old_value = self.value['value']
        self.value['value'] = event['value']
        self.on_update(self, old_value)

    def close(self):
        """
        Override this method to perform any kind of cleanup action
        needed when the property is destroyed.
        """
        pass


class DropdownList(Property):

    def __init__(self, uid: UID, name: str, values: List[str], init_value: int=0,
                 on_update: Optional[Callable]=None, data: Optional[Any]=None):
        super(DropdownList, self).__init__(uid, Property.Type.select, name, init_value, on_update, data)
        self.value['values'] = values


class Button(Property):

    class State(enum.Enum):
        RELEASED = enum.auto()
        PRESSED  = enum.auto()

    def __init__(self, uid: UID, init_value: str, on_update: Optional[Callable]=None, data: Optional[Any]=None):
        super(Button, self).__init__(uid, Property.Type.button, '', init_value, on_update, data)
        self.state = Button.State.RELEASED

    def handle(self, event) -> None:
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
        self._properties: Dict[UID, Property] = dict()

        self._win = vis.properties(list(self._properties.values()), env=env)
        self._vis.register_event_handler(self._dispatcher, self._win)

    def update_property_win(self) -> None:
        """
        Refreshes the UI
        """
        properties = [prop.value for prop in self._properties.values()]
        self._vis.properties(properties, win=self._win, env=self._env)

    def _dispatcher(self, event: dict) -> None:
        """
        Dispatches the event raised by visdom server on PropertiesManager window
        to the correct property.
        :param event: visdom event
        """
        if event['event_type'] != 'PropertyUpdate':
            return

        prop = list(self._properties.values())[event['propertyId']]
        prop.handle(event)

        self.update_property_win()

    def add(self, property: Property) -> None:
        """
        Recursively adds a property and its children to the properties window.
        To show it call PropertyManager.update_property_win().
        :param property_uid: ID associated with 'property'. Every property must have a unique ID.
        :param property:
        """
        self._properties[property.uid] = property
        for child_uid in property.children:
            self.add(property.children[child_uid])

    def remove(self, property_uid: UID) -> Property:
        """
        Recusively removes the property associated with 'name' and all its children.
        :param property_uid:
        :return: the removed property
        """
        prop = self._properties[property_uid]
        if prop.children:
            for child_uid in prop.children:
                self.remove(child_uid)
        del self._properties[property_uid]
        prop.close()
        return prop

    def get(self, property_uid: UID) -> Property:
        return self._properties[property_uid]

    def __contains__(self, property_uid: UID) -> bool:
        return property_uid in self._properties

    def __iter__(self) -> iter:
        return iter(self._properties)
