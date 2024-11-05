import importlib
import inspect
import pkgutil
from src.core.minedojo_base import MineDojoBase

available_observation = {}

for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, MineDojoBase) and obj is not MineDojoBase:
            available_observation[name] = obj 

# globals().update(available_observation)