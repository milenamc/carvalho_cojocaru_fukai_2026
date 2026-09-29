import importlib
import inspect
import pkgutil

# List of module names in the package
modules = ['datagraph', 'frames_multi', 'frames_single', 'graph', 'io', 'loaddata', 'plot', 'utils']

# Dynamically import all modules and gather classes and functions
for module_name in modules:
    module = importlib.import_module(f'.{module_name}', package=__name__)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            globals()[name] = obj

# Define __all__ to include all classes and functions from the modules
__all__ = []
for module_name in modules:
    module = importlib.import_module(f'.{module_name}', package=__name__)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            __all__.append(name)