import os
import inspect
import importlib
from .core import Estimator

ESTIMATOR_REGISTRY = {}

estimator_dir = os.path.dirname(__file__)
for file in os.listdir(estimator_dir):
    path = os.path.join(estimator_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        name = file[:file.find('.py')]
        module = importlib.import_module('estimators.' + name)
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for name, _cls in clsmembers:
        	if issubclass(_cls, Estimator) and not _cls == Estimator:
        		if not hasattr(_cls, 'name'):
        			raise ValueError("All estimators classes must have `name` attribute. Culprit: {}".format(name))
        		else:
        			ESTIMATOR_REGISTRY[_cls.name] = _cls
