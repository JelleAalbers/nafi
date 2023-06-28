import importlib

__version__ = '0.0.0'
__all__ = []


submodules = ['utils', 'frequentist', 'bayes', 'interval_finding', 'performance']

for submodule in submodules:
    # Import submodule
    submod = importlib.import_module(f'.{submodule}', package=__name__)
    # Import everything from __all__ in from submodule
    for x in submod.__all__:
        globals()[x] = getattr(submod, x)
    # Add everything from submodule to __all__
    __all__ += submod.__all__
