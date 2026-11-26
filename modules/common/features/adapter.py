"""Task adapters: a task's features.adapter file (standard library Python) replaces the generic views for its states.

The file may define any of features(state), edit_view(state), apply(state, changes), repair(state),
check(state) -> [str] and describe() -> str (the constraint text for LLM prompts); every function it leaves out is
the generic one: features = flatten.model_view, edit_view = flatten.edit_view, apply = flatten.set_paths, repair = a
deep copy, check = no errors, describe = "".

Before them an Adapter runs the task's declarative constraints (the task's features.constraints, by the module
features.constraints): repair repairs by them, then calls the adapter's repair; check gives their violations, then the
adapter's.
"""
import copy
import hashlib
import importlib.util
import itertools
import os
import sys

from . import FeatureError
from . import constraints as C
from . import flatten

METHODS = ("features", "edit_view", "apply", "repair", "check", "describe")
_LOADED = itertools.count(1)  # each load gets its own module name, so two adapters never share one


def _repair(state):
    return copy.deepcopy(state)


def _check(state):
    return []


def _describe():
    return ""


GENERIC = {"features": flatten.model_view, "edit_view": flatten.edit_view, "apply": flatten.set_paths,
           "repair": _repair, "check": _check, "describe": _describe}


class Adapter(object):
    """A loaded adapter: `path` (absolute) and `sha256` (of the file's bytes) are None for the generic adapter;
    `module` is the loaded module (None for the generic adapter); `constraints` the task's constraints
    (features.constraints.Constraints, empty without any)."""

    def __init__(self, path=None, sha256=None, module=None, constraints=None):
        self.path, self.sha256, self.module = path, sha256, module
        self.constraints = constraints if isinstance(constraints, C.Constraints) else C.Constraints(constraints)
        self._fn = {}
        for name in METHODS:
            fn = getattr(module, name, None) if module is not None else None
            if fn is not None and not callable(fn):
                raise FeatureError("features adapter %s: %s is not a function" % (path, name))
            self._fn[name] = fn if fn is not None else GENERIC[name]

    def features(self, state):
        return self._fn["features"](state)

    def edit_view(self, state):
        return self._fn["edit_view"](state)

    def apply(self, state, changes):
        return self._fn["apply"](state, changes)

    def repair(self, state):
        """The constraints' repair, then the adapter's."""
        return self._fn["repair"](self.constraints.repair(state))

    def check(self, state):
        """The constraints' violations, then the adapter's."""
        return self.constraints.check(state) + list(self._fn["check"](state))

    def violations(self, state):
        """[(rule, message)] of check: a constraint's rule is its stable text ("nonneg t[].v"); the adapter's own
        messages are their own rule."""
        return self.constraints.violations(state) + [(m, m) for m in self._fn["check"](state)]

    def describe(self):
        return self._fn["describe"]()


def load_adapter(path=None, constraints=None):
    """The adapter in the file at `path`, imported under a module name of its own; the generic adapter when `path`
    is None. Its repair and check run `constraints` (the task's features.constraints, a list) first. FeatureError when
    the file does not exist or is not a .py file, or an item of `constraints` is malformed; an error the file raises
    on import is raised as it is. Callers pass rsi.rules.adapter_path(task) (resolved against the repository root),
    never the task's features.adapter string as written, and rsi.rules.features_settings(task)["constraints"]."""
    constraints = C.Constraints(constraints)
    if path is None:
        return Adapter(constraints=constraints)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FeatureError("features adapter %s: no such file" % path)
    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    name = "_features_adapter_%d" % next(_LOADED)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FeatureError("features adapter %s: not a Python source file (the name must end in .py)" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return Adapter(path, sha256, module, constraints)
