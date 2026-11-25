"""The task's search space and a candidate's config_request (stdlib only).

A task's `search_space` maps configuration paths to what the training-tuning module may try. Its keys are
base, recipe, seed, max_steps, overrides.onejev.<flag> and overrides.lora.<field>; each value is exactly one of

  - a non-empty list of allowed values;
  - {"min": <number>, "max": <number>} (both ends allowed), except for base, recipe and seed (LISTED_PATHS);
  - {"values": [<allowed value>, ...], "dead": {<value, as text>: <why it is a dead end>}}, `dead` optional: notes
    for the module (trsi decides about dead ends); a dead value is still inside the space for registration.

A candidate's `config_request` is the configuration it was trained with, as requested:
{"base", "recipe", "overrides", "seed", "max_steps"}; overrides is {"onejev": {<flag>: value}, "lora": {<field>:
value}} or the same keys flat ({"onejev.lr": value}). Every value that is not null names a path (overrides.onejev.lr
for both spellings). In a task with a search_space (request_errors):

  - a null value is never checked (the recipe's or the train command's default);
  - a path that is a key of the space takes one of the allowed values (the same JSON value: true is not 1) or a
    number in the range;
  - a path that is not a key of the space is not searched: its value must be the default configuration's value at
    that path (the task's resources.onejev.default_config, config_request-shaped, when it gives one), else it is
    refused, naming the path. A task without a default configuration fixes a setting with a one-value list
    (e.g. "base": ["joey-1"]).
"""
import json

from rsi.rules import EPS, is_number

FIXED_PATHS = ("base", "recipe", "seed", "max_steps")
LISTED_PATHS = ("base", "recipe", "seed")  # trsi takes these as lists of values only, never as a range
OVERRIDE_GROUPS = ("onejev", "lora")
REQUEST_KEYS = ("base", "recipe", "overrides", "seed", "max_steps")


def is_path(key):
    """True for a configuration path a search space may name."""
    if key in FIXED_PATHS:
        return True
    parts = key.split(".", 2)
    return len(parts) == 3 and parts[0] == "overrides" and parts[1] in OVERRIDE_GROUPS and bool(parts[2].strip())


def _range(value):
    return isinstance(value, dict) and set(value) == {"min", "max"}


def _listed(value):
    return isinstance(value, dict) and "values" in value and set(value) <= {"values", "dead"}


def allowed_values(value):
    """The allowed values of a search_space entry given as a list or as {"values", "dead"}; None for a range."""
    if isinstance(value, list):
        return value
    return value["values"] if _listed(value) else None


def check_space(space, where="$.search_space"):
    """Why a task's search_space is malformed ([] when it is fine)."""
    if not isinstance(space, dict):
        return ["%s: expected object" % where]
    errors = []
    for key, value in space.items():
        at = "%s.%s" % (where, key)
        if not is_path(key):
            errors.append("%s: %r is no configuration path (one of %s, overrides.onejev.<flag> or "
                          "overrides.lora.<field>)" % (at, key, ", ".join(FIXED_PATHS)))
        elif isinstance(value, list) or _listed(value):
            values = allowed_values(value)
            if not isinstance(values, list) or not values:
                errors.append("%s: the list of allowed values is empty" % at)
            dead = value.get("dead") if isinstance(value, dict) else None
            if dead is not None and not (isinstance(dead, dict) and all(isinstance(v, str) for v in dead.values())):
                errors.append("%s.dead: expected {value: reason}" % at)
        elif _range(value) and key in LISTED_PATHS:
            errors.append("%s: %s takes a non-empty list of allowed values or {\"values\", \"dead\"}, not a range "
                          "{\"min\", \"max\"} (the training-tuning module tries each value)" % (at, key))
        elif _range(value):
            if not (is_number(value["min"]) and is_number(value["max"])):
                errors.append("%s: min and max must be numbers" % at)
            elif value["min"] > value["max"]:
                errors.append("%s: min %s is above max %s" % (at, value["min"], value["max"]))
        else:
            errors.append("%s: expected a non-empty list of allowed values, {\"min\", \"max\"} or "
                          "{\"values\", \"dead\"}" % at)
    return errors


def request_paths(request):
    """[(path, value)] of a config_request: its values that are not null, overrides as overrides.<group>.<name>."""
    out = []
    for key in REQUEST_KEYS:
        value = request.get(key)
        if key != "overrides":
            if value is not None:
                out.append((key, value))
            continue
        for name, v in (value or {}).items():
            if isinstance(v, dict) and name in OVERRIDE_GROUPS:
                out += [("overrides.%s.%s" % (name, k), x) for k, x in v.items() if x is not None]
            elif v is not None:
                out.append(("overrides.%s" % name, v))
    return out


def _same(a, b):
    """The same JSON value (a bool is not a number; 1 and 1.0 are the same number)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if is_number(a) and is_number(b):
        return a == b
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _show(value):
    return json.dumps(value, ensure_ascii=False)


def default_request(task):
    """The task's default configuration as a config_request (resources.onejev.default_config), or None."""
    d = ((task.get("resources") or {}).get("onejev") or {}).get("default_config")
    return d if isinstance(d, dict) else None


def request_errors(request, space, default=None):
    """Why a candidate's config_request is outside the task's search_space ([] when it is inside); `default` is the
    task's default configuration (default_request), whose values stand for a path the space does not name."""
    errors = []
    keys = sorted(space)
    fixed = dict(request_paths(default)) if isinstance(default, dict) else {}
    for path, value in request_paths(request):
        allowed = space.get(path)
        if allowed is None:
            if path in fixed and _same(value, fixed[path]):
                continue
            errors.append("config_request: %s = %s is not in the task's search_space (keys: %s)%s: only the "
                          "configuration space the user approved is searched"
                          % (path, _show(value), ", ".join(keys) or "none",
                             " and not the default configuration's %s" % _show(fixed[path]) if path in fixed else
                             " and the default configuration does not set it"))
        elif allowed_values(allowed) is not None:
            values = allowed_values(allowed)
            if not any(_same(value, a) for a in values):
                errors.append("config_request: %s = %s is not one of the allowed values %s of the task's search_space"
                              % (path, _show(value), _show(values)))
        elif not (is_number(value) and allowed["min"] - EPS <= value <= allowed["max"] + EPS):
            errors.append("config_request: %s = %s is outside the task's search_space range [%s, %s]"
                          % (path, _show(value), allowed["min"], allowed["max"]))
    return errors
