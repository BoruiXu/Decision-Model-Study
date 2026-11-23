"""JSON Schema subset validator (stdlib only)."""
import json
import math
import os
import re

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")


def load_schema(name):
    with open(os.path.join(SCHEMA_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def _is_type(value, t):
    if t == "null":
        return value is None
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        # json.load accepts NaN and Infinity; a schema number must be finite.
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    if t == "string":
        return isinstance(value, str)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    return False


def validate(instance, schema, path="$"):
    errors = []
    if "type" in schema:
        types = schema["type"]
        if not isinstance(types, list):
            types = [types]
        if not any(_is_type(instance, t) for t in types):
            if isinstance(instance, float) and not math.isfinite(instance):
                return ["%s: %s is not a finite number (expected %s)" % (path, instance, " or ".join(types))]
            return ["%s: expected %s" % (path, " or ".join(types))]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append("%s: not in enum %s" % (path, json.dumps(schema["enum"], ensure_ascii=False)))
    if _is_type(instance, "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append("%s: below minimum %s" % (path, schema["minimum"]))
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append("%s: above maximum %s" % (path, schema["maximum"]))
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append("%s: shorter than minLength %d" % (path, schema["minLength"]))
        if "pattern" in schema and not re.fullmatch(schema["pattern"], instance):
            errors.append("%s: does not match pattern %s" % (path, schema["pattern"]))
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append("%s: fewer than minItems %d" % (path, schema["minItems"]))
        if "items" in schema:
            for i, item in enumerate(instance):
                errors.extend(validate(item, schema["items"], "%s[%d]" % (path, i)))
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append("%s: missing required property '%s'" % (path, key))
        for key, value in instance.items():
            if key in props:
                errors.extend(validate(value, props[key], "%s.%s" % (path, key)))
            elif "additionalProperties" in schema:
                ap = schema["additionalProperties"]
                if ap is False:
                    errors.append("%s: unexpected property '%s'" % (path, key))
                elif isinstance(ap, dict):
                    errors.extend(validate(value, ap, "%s.%s" % (path, key)))
    return errors
