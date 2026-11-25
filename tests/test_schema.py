import unittest

from rsi.schema import validate


class TestValidate(unittest.TestCase):
    def test_keywords(self):
        # (value, schema, the errors: an exact list, or how many with a word each must contain)
        nested = {"type": "object", "properties": {"a": {"type": "object", "properties": {
            "b": {"type": "array", "items": {"type": "string"}}}}}}
        cases = (
            ("required missing", {}, {"type": "object", "required": ["label"]}, (1, "label")),
            ("additionalProperties false", {"a": 1, "b": 2},
             {"type": "object", "properties": {"a": {}}, "additionalProperties": False}, (1, "b")),
            ("additionalProperties schema ok", {"x": 1.5},
             {"type": "object", "additionalProperties": {"type": "number"}}, []),
            ("additionalProperties schema", {"x": "no"},
             {"type": "object", "additionalProperties": {"type": "number"}}, ["$.x: expected number"]),
            ("enum ok", "yes", {"enum": ["yes", "no"]}, []),
            ("enum", "maybe", {"enum": ["yes", "no"]}, (1, "")),
            ("nested path", {"a": {"b": [3]}}, nested, ["$.a.b[0]: expected string"]),
            ("bool is not an integer", True, {"type": "integer"}, (1, "")),
            ("bool is not a number", False, {"type": "number"}, (1, "")),
            ("integer", 3, {"type": "integer"}, []),
            ("number", 3.5, {"type": "number"}, []),
            ("float is not an integer", 3.5, {"type": "integer"}, (1, "")),
            ("pattern ok", "trainset-0003", {"type": "string", "pattern": "[a-z]+-[0-9]{4}"}, []),
            ("pattern is a full match", "xtrainset-0003x1", {"type": "string", "pattern": "[a-z]+-[0-9]{4}"}, (1, "")),
            ("nullable null", None, {"type": ["string", "null"]}, []),
            ("nullable string", "a", {"type": ["string", "null"]}, []),
            ("nullable wrong type", 1, {"type": ["string", "null"]}, (1, "")),
            ("minimum", 0, {"minimum": 1}, (1, "")),
            ("maximum", 5, {"maximum": 4}, (1, "")),
            ("minItems", [], {"minItems": 1}, (1, "")),
            ("minLength", "", {"minLength": 1}, (1, "")),
            ("bounds inclusive", 1, {"minimum": 1, "maximum": 1}, []),
        )
        for name, value, schema, expect in cases:
            with self.subTest(name):
                errs = validate(value, schema)
                if isinstance(expect, list):
                    self.assertEqual(errs, expect)
                else:
                    self.assertEqual(len(errs), expect[0], errs)
                    self.assertIn(expect[1], errs[0])

    def test_number_rejects_nan_and_infinity(self):
        # M4: json.load accepts NaN and Infinity; the number type does not.
        for bad in (float("nan"), float("inf"), float("-inf")):
            errs = validate(bad, {"type": "number", "minimum": 0})
            self.assertEqual(len(errs), 1, errs)
            self.assertIn("finite", errs[0])
            self.assertEqual(len(validate(bad, {"type": ["number", "null"]})), 1)
        self.assertEqual(len(validate({"x": float("nan")}, {"type": "object", "additionalProperties":
                                                             {"type": "number"}})), 1)
        self.assertEqual(validate(1e308, {"type": "number"}), [])

    def test_pure(self):
        s = {"type": "object", "required": ["a"]}
        validate({}, s)
        self.assertEqual(s, {"type": "object", "required": ["a"]})


if __name__ == "__main__":
    unittest.main()
