import unittest

from rsi.schema import validate


class TestValidate(unittest.TestCase):
    def test_required_missing(self):
        errs = validate({}, {"type": "object", "required": ["label"]})
        self.assertEqual(len(errs), 1)
        self.assertIn("label", errs[0])

    def test_additional_properties_false(self):
        s = {"type": "object", "properties": {"a": {}}, "additionalProperties": False}
        errs = validate({"a": 1, "b": 2}, s)
        self.assertEqual(len(errs), 1)
        self.assertIn("b", errs[0])

    def test_additional_properties_schema(self):
        s = {"type": "object", "additionalProperties": {"type": "number"}}
        self.assertEqual(validate({"x": 1.5}, s), [])
        self.assertEqual(validate({"x": "no"}, s), ["$.x: expected number"])

    def test_enum(self):
        s = {"enum": ["yes", "no"]}
        self.assertEqual(validate("yes", s), [])
        self.assertEqual(len(validate("maybe", s)), 1)

    def test_nested_path(self):
        s = {"type": "object", "properties": {"a": {"type": "object", "properties": {
            "b": {"type": "array", "items": {"type": "string"}}}}}}
        errs = validate({"a": {"b": [3]}}, s)
        self.assertEqual(errs, ["$.a.b[0]: expected string"])

    def test_bool_is_not_integer(self):
        self.assertEqual(len(validate(True, {"type": "integer"})), 1)
        self.assertEqual(len(validate(False, {"type": "number"})), 1)
        self.assertEqual(validate(3, {"type": "integer"}), [])
        self.assertEqual(validate(3.5, {"type": "number"}), [])
        self.assertEqual(len(validate(3.5, {"type": "integer"})), 1)

    def test_pattern(self):
        s = {"type": "string", "pattern": "[a-z]+-[0-9]{4}"}
        self.assertEqual(validate("trainset-0003", s), [])
        self.assertEqual(len(validate("xtrainset-0003x1", s)), 1)

    def test_nullable_type_list(self):
        s = {"type": ["string", "null"]}
        self.assertEqual(validate(None, s), [])
        self.assertEqual(validate("a", s), [])
        self.assertEqual(len(validate(1, s)), 1)

    def test_bounds_and_lengths(self):
        self.assertEqual(len(validate(0, {"minimum": 1})), 1)
        self.assertEqual(len(validate(5, {"maximum": 4})), 1)
        self.assertEqual(len(validate([], {"minItems": 1})), 1)
        self.assertEqual(len(validate("", {"minLength": 1})), 1)
        self.assertEqual(validate(1, {"minimum": 1, "maximum": 1}), [])

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
