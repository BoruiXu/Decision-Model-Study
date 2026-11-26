import os
import unittest

from rsi import schema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs", "interfaces")

MAPPING = {
    "task": "task.md",
    "trainset": "datasets.md", "valset": "datasets.md",
    "evalset": "datasets.md", "testset": "datasets.md",
    "corpus": "datasets.md", "pool": "datasets.md",
    "candidate": "models.md", "search": "models.md",
    "baseline": "compare.md", "comparison": "compare.md",
    "delivery": "delivery.md", "feedback": "delivery.md",
    "evaluation": "evaluation.md", "plan": "evaluation.md",
    "record": "record.md",
    "common": "README.md",
}


class TestDocs(unittest.TestCase):
    def test_every_schema_is_mapped_and_its_fields_documented(self):
        for fn in os.listdir(schema.SCHEMA_DIR):
            if fn.endswith(".json"):
                self.assertIn(fn[:-5], MAPPING)
        checked = 0
        for name, doc in sorted(MAPPING.items()):
            props = schema.load_schema(name).get("properties", {})
            with open(os.path.join(DOCS, doc), encoding="utf-8") as f:
                text = f.read()
            for key in props:
                self.assertIn("`%s`" % key, text, "%s: field %s missing in %s" % (name, key, doc))
            checked += 1
        self.assertEqual(checked, len(MAPPING))


if __name__ == "__main__":
    unittest.main()
