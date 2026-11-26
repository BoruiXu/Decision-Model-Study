"""A module's progress file (rsi/progress.py, docs/interfaces/progress.md): what is valid is kept, a bad file
never raises, and a live dispatch's old progress is stale."""
import datetime
import json
import os
import tempfile
import unittest

import fixtures
from rsi.progress import PROGRESS_FILE, PROGRESS_STALE_SECONDS, dispatch_progress, read_progress
from rsi.workspace import Workspace

UTC = datetime.timezone.utc


class ProgressCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.path = os.path.join(self.root, PROGRESS_FILE)

    def write(self, obj=None, raw=None):
        with open(self.path, "wb") as f:
            f.write(raw if raw is not None else json.dumps(obj, ensure_ascii=False).encode("utf-8"))


class TestReadProgress(ProgressCase):
    def test_read_progress_keeps_what_is_valid(self):
        now = datetime.datetime(2026, 11, 24, 12, 10, tzinfo=UTC)
        self.write({"version": 1, "updated_at": "2026-11-24T20:08:00+08:00", "summary": "x" * 250,
                    "facts": [{"label": "configs", "value": "3/3"}, {"label": 5}, {"label": "GPU hours", "value": 1.25}],
                    "table": {"columns": ["a", "b"], "rows": [[1], ["x", "y", "z"]]}, "notes": ["n"], "extra": 1})
        r = read_progress(self.path, now)
        self.assertEqual(r["path"], self.path)
        self.assertEqual(r["updated_at"], datetime.datetime(2026, 11, 24, 12, 8, tzinfo=UTC))
        self.assertEqual(r["age_s"], 120)
        self.assertEqual(len(r["summary"]), 201)
        self.assertTrue(r["summary"].endswith("…"))
        self.assertEqual(r["facts"], [("configs", "3/3"), ("GPU hours", "1.25")])
        self.assertEqual(len(r["problems"]), 1)
        self.assertEqual(r["table"], {"title": None, "columns": ["a", "b"], "rows": [["1", None], ["x", "y"]]})
        self.assertEqual(r["notes"], ["n"])
        self.write({"updated_at": "2026-11-24T12:09:00Z", "table": {"title": "t", "columns": ["c"],
                                                                     "rows": [[True], [False], [None], [0.1234567]]}})
        r = read_progress(self.path, now)
        self.assertEqual((r["age_s"], r["problems"]), (60, []))
        self.assertEqual(r["table"]["rows"], [["yes"], ["no"], [None], ["0.123457"]])

    def test_bad_progress_files_never_raise(self):
        # Review Focus 1: half written, not JSON, not an object, not UTF-8, too big, an unknown version.
        self.assertIsNone(read_progress(os.path.join(self.root, "missing.json")))
        big = json.dumps({"updated_at": "2026-11-24T12:00:00Z", "notes": ["x" * 1000] * 300}).encode()
        self.assertGreater(len(big), 256 * 1024)
        for raw in (b"{not json", b"[1]", b"\xff\xfe", big,
                    json.dumps({"version": 2, "updated_at": "2026-11-24T12:00:00Z", "summary": "s"}).encode()):
            self.write(raw=raw)
            r = read_progress(self.path)
            self.assertTrue(r["problems"], raw[:20])
            self.assertIsNone(r["summary"], raw[:20])
            self.assertEqual((r["facts"], r["table"], r["notes"]), ([], None, []), raw[:20])
        self.write({"summary": "s"})
        r = read_progress(self.path)
        self.assertEqual((r["summary"], r["age_s"], r["updated_at"]), ("s", None, None))
        self.assertEqual(len(r["problems"]), 1)
        # values out of range (a time before year 1 in UTC, an integer too big for a float) drop only themselves
        big = "1" + "0" * 400
        self.write(raw=('{"updated_at": "0001-01-01T00:00:00+01:00", "facts": [{"label": "a", "value": %s}], '
                        '"table": {"columns": ["c"], "rows": [[%s]]}, "notes": ["n"]}' % (big, big)).encode())
        r = read_progress(self.path)
        self.assertEqual((r["age_s"], r["facts"], r["table"]["rows"], r["notes"]), (None, [], [[None]], ["n"]))
        self.assertEqual(len(r["problems"]), 3)


class TestDispatchProgress(ProgressCase):
    def test_dispatch_progress_staleness(self):
        ws_dir = os.path.join(self.root, "ws")
        Workspace.init(ws_dir, fixtures.make_task_dir(os.path.join(self.root, "task"), fixtures.binary_task()))
        ws = Workspace(ws_dir)
        marker = os.path.join(ws_dir, "work", "round-1", "train", "watch-1.done")
        did = ws.add_dispatches([{"module": "train", "purpose": "train", "outdir": "work/round-1/train",
                                  "status": "long_job", "marker": marker}])[0]
        self.assertIsNone(dispatch_progress(ws, ws.dispatch(did)))       # no progress.json yet
        now = datetime.datetime(2026, 11, 24, 12, 10, tzinfo=UTC)
        then = now - datetime.timedelta(seconds=PROGRESS_STALE_SECONDS + 1)
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(os.path.join(os.path.dirname(marker), PROGRESS_FILE), "w", encoding="utf-8") as f:
            json.dump({"updated_at": then.strftime("%Y-%m-%dT%H:%M:%SZ"), "summary": "s"}, f)
        p = dispatch_progress(ws, ws.dispatch(did), now)
        self.assertEqual((p["summary"], p["stale"]), ("s", True))
        open(marker, "w").close()                                         # the long job has ended
        self.assertFalse(dispatch_progress(ws, ws.dispatch(did), now)["stale"])
        ws.update_dispatch(did, status="handed_back", marker=None)
        self.assertFalse(dispatch_progress(Workspace(ws_dir), Workspace(ws_dir).dispatch(did), now)["stale"])


if __name__ == "__main__":
    unittest.main()
