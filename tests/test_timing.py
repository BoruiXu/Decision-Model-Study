"""Run and round durations (rsi/timing.py), STATE opened_at, the run report and the progress pages."""
import datetime
import itertools
import json
import os
import re
import unittest
from unittest import mock

import test_dashboard as TD
from rsi import timing as T
from rsi.report import report_markdown
from rsi.workspace import Workspace

ISO = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$"


def iso(dt):
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestFormat(unittest.TestCase):
    def test_format_duration_and_start(self):
        cases = {0: "under a minute", 59: "under a minute", 60: "1 minute", 3599: "59 minutes", 3600: "1 hour",
                 3660: "1 hour 1 minute", 7380: "2 hours 3 minutes", 86399: "23 hours 59 minutes",
                 86400: "1 day 0 hours", 90061: "1 day 1 hour", 180000: "2 days 2 hours"}
        for s, text in cases.items():
            self.assertEqual(T.format_duration(s), text)
        now = datetime.datetime(2026, 11, 22, 15, 0, 0)  # naive: local
        same = iso(datetime.datetime(2026, 11, 22, 13, 51, 7))
        other = iso(datetime.datetime(2026, 11, 20, 9, 5, 0))
        self.assertEqual(T.format_start(same, now), "13:51")
        self.assertEqual(T.format_start(other, now), "11-20 09:05")


class TimingCase(TD.DashboardCase):
    def edit_state(self, fn):
        path = os.path.join(self.ws, "STATE.json")
        with open(path) as f:
            s = json.load(f)
        fn(s)
        with open(path, "w") as f:
            json.dump(s, f)

    def two_rounds(self):
        self.plan_task("ws2")
        self.two_promoted_rounds()
        return Workspace(self.ws)

    def strip_opened(self):
        def fn(s):
            for r in s["rounds"].values():
                r.pop("opened_at", None)
        self.edit_state(fn)


class TestWindows(TimingCase):
    def test_opened_at_and_windows(self):
        ws = Workspace(self.ws)
        self.assertRegex(ws.state["rounds"]["1"]["opened_at"], ISO)
        ticks = itertools.count()
        base = datetime.datetime(2026, 11, 22, tzinfo=datetime.timezone.utc)
        with mock.patch("rsi.workspace.now_iso", side_effect=lambda: iso(base + datetime.timedelta(minutes=next(ticks)))):
            ws = self.two_rounds()
        r1, r2 = ws.state["rounds"]["1"]["opened_at"], ws.state["rounds"]["2"]["opened_at"]
        self.assertRegex(r2, ISO)
        self.assertGreater(T.utc(r2), T.utc(r1))  # a later clock reading, not round 1's
        # The run's and the rounds' windows.
        ws.stop("alice", "enough")
        ws = Workspace(self.ws)
        start, end, approx = T.run_window(ws)
        self.assertEqual(start, ws.state["rounds"]["1"]["opened_at"])
        self.assertEqual(end, ws.state["stopped"]["at"])
        self.assertFalse(approx)
        w = T.round_windows(ws)
        self.assertEqual(w[1]["end"], w[2]["start"])
        self.assertEqual(w[2]["end"], ws.state["stopped"]["at"])
        self.assertFalse(w[1]["approx"] or w[2]["approx"])

    def test_old_workspace_falls_back(self):
        ws = self.two_rounds()
        self.strip_opened()
        ws = Workspace(self.ws)
        start, end, approx = T.run_window(ws)
        self.assertEqual(start, ws.entries("task")[0]["registered_at"])
        self.assertTrue(approx)
        w = T.round_windows(ws)
        self.assertEqual(w[2]["start"], ws.state["rounds"]["1"]["decision"]["at"])
        self.assertTrue(w[2]["approx"])
        later = datetime.datetime.now() + datetime.timedelta(hours=1)
        self.assertIn("running for about 1 hour", T.elapsed_text(ws, later))
        self.assertNotIn("about", T.elapsed_text(ws, datetime.datetime.now()))  # never "about under a minute"
        # The live labels still say the start is approximated, so the page script adds "about" after a minute.
        page = TD.pages(self.ws)[TD.ROUNDS]
        head = page.split("</header>")[0]
        self.assertRegex(head, r'<span class="chip time" data-start="[^"]+" data-prefix="running for " '
                               r'data-suffix="[^"]*" data-approx="1">running for under a minute')
        self.assertIn('data-approx="1">under a minute so far<', TD.round_block(page, 2).split("</summary>")[0])

    def test_elapsed_text(self):
        ws = Workspace(self.ws)
        start = T.utc(ws.state["rounds"]["1"]["opened_at"])
        local = start.astimezone().replace(tzinfo=None)
        now = local + datetime.timedelta(hours=2, minutes=13)
        text = T.elapsed_text(ws, now)
        self.assertTrue(text.startswith("running for 2 hours 13 minutes ("), text)
        self.assertRegex(text, r"\(started \d\d:\d\d\)$" if now.date() == local.date()
                         else r"\(started \d\d-\d\d \d\d:\d\d\)$")
        later = local + datetime.timedelta(days=2, hours=1)
        text = T.elapsed_text(ws, later)
        self.assertIn("2 days", text)
        self.assertRegex(text, r"\d\d-\d\d \d\d:\d\d")
        ws.stop("alice", "done")  # stopped: the whole run
        def fn(s):
            s["rounds"]["1"]["opened_at"] = "2026-11-22T01:00:00Z"
            s["stopped"]["at"] = "2026-11-22T03:48:30Z"
        self.edit_state(fn)
        ws = Workspace(self.ws)
        self.assertEqual(T.elapsed_text(ws, datetime.datetime.now()), "total time: 2 hours 48 minutes")


class TestReport(TimingCase):
    def test_report_shows_durations(self):
        ws = self.two_rounds()
        text = report_markdown(ws)
        self.assertIn("- Elapsed: ", text)
        rows = [l for l in text.splitlines() if l.startswith("| 2 |")]
        self.assertNotIn("in progress", rows[0])  # round 2 is decided: "N minutes so far", as on the rounds page
        self.assertRegex(rows[0], r"\| [^|]*minute[^|]* so far \|$")
        ws.stop("alice", "enough")
        text = report_markdown(Workspace(self.ws))
        self.assertIn("- Total time: ", text)
        self.assertNotIn("- Elapsed: ", text)
        head = [l for l in text.splitlines() if l.startswith("| Round")][0]
        self.assertIn("Elapsed", head)
        self.assertNotIn("in progress", text)
        self.strip_opened()
        self.edit_state(lambda s: s["stopped"].update(at=iso(datetime.datetime.now() + datetime.timedelta(hours=1))))
        self.assertIn("- Total time: about 1 hour", report_markdown(Workspace(self.ws)))
        # The total time counts from the first `rsi run` (its first action in run.jsonl), not from `rsi init` hours before; so
        # does round 1's time.
        self.edit_state(lambda s: (s["rounds"]["1"].update(opened_at="2026-11-22T01:00:00Z"),
                                   s["rounds"]["2"].update(opened_at="2026-11-22T09:00:00Z"),
                                   s["stopped"].update(at="2026-11-22T09:25:00Z")))
        with open(os.path.join(self.ws, "work", "run.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": "2026-11-22T08:20:00Z", "event": "action", "step": 1}) + "\n")
        text = report_markdown(Workspace(self.ws))
        self.assertIn("- Total time: 1 hour 5 minutes (", text)
        self.assertRegex([l for l in text.splitlines() if l.startswith("| 1 |")][0], r"\| 40 minutes \|$")
        # Final review B-1: the pages count the same way (top bar and round 1's chip).
        ws = Workspace(self.ws)
        self.assertEqual(T.elapsed_text(ws, datetime.datetime.now()), "total time: 1 hour 5 minutes")
        self.assertEqual(T.round_windows(ws)[1]["start"], "2026-11-22T08:20:00Z")
        self.assertIn("40 minutes", TD.round_block(TD.pages(self.ws)[TD.ROUNDS], 1).split("</summary>")[0])


class TestReportClock(TimingCase):
    def test_running_report_is_deterministic_and_local(self):
        ws = Workspace(self.ws)
        start = T.utc(ws.state["rounds"]["1"]["opened_at"])
        now = start.astimezone().replace(tzinfo=None) + datetime.timedelta(hours=2, minutes=13)
        text = report_markdown(ws, now)
        self.assertEqual(text, report_markdown(ws, now))
        self.assertIn("- Elapsed: 2 hours 13 minutes (", text)
        self.assertIn("in progress for 2 hours 13 minutes", text)
        Workspace(self.ws).stop("alice", "done")  # the stopped line: local time, full format
        text = report_markdown(Workspace(self.ws))
        line = [l for l in text.splitlines() if l.startswith("- Stopped")][0]
        self.assertNotRegex(line, r"\dT\d\d:\d\d:\d\dZ")
        self.assertRegex(line, r"\d{4}-\d\d-\d\d \d\d:\d\d")

    def test_stopped_without_valid_time_shows_no_live_duration(self):
        Workspace(self.ws).stop("alice", "done")
        for bad in ("garbage", None):
            def fn(s):
                s["stopped"]["at"] = bad
            self.edit_state(fn)
            ws = Workspace(self.ws)
            self.assertEqual(T.elapsed_text(ws, datetime.datetime.now()), "total time: unknown")
            self.assertIn("- Total time: unknown", report_markdown(ws))
            for rel, html in TD.pages(self.ws).items():
                self.assertNotIn('data-start="', html, rel)
            self.assertIn("unknown", TD.round_block(TD.pages(self.ws)[TD.ROUNDS], 1).split("</summary>")[0])


class TestPages(TimingCase):
    def test_header_shows_elapsed(self):
        page = self.overview()
        self.assertIn("running for", page)
        self.assertIn('data-start="', page.split("</header>")[0])
        Workspace(self.ws).stop("alice", "done")
        pgs = TD.pages(self.ws)
        for rel, html in pgs.items():
            self.assertNotIn('data-start="', html, rel)
        self.assertIn("total time: ", pgs[TD.OVERVIEW])

    def test_rounds_page_durations(self):
        s = TD.round_block(TD.pages(self.ws)[TD.ROUNDS], 1).split("</summary>")[0]  # an undecided open round
        self.assertEqual(s.count("in progress"), 1, s)
        self.assertRegex(s, r'data-suffix=" so far"[^>]*>[^<]*minute[^<]* so far<')
        self.two_rounds()
        page = TD.pages(self.ws)[TD.ROUNDS]
        s1 = TD.round_block(page, 1).split("</summary>")[0]
        s2 = TD.round_block(page, 2).split("</summary>")[0]
        self.assertRegex(s1, r">(under a minute|\d+ minutes?)<")
        self.assertNotIn("data-start", s1)
        self.assertNotIn("in progress", s2)  # round 2 is decided: its time says N minutes so far (M12)
        self.assertRegex(s2, r">[^<]*minute[^<]* so far<")
        self.assertIn("data-start=", s2)



if __name__ == "__main__":
    unittest.main()
