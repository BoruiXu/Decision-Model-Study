"""GPU snapshots (rsi/gpus.py): parsing nvidia-smi's two queries, the owner of each process, and a snapshot that never
raises (nvidia-smi missing, hanging, failing, MIG's [N/A]). No test calls the real nvidia-smi: `run` and `which` are
fakes."""
import functools
import os
import subprocess
import tempfile
import time
import unittest

from rsi import gpus as G
from rsi.gpus import GPU_FILE, SMI_TIMEOUT_S, load_snapshot, read_snapshot, run_bounded, snapshot, write_snapshot

GPUS = ("0, GPU-a, NVIDIA A100, 37000, 81920, 97\n1, GPU-b, NVIDIA A100, 71000, 81920, 80\n"
        "2, GPU-c, NVIDIA A100, [N/A], [N/A], [N/A]\n")
APPS = "GPU-a, 10, 36000\nGPU-b, 20, 70000\nGPU-b, 30, [N/A]\n"
SMI = lambda _name: "/usr/bin/nvidia-smi"  # noqa: E731


class TestSnapshot(unittest.TestCase):
    def test_snapshot_parses_cards_and_owners(self):
        calls = []

        def run(args, **kw):
            self.assertEqual(kw["timeout"], SMI_TIMEOUT_S)
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, GPUS if any("query-gpu" in a for a in args) else APPS, "")

        owners = {10: (os.getuid(), "me"), 20: (4242, "alice")}  # pid 30: unknown
        snap = snapshot(run=run, which=SMI, owner=owners.get)
        self.assertEqual(snap["status"], "ok")
        self.assertIsNone(snap["error"])
        self.assertEqual(snap["uid"], os.getuid())
        self.assertRegex(snap["at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual([a[1:] for a in calls], [
            ["--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            ["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]])
        self.assertEqual([g["index"] for g in snap["gpus"]], [0, 1, 2])
        g0 = snap["gpus"][0]
        self.assertEqual((g0["uuid"], g0["name"], g0["memory_used_mib"], g0["memory_total_mib"], g0["utilization_pct"]),
                         ("GPU-a", "NVIDIA A100", 37000, 81920, 97))
        self.assertEqual(g0["processes"], [{"pid": 10, "memory_mib": 36000, "owner": "ours", "user": "me"}])
        self.assertEqual([(p["owner"], p["user"], p["memory_mib"]) for p in snap["gpus"][1]["processes"]],
                         [("other", "alice", 70000), ("unknown", None, None)])
        g2 = snap["gpus"][2]  # MIG: [N/A] is unknown, never an error
        self.assertEqual((g2["memory_used_mib"], g2["memory_total_mib"], g2["utilization_pct"], g2["processes"]),
                         (None, None, None, []))

    def test_snapshot_survives_missing_hanging_and_mig_nvidia_smi(self):
        # Review focus 3: whatever nvidia-smi does, a snapshot comes back with a status; it never raises.
        snap = snapshot(run=None, which=lambda _name: None)
        self.assertEqual((snap["status"], snap["gpus"]), ("missing", []))

        def hangs(args, **kw):
            raise subprocess.TimeoutExpired(args, kw["timeout"])

        snap = snapshot(run=hangs, which=SMI)
        self.assertEqual(snap["status"], "error")
        self.assertIn("timed out", snap["error"])
        snap = snapshot(run=lambda args, **kw: subprocess.CompletedProcess(args, 9, "", "NVML: driver not loaded\n"),
                        which=SMI)
        self.assertEqual(snap["status"], "error")
        self.assertIn("9", snap["error"])
        self.assertIn("driver not loaded", snap["error"])

        def broken_owner(_pid):
            raise PermissionError("no /proc for you")

        ok = lambda args, **kw: subprocess.CompletedProcess(  # noqa: E731
            args, 0, GPUS if any("query-gpu" in a for a in args) else APPS, "")
        snap = snapshot(run=ok, which=SMI, owner=broken_owner)
        self.assertEqual(snap["status"], "ok")
        self.assertEqual([p["owner"] for g in snap["gpus"] for p in g["processes"]], ["unknown"] * 3)
        # An nvidia-smi stuck in the kernel (D state): it never answers and the kill does not end it. The default run
        # (run_bounded) waits SMI_TIMEOUT_S, then a second after the kill, and gives up on it instead of waiting for
        # it the way subprocess.run does; while it still runs no other nvidia-smi is started.
        self.addCleanup(lambda: G.ABANDONED.__setitem__(slice(None), []))
        started, waits, killed = [], [], []

        class Stuck(object):
            stdout = stderr = None
            pid, ended = 4242, False

            def __init__(self, args, **_kw):
                self.args = args
                started.append(self)

            def communicate(self, timeout=None):
                waits.append(timeout)
                time.sleep(min(timeout, 0.05))
                raise subprocess.TimeoutExpired(self.args, timeout)

            def kill(self):
                killed.append(self.args[0])

            def poll(self):
                return 0 if self.ended else None

        run = functools.partial(run_bounded, popen=Stuck)
        start = time.monotonic()
        snap = snapshot(run=run, which=SMI)
        self.assertLess(time.monotonic() - start, 3)
        self.assertEqual((snap["status"], killed, waits), ("error", ["/usr/bin/nvidia-smi"], [SMI_TIMEOUT_S, 1]))
        self.assertIn("timed out", snap["error"])
        snap = snapshot(run=run, which=SMI)  # the first one still runs: no new one
        self.assertEqual((snap["status"], len(started)), ("error", 1))
        self.assertIn("the previous nvidia-smi has not ended", snap["error"])
        started[0].ended = True  # once it has ended, the next snapshot runs nvidia-smi again
        snapshot(run=run, which=SMI)
        self.assertEqual(len(started), 2)

    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(read_snapshot(root))
            snap = {"at": "2026-11-24T12:00:00Z", "status": "missing", "error": None, "uid": 501, "gpus": []}
            path = write_snapshot(root, snap)
            self.assertEqual(path, os.path.join(root, GPU_FILE))
            self.assertEqual(read_snapshot(root), snap)
            with open(path, "w") as f:
                f.write("{half")
            self.assertIsNone(read_snapshot(root))
            self.assertIn("not valid JSON", load_snapshot(root)[1])  # unreadable, not absent
            os.remove(path)
            self.assertEqual(load_snapshot(root), (None, None))


if __name__ == "__main__":
    unittest.main()
