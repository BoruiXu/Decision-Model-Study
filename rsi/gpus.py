"""GPU snapshots for the pages: <ws>/work/gpus.json.

snapshot() runs nvidia-smi twice (the cards, then the compute processes; each call at most SMI_TIMEOUT_S seconds) and
returns every card with its processes, each process marked ours (its owner's uid is this process's), other (with the
owner's user name) or unknown (no owner found: a process of another pid namespace, or one gone meanwhile). It never
raises: without nvidia-smi the status is "missing"; a call that fails, hangs or prints something else is "error" with
the reason; values like [N/A] (MIG) are None. Every call is bounded (run_bounded: a child that will not die after
its timeout is left behind, never waited for). Only the engine's refresh while it waits takes a snapshot
(rsi.engine.Engine.tick) and writes it with write_snapshot; the pages and every other command only read the latest one
(read_snapshot; load_snapshot also says why a file there cannot be read) and never run nvidia-smi. The framework shows the cards; it never picks one or touches a process.
"""
import datetime
import json
import os
import shutil
import subprocess
import tempfile

try:
    import pwd
except ImportError:  # not POSIX: no user names
    pwd = None

GPU_FILE = os.path.join("work", "gpus.json")
SMI_TIMEOUT_S = 5
QUERY_GPUS = ("--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits")
QUERY_APPS = ("--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits")
# The children run_bounded gave up on (still there a second after the kill: e.g. an nvidia-smi stuck in a driver that
# hangs). While an nvidia-smi among them still runs, snapshot starts no other one: a driver that never recovers would
# otherwise leave one more hung process, and a 6-second wait, behind every refresh.
ABANDONED = []


def _int(text):
    """A whole number from nvidia-smi's csv (37000, 97, 81920.0), or None for [N/A], [Not Supported] and the like."""
    try:
        return int(float(text))
    except (TypeError, ValueError, OverflowError):
        return None


def _rows(text, width):
    """The csv lines of `text` with at least `width` comma-separated fields, each stripped."""
    out = []
    for line in (text or "").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= width:
            out.append(cells)
    return out


def _user(uid):
    if pwd is None:
        return None
    try:
        return pwd.getpwuid(uid).pw_name
    except (KeyError, OverflowError):
        return None


def run_bounded(args, timeout=SMI_TIMEOUT_S, popen=subprocess.Popen, env=None, **_kw):
    """subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env) that never hangs: subprocess.run
    kills a child that timed out and then waits for it without a limit, so a child stuck in the kernel (an nvidia-smi
    in D state) would block the engine's wait for good. Here a child that does not die within a second of the kill is
    left behind (its pipes closed, kept in ABANDONED) and TimeoutExpired raised all the same. `popen` stands for
    subprocess.Popen. rsi.runners.base.process_info runs `ps` with it too (the engine's wait for agents)."""
    p = popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
              env=env)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            p.kill()
            p.communicate(timeout=1)
        except Exception:  # still there after the kill (or already gone): leave it
            ABANDONED.append(p)
            for f in (p.stdout, p.stderr):
                try:
                    if f is not None:
                        f.close()
                except Exception:
                    pass
        raise subprocess.TimeoutExpired(args, timeout)
    return subprocess.CompletedProcess(args, p.returncode, out, err)


def still_running(name):
    """The children run_bounded abandoned whose program is `name` (e.g. nvidia-smi) and that still run; those that
    have ended are dropped from ABANDONED (poll reaps them)."""
    alive = []
    for p in ABANDONED:
        try:
            if p.poll() is None:
                alive.append(p)
        except Exception:  # cannot tell: count it as ended
            pass
    ABANDONED[:] = alive
    return [p for p in alive if os.path.basename(str((getattr(p, "args", None) or [""])[0])) == name]


def proc_owner(pid, run=run_bounded):
    """(uid, user name or None) of process `pid`, or None when it cannot be found: the owner of /proc/<pid>; on a
    machine without /proc `ps -o uid= -p <pid>`."""
    if os.path.isdir("/proc"):
        try:
            uid = os.stat("/proc/%d" % pid).st_uid
        except OSError:
            return None
    else:
        r = run(["ps", "-o", "uid=", "-p", str(pid)], capture_output=True, text=True, timeout=SMI_TIMEOUT_S)
        uid = _int((r.stdout or "").strip()) if r.returncode == 0 else None
        if uid is None:
            return None
    return uid, _user(uid)


def _smi(run, smi, query):
    """stdout of `nvidia-smi <query>`; raises RuntimeError with the reason when it fails or hangs."""
    try:
        r = run([smi] + list(query), capture_output=True, text=True, timeout=SMI_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out (no return in %d seconds)" % SMI_TIMEOUT_S)
    except OSError as e:
        raise RuntimeError("nvidia-smi cannot run: %s" % (e.strerror or e))
    if r.returncode != 0:
        why = ((r.stderr or "").strip() or (r.stdout or "").strip()).splitlines()
        raise RuntimeError("nvidia-smi exit code %s%s" % (r.returncode, ": %s" % why[0][:200] if why else ""))
    return r.stdout or ""


def snapshot(run=run_bounded, which=shutil.which, owner=None):
    """Every card nvidia-smi shows, now: {"at" (UTC ISO, Z), "status" ("ok", "missing" or "error"), "error" (str or
    None), "uid" (this process's), "gpus": [{"index", "uuid", "name", "memory_used_mib", "memory_total_mib",
    "utilization_pct", "processes": [{"pid", "memory_mib", "owner" ("ours", "other", "unknown"), "user"}]}]}. `run`
    (default run_bounded, called like subprocess.run) and `which` (shutil.which) run and find nvidia-smi; `owner(pid)`
    gives (uid, user name) or None (default proc_owner, through `run`). Never raises and never hangs; on an error the
    cards read before it are kept. While an nvidia-smi run_bounded gave up on still runs, no new one is started: the
    status is "error" ("the previous nvidia-smi has not ended")."""
    getuid = getattr(os, "getuid", None)
    snap = {"at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "ok",
            "error": None, "uid": getuid() if getuid else None, "gpus": []}
    try:
        smi = which("nvidia-smi")
        if not smi:
            snap["status"] = "missing"
            return snap
        stuck = still_running("nvidia-smi")
        if stuck:  # the driver hangs: starting another one would only hang too
            snap["status"] = "error"
            snap["error"] = ("the previous nvidia-smi has not ended (process %s cannot be killed; it may hang in the "
                             "driver); not run again this time" % (getattr(stuck[0], "pid", None) or "?"))
            return snap
        for cells in _rows(_smi(run, smi, QUERY_GPUS), 6):
            index = _int(cells[0])
            if index is None:
                continue
            snap["gpus"].append({"index": index, "uuid": cells[1], "name": ", ".join(cells[2:-3]),
                                 "memory_used_mib": _int(cells[-3]), "memory_total_mib": _int(cells[-2]),
                                 "utilization_pct": _int(cells[-1]), "processes": []})
        cards = dict((g["uuid"], g) for g in snap["gpus"])
        owner = owner or (lambda pid: proc_owner(pid, run))
        for cells in _rows(_smi(run, smi, QUERY_APPS), 3):
            pid, card = _int(cells[1]), cards.get(cells[0])
            if pid is None or card is None:
                continue
            try:
                found = owner(pid)
            except Exception:  # a process gone meanwhile, ps hanging, no permission: its owner is unknown
                found = None
            if found is None:
                who, user = "unknown", None
            else:
                who, user = ("ours" if found[0] == snap["uid"] else "other"), found[1]
                if user is None:
                    user = "uid %s" % found[0]
            card["processes"].append({"pid": pid, "memory_mib": _int(cells[2]), "owner": who, "user": user})
    except Exception as e:
        snap["status"] = "error"
        snap["error"] = str(e) if isinstance(e, RuntimeError) else "%s: %s" % (type(e).__name__, e)
    return snap


def write_snapshot(root, snap):
    """Write `snap` to <root>/work/gpus.json atomically (a temp file in work/, then os.replace); return the path."""
    path = os.path.join(root, GPU_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".gpus.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False)
            f.write("\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def load_snapshot(root):
    """(snapshot, problem) for <root>/work/gpus.json: (the snapshot, None); (None, None) when there is no such file;
    (None, why) when it cannot be read or is no JSON object (cut off, edited by hand). Never raises."""
    try:
        with open(os.path.join(root, GPU_FILE), encoding="utf-8") as f:
            snap = json.load(f)
    except FileNotFoundError:
        return None, None
    except OSError as e:
        return None, "cannot read: %s" % (e.strerror or e)
    except Exception as e:  # not UTF-8, not JSON, nested too deep
        return None, "not valid JSON: %s" % e
    if not isinstance(snap, dict):
        return None, "not a JSON object"
    return snap, None


def read_snapshot(root):
    """The latest snapshot in <root>/work/gpus.json, or None when there is none or it cannot be read
    (load_snapshot)."""
    return load_snapshot(root)[0]
