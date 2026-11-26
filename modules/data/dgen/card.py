"""The GPU card score-pool runs model.predict on (standard library only), chosen as the compare module and trsi choose:
the idle card (under IDLE_MIB MiB in use, from nvidia-smi) with the least memory in use among the task's
resources.gpu_ids (every card when not set), waiting for one up to a bounded time. Its uuid goes to
CUDA_VISIBLE_DEVICES (CUDA may number the cards differently from nvidia-smi)."""
import subprocess
import time

from . import DgenError

IDLE_MIB = 1024
POLL_S = 30
SMI_TIMEOUT_S = 60


class SmiTimeout(DgenError):
    """nvidia-smi did not answer within SMI_TIMEOUT_S (a hung driver): polled again within the wait, else the error."""


def gpu_table():
    """[(index, uuid, MiB in use)] of every card; None only when there is no nvidia-smi to run (no GPU driver here).
    SmiTimeout when it hangs; DgenError when it cannot be run otherwise or fails. A card whose memory nvidia-smi does
    not give as a number (e.g. MIG) counts as busy."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=SMI_TIMEOUT_S)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        raise SmiTimeout("nvidia-smi did not answer within %g s" % SMI_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        raise DgenError("nvidia-smi cannot be run: %s" % e)
    if out.returncode != 0:
        raise DgenError("nvidia-smi failed (exit %d): %s" % (out.returncode, out.stderr.strip()[-300:]))
    rows = []
    for line in out.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                used = float(parts[2])
            except ValueError:
                used = float("inf")
            rows.append((int(parts[0]), parts[1], used))
    return rows


def pick_card(gpu_ids, wait_s, poll_s=POLL_S):
    """The uuid of the idle card with the least memory in use among `gpu_ids` (None or []: all cards), polled every
    poll_s seconds for up to wait_s; None only when there is no nvidia-smi. DgenError when no card is idle in time,
    nvidia-smi keeps hanging past the wait, or it fails."""
    allowed = set(int(i) for i in gpu_ids) if gpu_ids else None
    end = time.time() + wait_s
    while True:
        try:
            table = gpu_table()
        except SmiTimeout as e:  # a hung nvidia-smi: asked again within the wait, never "no card choice"
            if time.time() >= end:
                raise DgenError("%s, so no card could be chosen within %g s" % (e, wait_s))
            time.sleep(min(poll_s, max(0.0, end - time.time())))
            continue
        if table is None:
            return None
        idle = sorted((used, i, u) for i, u, used in table if (allowed is None or i in allowed) and used < IDLE_MIB)
        if idle:
            return idle[0][2]
        if time.time() >= end:
            raise DgenError("no idle card (under %d MiB in use) among %s after %d s: %s" % (
                IDLE_MIB, sorted(allowed) if allowed else "all cards", wait_s,
                ", ".join("card %d %g MiB" % (i, used) for i, u, used in table) or "nvidia-smi lists no card"))
        time.sleep(min(poll_s, max(0.0, end - time.time())))
