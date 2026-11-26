"""A stand-in for `claude -p` in tests: it never calls a model.

It reads its prompt from stdin and appends {"argv", "stdin", "cwd", "files", "pid"} (argv without the program, the
working directory and what it held at the start, its pid) as one line to $FAKE_CLAUDE_LOG. Its reply is the next one of
$FAKE_CLAUDE_SCRIPT, the path of a JSON list of replies taken in order (call k gets item k, k counted from the lines
already in the log):
- an object: printed as the result object of `--output-format json --json-schema`, {"type": "result",
  "structured_output": <the object without "_cost">, "total_cost_usd": <its "_cost", else 0.12>, "session_id": <the
  --session-id value>}, exit 0; an object with "_child" (a path) first starts `sleep 60` in its process group (the
  call's) with SIGTERM ignored and writes that child's pid to the path; one with "_sleep" then sleeps that long (s);
- {"raw": text, "exit": code}: prints the text as it is and exits with the code (default 0).
With no reply left it prints a message and exits 1.
"""
import json
import os
import signal
import subprocess
import sys
import time


def main():
    argv = sys.argv[1:]
    stdin = sys.stdin.read()
    log = os.environ["FAKE_CLAUDE_LOG"]
    try:
        with open(log, encoding="utf-8") as f:
            k = sum(1 for line in f if line.strip())
    except OSError:
        k = 0
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"argv": argv, "stdin": stdin, "cwd": os.getcwd(), "files": sorted(os.listdir(".")),
                            "pid": os.getpid()}) + "\n")
    with open(os.environ["FAKE_CLAUDE_SCRIPT"], encoding="utf-8") as f:
        script = json.load(f)
    if k >= len(script):
        sys.stdout.write("fake claude: no reply scripted for call %d\n" % (k + 1))
        return 1
    reply = script[k]
    if "raw" in reply:
        sys.stdout.write(reply["raw"])
        return int(reply.get("exit", 0))
    reply = dict(reply)
    cost = reply.pop("_cost", 0.12)
    child_file = reply.pop("_child", None)
    if child_file:  # a group member that ignores SIGTERM: only a SIGKILL to the group ends it
        old = signal.signal(signal.SIGTERM, signal.SIG_IGN)  # inherited through exec
        child = subprocess.Popen(["sleep", "60"])
        signal.signal(signal.SIGTERM, old)
        with open(child_file, "w") as f:
            f.write("%d\n" % child.pid)
    time.sleep(reply.pop("_sleep", 0))
    session = argv[argv.index("--session-id") + 1] if "--session-id" in argv else None
    sys.stdout.write(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                                 "structured_output": reply, "total_cost_usd": cost, "session_id": session}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
