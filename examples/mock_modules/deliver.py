"""Mock rsi-deliver (see mocklib for the call, the task and the knobs).

Packages each candidate the brief gives (candidate; after a refreshed evalset also best_candidate): package/
holds the candidate's own prediction entry (model.predict) and model file; the package runs on the candidate's
valset inputs (reproduction), on the evalset inputs (with per-row timing for the latency) and, for a milestone
release, on the testset inputs. Only input files are given, never labels. Sent back after a refusal, it
packages again exactly the subdirectories the registration refused (named in the feedback).
"""
import os
import re
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mocklib as M  # noqa: E402

ROLES = (("candidate", "valset_inputs", "delivery"), ("best_candidate", "best_valset_inputs", "delivery-best"))


def pct(xs, q):
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def run(pkg, inputs, out, timing=None):
    cmd = [os.path.join(pkg, "predict"), "--inputs", inputs, "--out", out] + (["--timing", timing] if timing else [])
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def package(c, cand_dir, val_inputs, base, malformed):
    tol = (c.task.get("delivery") or {}).get("reproduction_tolerance", 1e-6)
    cand_id = M.art_id(cand_dir)
    cm = M.read_json(os.path.join(cand_dir, "manifest.json"))
    d = c.new_dir(base)
    pkg = os.path.join(d, "package")
    os.makedirs(pkg)
    shutil.copy2(os.path.join(cand_dir, cm["model"]["predict"]), os.path.join(pkg, "predict"))
    shutil.copy2(os.path.join(cand_dir, cm["model"]["location"]), os.path.join(pkg, "model.json"))
    pv = os.path.join(d, "predictions_val.jsonl")
    run(pkg, val_inputs, pv)
    pe, timing = os.path.join(d, "predictions_eval.jsonl"), os.path.join(c.outdir, ".timing-%s.jsonl" % base)
    run(pkg, c.inputs["evalset_inputs"], pe, timing)
    want = dict((r["id"], r) for r in M.read_jsonl(os.path.join(cand_dir, cm["files"]["predictions_val"]["path"])))
    got = M.read_jsonl(pv)
    bad = [r["id"] for r in got if r["decision"] != want[r["id"]]["decision"]
           or any(abs(r["probs"][x] - want[r["id"]]["probs"][x]) > tol for x in M.LABELS)]
    times = [t["ms"] for t in M.read_jsonl(timing)]
    os.remove(timing)
    files = {"predictions_val": {"path": "predictions_val.jsonl", "rows": len(got)},
             "predictions_eval": {"path": "predictions_eval.jsonl", "rows": len(M.read_jsonl(pe))}}
    ids = c.brief["ids"]
    m = M.base_manifest(c.brief, "delivery", "deliver", [cand_id, cm["valset_id"], ids["evalset"]])
    if "testset_inputs" in c.inputs:  # a milestone release: the user agreed to use the sealed testset
        pt = os.path.join(d, "predictions_test.jsonl")
        run(pkg, c.inputs["testset_inputs"], pt)
        files["predictions_test"] = {"path": "predictions_test.jsonl", "rows": len(M.read_jsonl(pt))}
        m["testset_id"] = ids["testset"]
        m["parents"].append(ids["testset"])
    preds = dict((r["id"], r) for r in M.read_jsonl(pe))
    trace = []
    for i, (r, ms) in enumerate(zip(M.read_jsonl(c.inputs["evalset_inputs"]), times)):
        p = preds[r["id"]]
        trace.append({"id": "call-%06d" % i, "state": r["state"], "output": {"probs": p["probs"],
                                                                            "decision": p["decision"]},
                      "model": cand_id, "latency_ms": round(ms, 4), "timestamp": M.now_iso(), "outcome": None})
    files["trace"] = {"path": "trace.jsonl", "rows": M.write_jsonl(os.path.join(d, "trace.jsonl"), trace)}
    m["files"] = files
    m.update({"candidate_id": cand_id, "evalset_id": ids["evalset"], "valset_id": cm["valset_id"],
              "package": {"location": "package",
                          "entry": "command line: package/predict --inputs <inputs.jsonl> --out <predictions.jsonl>",
                          "environment": "python3 (3.9+), stdlib only"},
              "runtime": {"latency_ms_p50": round(statistics.median(times), 4),
                          "latency_ms_p95": round(pct(times, 0.95), 4), "cost_unit": "cpu_seconds",
                          "cost_per_1k": round(sum(times) / len(times), 4), "consistent_with_offline": not bad}})
    if malformed:  # the knob: a flag written as text, and no self-check, so registration refuses it
        m["runtime"]["consistent_with_offline"] = "yes"
    M.write_json(os.path.join(d, "manifest.json"), m)
    if not malformed:
        c.check("delivery", d)
    return d, {"candidate": cand_id, "reproduces_val": not bad, "p50": m["runtime"]["latency_ms_p50"],
               "p95": m["runtime"]["latency_ms_p95"]}


def main():
    c = M.Call("deliver")
    if c.rehand_if_answered():
        return
    st = c.state()
    refused = re.findall(r"子目录 (\S+?)（delivery）登记被拒", c.feedback)
    redo = [st["made"][x] for x in refused if x in st.get("made", {})] if c.handed_back_before() else []
    malformed = bool(c.knob("malformed")) and c.first
    made, dirs, gates = dict(st.get("made") or {}), [], []
    for role, vrole, base in ROLES:
        if role not in c.inputs or (redo and role not in redo):
            continue
        d, g = package(c, c.inputs[role], c.inputs[vrole], base, malformed and role == "candidate")
        made[os.path.basename(d)] = role
        dirs.append(d)
        gates.append(g)
    if not dirs:
        c.fail("简报里没有要交付的候选")
    st["made"] = made
    c.save_state(st)
    note = "；".join("%s：能加载，复现验证集 %s，延迟 p50 %s ms、p95 %s ms" % (
        g["candidate"], "是" if g["reproduces_val"] else "否", g["p50"], g["p95"]) for g in gates)
    if redo:
        return c.handback(dirs, notes="按登记的错误重新打包 %s。%s" % ("、".join(refused), note))
    r = M.write_record(c, "record", "候选要一个能调用的交付包", "包着候选自己的预测入口就能复现它",
                       "打包 %s" % "、".join(g["candidate"] for g in gates), "eval latency_ms_p95", None,
                       gates[0]["p95"], "success", "交付包 = 预测入口 + model.json；cost_per_1k 是每千次的 CPU 秒",
                       artifacts=[g["candidate"] for g in gates])
    c.handback(dirs + [r], notes=note)


if __name__ == "__main__":
    main()
