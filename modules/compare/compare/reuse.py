"""Each baseline's resolved spec, its spec_sha256, and the plan: reuse a registered baseline, re-predict it on a new
evalset, or train it.

A registered baseline is reused when four things match: spec_sha256, trainset_id (any for zero_shot, which does not
depend on the trainset), valset_id and evalset_id. When only the evalset differs (a refresh), it is re-predicted: a
new baseline with reused_from (a CPU baseline is refitted, which is deterministic). Anything else is trained.
"""
import collections
import hashlib
import json
import re

from rsi.rules import adapter_path

from features import FEATURES_VERSION

from . import COMPARE_VERSION
from . import cpu
from . import gpu

KNOWN_BASELINES = ("lr", "gbdt", "mlp", "zero_shot", "few_shot", "team_recipe")

Plan = collections.namedtuple("Plan", "name kind spec spec_sha256 action source_id")


def spec_sha256(name, spec, features_sig, policy):
    """The sha256 of the baseline's name, its resolved spec, the features signature, COMPARE_VERSION and the policy
    key (policy_key: the task's policy section and the dpolicy version; canonical JSON)."""
    text = json.dumps({"name": name, "spec": spec, "features": features_sig, "version": COMPARE_VERSION,
                       "policy": policy}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def features_key(task):
    """The part of the features a CPU baseline's spec depends on: the features code's version and the sha256 of the
    task's adapter file (None: the generic views). Neither the adapter's location nor the column count (which follows
    the trainset, matched separately) is in it."""
    path = adapter_path(task)
    sha = None
    if path is not None:
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
    return {"version": FEATURES_VERSION, "adapter_sha256": sha}


def policy_key(task):
    """What a baseline's decision policy depends on besides its predictions: None without the task's policy section
    (nothing is fitted), else {"section": the section, "version": DPOLICY_VERSION}, so that a change of dpolicy fits
    the baselines again instead of reusing them."""
    section = task.get("policy")
    if section is None:
        return None
    from dpolicy import DPOLICY_VERSION
    return {"section": section, "version": DPOLICY_VERSION}


def gpu_spec(name, settings, task):
    """The resolved specification of GPU baseline `name` (scored as text: no features); team_recipe's config is the
    one the train command gets (gpu.team_config: a null base resolved)."""
    resources = task.get("resources") or {}
    default_base = ((resources.get("onejev") or {}).get("default_config") or {}).get("base")
    if name == "team_recipe":
        return {"model": "the team's recipe, trained by the train command",
                "config": gpu.team_config(settings, resources)}
    out = {"model": "the base model, %s" % name.replace("_", "-"), "base": settings[name]["base"] or default_base}
    if name == "few_shot":
        out.update(k=settings["few_shot"]["k"], max_example_chars=settings["few_shot"]["max_example_chars"],
                   seed=settings["seed"])
    return out


def _order(bid):
    m = re.search(r"(\d+)$", bid)
    return (int(m.group(1)) if m else -1, bid)


def _match(inputs, name, sha):
    """(action, source_id) for baseline `name` with spec_sha256 `sha` against the registered baselines (the latest
    match wins)."""
    ids = inputs.ids
    exact, older = [], []
    for bid in sorted(inputs.baselines, key=_order):
        m = inputs.baselines[bid]["manifest"]
        if (m.get("name") != name or m.get("spec_sha256") != sha or m.get("valset_id") != ids.get("valset")
                or m.get("testset_id") is not None):
            continue
        if name != "zero_shot" and m.get("trainset_id") != ids.get("trainset"):
            continue
        (exact if m.get("evalset_id") == ids.get("evalset") else older).append(bid)
    if exact:
        return "reuse", exact[-1]
    if older:
        return "repredict", older[-1]
    return "train", None


def unknown(settings):
    """The names of compare.baselines the module does not know, in order."""
    return [n for n in settings["baselines"] if n not in KNOWN_BASELINES]


def plan(inputs):
    """One Plan per known name of compare.baselines, in order (unknown names are not planned: see unknown)."""
    task = inputs.task
    settings = inputs.settings
    policy = policy_key(task)
    fkey = None
    out = []
    for name in settings["baselines"]:
        if name not in KNOWN_BASELINES:
            continue
        if name in cpu.CPU_BASELINES:
            if fkey is None:
                fkey = features_key(task)
            kind, sp, sig = "cpu", cpu.spec(name, settings), fkey
        else:
            kind, sp, sig = "gpu", gpu_spec(name, settings, task), None
        sha = spec_sha256(name, sp, sig, policy)
        action, source = _match(inputs, name, sha)
        out.append(Plan(name, kind, sp, sha, action, source))
    return out
