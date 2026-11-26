"""What the compare module reads: only the paths in its brief's `inputs` (never a label file of the evalset or the
testset, never the corpus, the pool or the raw data)."""
import copy
import json
import os

from rsi.rules import compare_settings


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def set_file(d, logical):
    """The path of logical file `logical` (e.g. rows, inputs) of the artifact directory `d`, from its manifest."""
    return os.path.join(d, read_json(os.path.join(d, "manifest.json"))["files"][logical]["path"])


class Inputs(object):
    """A compare brief's inputs, loaded.

    - task: the task definition; settings: rsi.rules.compare_settings(task);
    - train_rows, val_rows: the rows (with labels) of the trainset and the valset; val_inputs: the valset's
      inputs.jsonl (its meta.cluster gives the comparison's clusters); eval_inputs: the evalset's inputs.jsonl;
    - candidate_dir: the candidate to compare (None when the brief gives none);
    - baselines: id -> {"manifest", "dir"} of every registered baseline the brief gives;
    - baseline_rows: id -> {"train", "val"}: the rows of the trainset (none for a baseline without one) and of the
      valset each of those baselines names, for refits (None when the brief gives no rows of that set);
    - ids: the set ids to write into manifests (trainset, valset, evalset; testset at the milestone); purpose, round,
      task_path, brief;
    - resources (resources.onejev, resources.gpu_ids) and train_command (for team_recipe);
    - milestone_compare only: comparison (the manifest of the best version's comparison), testset_inputs (the
      testset's inputs.jsonl, never its labels) and testset_id;
    - budget_left, spend_file: what remains of each resource, and where spend goes (docs/interfaces/spend.md).
    """

    @classmethod
    def load(cls, brief_path):
        self = cls()
        self.brief_path = os.path.abspath(brief_path)
        self.brief = b = read_json(brief_path)
        inp = b["inputs"]
        self.purpose, self.round, self.ids = b["purpose"], b["round"], dict(b["ids"])
        self.task_path = inp["task"]
        self.task = read_json(self.task_path)
        self.settings = compare_settings(self.task)
        self.budget_left, self.spend_file = b.get("budget_left"), b.get("spend_file")
        self.resources, self.train_command = b.get("resources") or {}, b.get("train_command")
        self.candidate_dir = inp.get("candidate")
        self.train_rows = read_jsonl(set_file(inp["trainset"], "rows")) if "trainset" in inp else []
        if "valset" in inp:
            self.val_rows = read_jsonl(set_file(inp["valset"], "rows"))
            self.val_inputs = read_jsonl(set_file(inp["valset"], "inputs"))
        else:
            self.val_rows, self.val_inputs = [], []
        self.eval_inputs = read_jsonl(inp["evalset_inputs"]) if "evalset_inputs" in inp else []
        milestone = self.purpose == "milestone_compare"
        self.testset_inputs = read_jsonl(inp["testset_inputs"]) if milestone and "testset_inputs" in inp else []
        self.testset_id = self.ids.get("testset") if milestone else None
        self.comparison = (read_json(os.path.join(inp["comparison"], "manifest.json"))
                           if milestone and "comparison" in inp else None)
        self.baselines = {}
        for key, path in sorted(inp.items()):
            mpath = os.path.join(path, "manifest.json")
            if not os.path.isfile(mpath):
                continue
            m = read_json(mpath)
            if m.get("kind") == "baseline":
                self.baselines[m.get("id") or key] = {"manifest": m, "dir": path}
        cache = {}

        def rows_of(kind, sid):
            key = "%s_rows_%s" % (kind, sid)
            if key not in cache:
                if key in inp:
                    cache[key] = read_jsonl(inp[key])
                elif kind in inp and sid == self.ids.get(kind):
                    cache[key] = self.train_rows if kind == "trainset" else self.val_rows
                else:
                    cache[key] = None
            return cache[key]

        self.baseline_rows = dict(
            (bid, {"train": rows_of("trainset", b["manifest"]["trainset_id"]) if b["manifest"].get("trainset_id")
                   else [], "val": rows_of("valset", b["manifest"].get("valset_id"))})
            for bid, b in self.baselines.items())
        return self

    def of_baseline(self, bid):
        """A copy of these inputs seen from registered baseline `bid` (its sets): ids its trainset, valset and evalset,
        val_rows its valset's rows, train_rows its trainset's rows, eval_inputs None (its evalset's inputs are not
        given), testset_id and testset_inputs unchanged. Used to write a baseline made from it at the milestone."""
        m = self.baselines[bid]["manifest"]
        out = copy.copy(self)
        out.ids = {"trainset": m.get("trainset_id"), "valset": m["valset_id"], "evalset": m["evalset_id"]}
        out.val_rows = self.baseline_rows[bid]["val"]
        out.train_rows = self.baseline_rows[bid]["train"]
        out.eval_inputs = None
        return out
