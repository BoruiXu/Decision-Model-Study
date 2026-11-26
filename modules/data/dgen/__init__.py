"""dgen: failure-targeted data generation for the data module (a port of the team's dgen 0.4).

Run as `PYTHONPATH=<repo>/modules/data:<repo>/modules/common:<repo> <python> -m dgen <command>` with the ojh venv's
Python (rsi.briefs.dgen_command gives the prefix): score the pool with the best version's candidate, diagnose its
errors, generate rows by a recipe and filter them; a trainset is then built from the verified rows by `dset grow
--with`. See dgen.cli for the commands.

- inputs: the brief's inputs (only those: the task, the trainset, the valset's inputs, the pool, the best candidate);
- space: the z-space (the Featurizer's model view, sign·log1p, z-scored on the trainset and the pool);
- diagnose, weakness, cllm: the candidate's errors on the pool, their weakness types and the CLLM reference;
- recipes, rows: the recipe format, the generation methods and the rows they build;
- llm_gen, prompts/: the LLM methods (llm_imitate, llm_pattern) with an independent reviewer, through
  modules/common/llm's `claude -p` calls (standard library only);
- quality: the filters; neardup: near-duplicate groups in the z-space;
- variants, prompts/paraphrase*.md: prompt variation (rsi.variants specs): reviewed paraphrases of the task's prompt
  and the variant rows of a share of the trainset (standard library only).

numpy and scikit-learn (xgboost optional) are imported only by the steps that use them (need_ml): without them a
step that needs them is an input error that names them, and the rest still runs.
"""
import importlib
import os
import sys

# 1.1.0: llm_imitate and llm_pattern; 1.2.0: the task's constraints, neardup; 1.3.0: paraphrase, variants
DGEN_VERSION = "1.3.0"

# The repository root (four levels above this package's directory) and modules/common: dgen uses the framework's rsi
# and the features component.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(REPO_ROOT, "modules", "common"), REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ML_LIBRARIES = (("numpy", "numpy"), ("sklearn", "scikit-learn"))


class DgenError(Exception):
    """An error in the input (the brief, a recipe, a file, a missing library): dgen prints one JSON line
    {"error": <reason>} and exits 1."""


def need_ml(what):
    """Import numpy and scikit-learn for `what` (e.g. "dgen diagnose"); DgenError naming each library that does not
    import, so the agent can install it or choose a recipe that does not need it."""
    missing = []
    for module, name in ML_LIBRARIES:
        try:
            importlib.import_module(module)
        except Exception as e:  # noqa: BLE001  (an ImportError, or a library that fails while loading)
            missing.append("%s (%s: %s)" % (name, type(e).__name__, e))
    if missing:
        raise DgenError("%s needs numpy and scikit-learn, and %s %s not import in %s: install them with "
                        "modules/common/install_deps.sh --python %s, or choose a recipe that does not need them "
                        "(hard, with an existing diagnosis)" % (what, " and ".join(missing),
                                                               "does" if len(missing) == 1 else "do", sys.executable,
                                                               sys.executable))
