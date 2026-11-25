"""ManualRunner: a person (or another session) does the module's work.

launch writes the prompt to prompt-<n>.md in the output directory (next to brief.md, brief.json and, for a
return, the feedback file) and returns at once; the engine does not wait for it but stops and says so. The
agent has ended once handback.json appears in the output directory; then `rsi run` goes on.
"""
import os

from rsi.runners import base


class ManualRunner(base.Runner):
    name = "manual"

    def launch(self, ws, dispatch, prompt, resume_session=None, new_session=None):
        out = base.outdir(ws, dispatch)
        with open(os.path.join(out, "prompt-%d.md" % base.launch_number(dispatch)), "w", encoding="utf-8") as f:
            f.write(prompt)
        return self.attach(ws, dispatch)

    def started(self, ws, dispatch):
        return os.path.exists(os.path.join(base.outdir(ws, dispatch), "prompt-%d.md" % base.launch_number(dispatch)))

    def attach(self, ws, dispatch):
        return {"runner": self, "waits": False, "outdir": base.outdir(ws, dispatch)}

    def poll(self, handle):
        if os.path.isfile(os.path.join(handle["outdir"], "handback.json")):
            return base.result(exit_code=0)
        return None
