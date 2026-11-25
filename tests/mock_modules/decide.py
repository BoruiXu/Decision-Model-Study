"""Mock decision agent for the engine tests: MockRunner.decide runs

    decide.py --request <dir>/request.json

in the decision's directory (work/decisions/<n>-<point>/). The request holds the point, the options the agent
may pick from, the code's default, the prompt and the JSON schema. The answer comes from RSI_MOCK_BEHAVIOR's
"decide" entry (per point, or one for every point): {"choice": ..., "reason": ...} is printed as JSON, {"raw":
text} prints the text as it is; without an entry the agent takes the code's default.
"""
import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    a = p.parse_args()
    with open(a.request, encoding="utf-8") as f:
        req = json.load(f)
    behavior = {}
    if os.path.isfile(os.environ.get("RSI_MOCK_BEHAVIOR") or ""):
        with open(os.environ["RSI_MOCK_BEHAVIOR"], encoding="utf-8") as f:
            behavior = json.load(f).get("decide", {})
    answer = behavior.get(req["point"], behavior)
    if "raw" in answer:
        sys.stdout.write(answer["raw"] + "\n")
        return
    choice = answer.get("choice", req["default"])
    print(json.dumps({"choice": choice, "reason": answer.get("reason", "mock agent: %s" % choice)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
