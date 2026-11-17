"""Request text extraction for the text baselines."""
import json


def text_of(row, mode):
    """`messages`: same as v1 `request_text`; `state_json`: the whole state as canonical JSON."""
    if mode == "messages":
        # Provenance: dynamicserve-agreement-router/data_pipeline_v2/sanity_check/common.py `request_text`.
        messages = row["state"]["messages"] if "state" in row else row["messages"]
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    if mode == "state_json":
        return json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
    raise ValueError(f"unknown text mode {mode!r}")
