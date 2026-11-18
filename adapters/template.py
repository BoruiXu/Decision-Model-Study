"""Skeleton for a new dataset adapter. Copy this file, rename it, and implement ``adapt``.

An adapter reads one source file and yields OneJev rows, each a dict with:
  id        unique string id
  state     {"messages": [{"role": ..., "content": ...}, ...]}  the model input
  question  {"type": str, "instructions": str, "criteria": {label: description}}
  target    {label: probability}; keys must be in ``criteria`` and sum to 1
  meta      free-form dict (fields referenced by the experiment config live here)
  split     split name; names starting with "test" are never read without --allow-test

Select it in the experiment config with ``data.adapter: <module name>`` or ``data.adapter: <path to a .py
file>`` (relative to the experiment file).

Keyword arguments come from the experiment config and are passed as ``adapt(path, **args)``:
  data.adapter_args             every file
  data.train_adapter_args       the train file (merged over adapter_args)
  data.eval.<set>.adapter_args  that eval set's files (merged over adapter_args)
For example one file with a split column can feed train and eval: ``train_adapter_args: {split: train}``
and ``eval.val.adapter_args: {split: val}``, and the adapter keeps the rows whose column equals ``split``.
Give every keyword a default so the adapter also works without arguments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


def adapt(path: Path, **args) -> Iterator[dict]:
    raise NotImplementedError("copy adapters/template.py and implement adapt(path, **args)")
