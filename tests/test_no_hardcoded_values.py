from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ["harness", "adapters", "skills", "catalog"]
FORBIDDEN = ["/Users/", "/home/", "xbr", "/shared/ssd", "gpu_locks"]


def test_no_hardcoded_values():
    hits = []
    for d in DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for f in base.rglob("*"):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            try:
                text = f.read_text()
            except UnicodeDecodeError:
                continue
            hits += [f"{f.relative_to(ROOT)}: {s}" for s in FORBIDDEN if s in text]
    assert not hits, hits
