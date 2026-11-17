import hashlib
import subprocess

import pytest

from harness.onejev import OneJevMismatch, ensure_onejev

PATCH = """\
diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-hello
+patched
"""


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    (repo / "a.txt").write_text("hello\n")
    _git("add", ".", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=repo)
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    patch = tmp_path / "serve.patch"
    patch.write_text(PATCH)
    return repo, rev, patch


def _lock(repo, rev, sha):
    return {"repo": str(repo), "revision": rev, "patched_sha256": {"a.txt": sha}}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_patch_applied_once(tmp_path, upstream):
    repo, rev, patch = upstream
    lock = _lock(repo, rev, _sha("patched\n"))
    dest = tmp_path / "vendor" / "OneJev"
    assert ensure_onejev(dest, lock, patch) == dest
    assert (dest / "a.txt").read_text() == "patched\n"
    ensure_onejev(dest, lock, patch)  # second call must not re-apply or fail
    assert (dest / "a.txt").read_text() == "patched\n"


def test_patched_sha_mismatch_raises(tmp_path, upstream):
    repo, rev, patch = upstream
    lock = _lock(repo, rev, _sha("something else\n"))
    with pytest.raises(OneJevMismatch):
        ensure_onejev(tmp_path / "OneJev", lock, patch)


def test_wrong_revision_raises(tmp_path, upstream):
    repo, rev, patch = upstream
    dest = tmp_path / "OneJev"
    ensure_onejev(dest, _lock(repo, rev, _sha("patched\n")), patch)
    (dest / "b.txt").write_text("x")
    _git("add", "b.txt", cwd=dest)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x", cwd=dest)
    with pytest.raises(OneJevMismatch):
        ensure_onejev(dest, _lock(repo, rev, _sha("patched\n")), patch)
