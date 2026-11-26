"""install_env.sh against a fake ojh repo (a venv whose python is this interpreter without site-packages (-S)
plus fake torch/transformers/fla packages, a fake Hopper nvidia-smi and a fake CUDA 12.8 nvcc; no network, no GPU):
--check, which installs or edits nothing; an install whose compare libraries fail (a fake uv), which only warns; the
machine config edit of install_env.py; and install_deps.sh --check."""
import os
import subprocess
import sys

import yaml

from conftest import MODULE

SCRIPT = MODULE / "install_env.sh"
HELPER = MODULE / "install_env.py"
DEPS = MODULE.parent / "common" / "install_deps.sh"
SYSTEM_PATH = ":".join(d for d in ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")
                       if os.path.isdir(d))
DIST = {"torch": "2.8.0+cu128", "triton": "3.4.0", "transformers": "5.19.0", "flash-linear-attention": "0.5.2"}
MODULES = {
    "torch/__init__.py": "__version__ = '2.8.0+cu128'\n\nclass version:\n    cuda = '12.8'\n\n"
                         "class _C:\n    _GLIBCXX_USE_CXX11_ABI = True\n",
    "transformers/__init__.py": "__version__ = '5.19.0'\n",
    "fla/__init__.py": "__version__ = '0.5.2'\n",
    "fla/ops/__init__.py": "",
    "fla/ops/gated_delta_rule/__init__.py": "def chunk_gated_delta_rule(*args, **kwargs):\n    pass\n",
}


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def fake_ojh(tmp, gpu=True):
    """(ojh repo, site-packages, fake bin dir, CUDA home, machine.yaml, environment) of a fake ojh install under tmp;
    gpu False: nvidia-smi finds no GPU."""
    ojh, site, fakebin, cuda = tmp / "ojh", tmp / "site", tmp / "bin", tmp / "cuda-12.8"
    for rel, text in MODULES.items():
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(text)
    for name, version in DIST.items():
        info = site / ("%s-%s.dist-info" % (name.replace("-", "_"), version))
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: %s\nVersion: %s\n" % (name, version))
    (ojh / ".venv" / "bin").mkdir(parents=True)
    (ojh / "constraints.txt").write_text("causal-conv1d==1.7.0\n")
    executable(ojh / "setup.sh", "touch %s\n" % (tmp / "setup-ran"))
    executable(ojh / ".venv" / "bin" / "python", '#!/bin/sh\nPYTHONPATH=%s _PYTHON_HOST_PLATFORM=linux-x86_64 '
               'exec %s -S "$@"\n' % (site, sys.executable))
    executable(ojh / ".venv" / "bin" / "ojh", "#!/bin/sh\necho '[ ok ] python'\necho 'no blocking problem'\n")
    executable(fakebin / "nvidia-smi", '#!/bin/sh\ncase "$1" in --query-gpu=*) '
               'echo "0, GPU-fake0, NVIDIA H100 PCIe, 9.0, 10, 81559" ;; esac\n' if gpu else "#!/bin/sh\n")
    executable(cuda / "bin" / "nvcc", "#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V12.8.93'\n")
    machine = tmp / "ojh-home" / "machine.yaml"
    machine.parent.mkdir()
    machine.write_text('# fake machine config\npython: %s/.venv/bin/python\nenv:\n  HF_HUB_OFFLINE: "1"  # keep\n'
                       % ojh)
    env = {k: v for k, v in os.environ.items() if k not in ("OJH_TOOLS_DIR", "CUDA_PATH", "CUDA_VISIBLE_DEVICES")}
    env.update(PATH="%s:%s" % (fakebin, os.environ["PATH"]), CUDA_HOME=str(cuda), http_proxy="http://127.0.0.1:9",
               https_proxy="http://127.0.0.1:9")
    return ojh, site, fakebin, cuda, machine, env


def test_check_prints_the_plan_and_changes_nothing(tmp_path):
    tmp = tmp_path.resolve()
    ojh, site, fakebin, cuda, machine, env = fake_ojh(tmp)
    before = machine.read_bytes()

    out = subprocess.run(["bash", str(SCRIPT), "--ojh", str(ojh), "--machine", str(machine), "--check"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
    text = out.stdout
    assert out.returncode == 1, text
    tag = "cp%d%d" % sys.version_info[:2]
    wheel = "causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-%s-%s-linux_x86_64.whl" % (tag, tag)
    cache = machine.parent / "cache"
    for expected in (
            "venv exists",
            "torch 2.8.0+cu128 (CUDA 12.8, cxx11abi TRUE)",
            "nvcc        %s (CUDA 12.8, from CUDA_HOME" % cuda,
            "would    install pyyaml (with constraints.txt)",
            "would    install pytest (with constraints.txt)",
            "from the prebuilt wheel %s (https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/%s)"
            % (wheel, wheel.replace("+", "%2B")),
            "missing scikit-learn",
            "would    install modules/common/requirements.txt (install_deps.sh; not required: training does not",
            "would    install tilelang==0.1.15",
            "would set CUDA_HOME: None -> %s" % cuda,
            "would set PATH: None -> %s/bin:%s/.venv/bin:%s\n" % (cuda, ojh, SYSTEM_PATH),   # never the caller's
            "would set TILELANG_CACHE_DIR: None -> %s/tilelang" % cache,
            "ok       import torch 2.8.0+cu128",
            "ok       import fla 0.5.2",
            "skip     GPU forward + backward test (not run with --check)",
            "ok       ojh doctor: no blocking problem",
            "result: not ready"):
        assert expected in text, "%r not in:\n%s" % (expected, text)
    assert machine.read_bytes() == before
    assert sorted(p.name for p in machine.parent.iterdir()) == ["machine.yaml"]   # no backup, no cache dir
    assert not (tmp / "setup-ran").exists()


def test_the_compare_libraries_never_block_training(tmp_path):
    # Step 3 (install_deps.sh) failing only warns (training does not need them; ③ Compare is a report), unless the
    # install moved a protected version such as torch's.
    tmp = tmp_path.resolve()
    ojh, site, fakebin, cuda, machine, env = fake_ojh(tmp, gpu=False)
    executable(fakebin / "uv", '#!/bin/sh\ncase "$*" in --version) echo "uv 0.0.0" ;;\n'
               '  *" -r "*) [ "$FAKE_UV" = move-torch ] || exit 1\n'
               '    rm -rf %s/torch-*.dist-info; mkdir %s/torch-9.9.9.dist-info\n'
               '    printf "Metadata-Version: 2.1\\nName: torch\\nVersion: 9.9.9\\n" \\\n'
               '      > %s/torch-9.9.9.dist-info/METADATA ;;\nesac\n' % (site, site, site))
    for fake_uv, expected in (("fail", "warn     numpy, scikit-learn, xgboost not installed (install_deps.sh exit 1"),
                              ("move-torch", "failed   torch changed from 2.8.0+cu128 to 9.9.9 by: install_deps.sh")):
        text = subprocess.run(["bash", str(SCRIPT), "--ojh", str(ojh), "--machine", str(machine), "--no-gpu-test"],
                              env=dict(env, FAKE_UV=fake_uv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=120).stdout
        assert expected in text, "%r not in:\n%s" % (expected, text)
        failed = text.split("\n  failed:\n")[-1] if "\n  failed:\n" in text else ""
        assert ("install_deps.sh" in failed) == (fake_uv == "move-torch"), text


def test_machine_env_edit_applies_once_keeps_comments_and_crlf(tmp_path):
    cuda = tmp_path / "cuda-12.8"
    executable(cuda / "bin" / "nvcc", "#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V12.8.93'\n")
    cases = {   # name: (machine.yaml text, PATH expected after the edit)
        "lf": ('python: /v/bin/python\nenv:\n  HF_HUB_OFFLINE: "1"  # keep\nout_root: /runs\n',
               "%s/bin:/v/bin:%s" % (cuda, SYSTEM_PATH)),
        "crlf": ('# config\r\nenv:\r\n  HF_HOME: /hf\r\nlock_dir: /locks\r\n', "%s/bin:/v/bin:%s" % (cuda, SYSTEM_PATH)),
        "path": ("env:\n  PATH: /opt/nv/tk/bin:/v/bin:/usr/bin  # full PATH\n  CUDA_HOME: /opt/nv/tk\n",
                 "%s/bin:/v/bin:/usr/bin" % cuda),     # the replaced CUDA_HOME's bin leaves PATH
    }
    for name, (text, path) in cases.items():
        machine = tmp_path / name / "machine.yaml"
        machine.parent.mkdir()
        machine.write_bytes(text.encode())

        def apply(*extra):
            return subprocess.run([sys.executable, str(HELPER), "machine-env", "--machine", str(machine),
                                   "--cuda-home", str(cuda), "--torch-cuda", "12.8", "--venv-bin", "/v/bin",
                                   "--tilelang-cache", "/c/tilelang", "--apply"] + list(extra),
                                  stdout=subprocess.PIPE, universal_newlines=True).stdout

        assert "ME_STATUS=failed" in apply("--expect", "PATH=/not/what/the/gpu/test/used"), name
        assert machine.read_bytes() == text.encode(), name            # refused: nothing written, no backup
        assert "ME_STATUS=changed" in apply("--expect", "PATH=" + path), name
        edited = machine.read_bytes().decode()
        env = yaml.safe_load(edited)["env"]
        assert (env["CUDA_HOME"], env["PATH"], env["TILELANG_CACHE_DIR"]) == (str(cuda), path, "/c/tilelang"), name
        backups = [p.name for p in machine.parent.iterdir() if ".bak-" in p.name]
        assert len(backups) == 1 and (machine.parent / backups[0]).read_bytes() == text.encode(), name
        assert "ME_STATUS=right" in apply(), name                  # the second run changes nothing
        assert machine.read_bytes().decode() == edited and len(list(machine.parent.iterdir())) == 2, name
    assert "HF_HUB_OFFLINE: \"1\"  # keep" in (tmp_path / "lf" / "machine.yaml").read_text()
    crlf = (tmp_path / "crlf" / "machine.yaml").read_bytes()
    assert crlf.count(b"\r\n") == crlf.count(b"\n") and b"lock_dir: /locks\r\n" in crlf
    assert "# full PATH" in (tmp_path / "path" / "machine.yaml").read_text()   # a replaced key keeps its comment


def fake_python(tmp, missing=()):
    """A python (this interpreter without site-packages) that has numpy, scikit-learn, xgboost and pyyaml at the
    versions requirements.txt asks for, but for the import names in `missing`."""
    site = tmp / "deps-site"
    for name, mod, version in (("numpy", "numpy", "2.2.6"), ("scikit_learn", "sklearn", "1.9.1"),
                               ("xgboost", "xgboost", "3.2.0"), ("PyYAML", "yaml", "6.0.2")):
        if mod in missing:
            continue
        (site / mod).mkdir(parents=True)
        (site / mod / "__init__.py").write_text("__version__ = %r\n" % version)
        info = site / ("%s-%s.dist-info" % (name, version))
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: %s\nVersion: %s\n" % (name, version))
    fake = tmp / "fake-python"
    executable(fake, '#!/bin/sh\nPYTHONPATH=%s exec %s -S "$@"\n' % (site, sys.executable))
    return str(fake)


def run(args):
    return subprocess.run(["bash"] + [str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=60)


def test_install_deps_check_reports_what_is_missing(tmp_path):
    fake = fake_python(tmp_path, missing={"sklearn"})        # a script that fails `import sklearn`
    p = run([DEPS, "--python", fake, "--check"])
    assert p.returncode == 1 and "missing scikit-learn" in p.stdout
    assert "ok pyyaml 6.0.2" in p.stdout                     # requirements.txt's pyyaml, imported as yaml
