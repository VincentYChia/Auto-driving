"""The v1-xl and v1.0-pool Anvil scripts (PRETRAIN_V1_PLAN.md §10, §11): static safety checks, the bash helpers
with stubbed ``myquota`` / ``squeue``, the CLI contract the sbatch relies on (spec I4), and ``verify_v1_xl.py`` on a
tiny build (``verify_v1_pool.py`` on a synthetic plan: test_v1_pool_verify.py)."""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "scripts", "anvil", "data_pipeline")
SH = os.path.join(HERE, "build_v1_xl.sh")
SBATCH = os.path.join(HERE, "build_v1_xl.sbatch")
VERIFY = os.path.join(HERE, "verify_v1_xl.py")
POOL_SH = os.path.join(HERE, "build_v1_pool.sh")
POOL_SBATCH = os.path.join(HERE, "build_v1_pool.sbatch")
POOL_VERIFY = os.path.join(HERE, "verify_v1_pool.py")
SRC = ("/anvil/scratch/x-vchia/navsim_workspace/training_data/v1.0-large/archives_hf/openscene-v1.1/"
       "openscene_sensor_trainval_camera")
MYQUOTA = """
Type       Location             Size    Limit    Use   Files   Limit    Use
===========================================================================
home       x-vchia            13.8GB   25.0GB  55.6%      -       -      -
scratch    x-vchia             1.5TB  100.0TB   1.6%  {used}    1.0M  91.9%
projects   x-cis251316         3.3TB    5.0TB  66.5%    4.1M    5.2M  79.7%
"""


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def code_lines(path):
    """Lines without comments (shell scripts: everything after an unquoted-looking #)."""
    out = []
    for line in read(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(re.sub(r"\s+#\s.*$", "", line))
    return out


def find_bash():
    for candidate in (os.environ.get("BASH"), r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")):
        if candidate and os.path.exists(candidate) and "system32" not in candidate.lower():
            return candidate
    return None


BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash")


def posix(path):
    return path.replace("\\", "/")


# --- static checks -------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [SH, SBATCH, POOL_SH, POOL_SBATCH])
def test_shell_scripts_are_strict_and_never_delete_or_move(path):
    text = read(path)
    assert "set -euo pipefail" in text
    lines = code_lines(path)
    for line in lines:
        assert not re.search(r"\b(rm|mv|unlink|rmdir|shred|truncate)\b|-delete\b|rsync\b", line), line
        assert not ("rm " in line and "archives_hf" in line), line
        assert "build_v1_large" not in line, line  # never call the v1-large job: its loop deletes the archives
    assert "x-cis251316" not in text and "/anvil/projects" not in text  # no project-space copies


@needs_bash
@pytest.mark.parametrize("path", [SH, SBATCH, POOL_SH, POOL_SBATCH])
def test_shell_scripts_parse(path):
    subprocess.run([BASH, "-n", posix(path)], check=True)


@pytest.mark.parametrize("path", [SBATCH, POOL_SBATCH])
def test_source_dir_is_explicit_and_only_read(path):
    assert f"V1XL_SRC={SRC}\n" in read(SH)
    sbatch = read(path)
    assert 'SRC="$V1XL_SRC"' in sbatch and not re.search(r'SRC="\$OUT', sbatch)
    assert '--source-dir "$SRC" --storage pack' in sbatch


def test_pool_scripts_reuse_the_v1_xl_helpers_and_constants():
    text = read(POOL_SH)
    sourcing = 'source "$(dirname "${BASH_SOURCE[0]}")/build_v1_xl.sh"'
    assert sourcing in text and "BUILD_TAG=v1-pool" in text
    assert text.index(sourcing) < text.index('if [[ "${BASH_SOURCE[0]}" != "$0" ]]')  # defined when sourced too
    for name in ("V1XL_SHA_CANDIDATES", "V1XL_SHA_LOGS", "V1XL_SHA_INDEX_MANIFEST", "V1XL_SHA_ARCHIVES_JSON"):
        assert f'"${name}"' in text, name
    assert 'source "$REPO_ROOT/scripts/anvil/data_pipeline/build_v1_pool.sh"' in read(POOL_SBATCH)


@pytest.mark.parametrize("path, time, partition", [(SBATCH, "00:30:00", "debug"), (POOL_SBATCH, "04:00:00", "shared")])
def test_sbatch_takes_account_from_the_command_line_and_stays_cpu_only(path, time, partition):
    directives = [line for line in read(path).splitlines() if line.startswith("#SBATCH")]
    assert not any(re.search(r"--account|-A\b|--gpus|--gres", d) for d in directives), directives
    assert f"#SBATCH --time={time}" in directives and f"#SBATCH --partition={partition}" in directives
    assert "gpu|gpu-debug|ai)" in read(path)


@pytest.mark.parametrize("path, sets, views, steps", [
    (SBATCH, "SETS=(v1-xl v1-xl-nd v1-large)", "VIEWS=(v1-xl v1-xl-nd)",
     ["fetch", "repro", "crosscheck", "canonical", "qa", '"view $view"', "verify"]),
    (POOL_SBATCH, "SETS=(v1-nt v1-xl-mix v1-nt-mix v1-all)", "VIEWS=(v1-nt v1-xl-mix v1-nt-mix v1-all)",
     ["fetch", "crosscheck", "canonical", "qa", '"view $view"', "verify"]),
])
def test_sbatch_runs_the_spec_steps_with_guards(path, sets, views, steps):
    text = read(path)
    assert sets in text and '--sets "${SETS[@]}"' in text
    assert views in text and "--storage zip" in text
    for step in steps:
        assert f"quota_guard {step}" in text, step
    assert text.count("quota_guard ") >= len(steps)


@pytest.mark.parametrize("path, name", [(SBATCH, "v1-xl"), (POOL_SBATCH, "v1-pool")])
def test_sbatch_runs_one_build_at_a_time(path, name):
    text = read(path)
    assert "#SBATCH --dependency=singleton" in text.splitlines()  # Slurm holds a second job of this name
    assert f"#SBATCH --job-name={name}" in text.splitlines()
    assert f'exec 9>> "$OUT/.build_{name.replace("-", "_")}.lock"' in text and "flock -n 9 ||" in text
    assert text.index("flock -n 9") < text.index("# --- step 4")  # before the first write


def test_pool_sbatch_memory_workers_and_time_guards():
    """--mem=200G and 16 view workers (v1-xl: 64 ran out at 120 GB, 16 peaked at 44 GB, V); guards cover the steps."""
    text = read(POOL_SBATCH)
    assert "#SBATCH --mem=200G" in text.splitlines() and 'WORKERS="${WORKERS:-16}"' in text
    assert 'SHARDS="${SHARDS:-64}"' in text and "#SBATCH --cpus-per-task=64" in text.splitlines()
    guards = dict((step, int(minutes)) for minutes, step in re.findall(r"need_minutes (\d+) (\w+)", text))
    assert guards["canonical"] >= 45  # ~43 min for 428,694 frames (D: v1-xl's 20.6 min for 203,727, V)
    assert guards["fetch"] >= 10 and guards["qa"] >= 5 and guards["verify"] >= 10
    view_minutes = dict(re.findall(r"(v1-[\w-]+)\) echo (\d+)", text))
    samples = {"v1-nt": 82_626, "v1-xl-mix": 240_034, "v1-nt-mix": 261_929}  # with dev
    for view, n in samples.items():  # ~2.0 ms per sample at 16 workers (v1-xl: 6.6 min for 202,000, V)
        assert int(view_minutes[view]) >= 1.5 * n * 0.002 / 60, view
    assert re.search(r"\*\) echo (\d+)", text) and int(re.search(r"\*\) echo (\d+)", text).group(1)) >= 1.5 * 428_694 * 0.002 / 60
    assert 'need_minutes "$(view_minutes "$view")" "view $view"' in text


@pytest.mark.parametrize("path", [VERIFY, POOL_VERIFY])
def test_verify_never_deletes_or_moves(path):
    text = read(path)
    assert not re.search(r"os\.(remove|unlink|rmdir|rename|removedirs)|shutil\.(rmtree|move)", text)
    assert all(".part" in line for line in text.splitlines() if "os.replace" in line)


# --- bash helpers --------------------------------------------------------------------------------


def run_helper(body, env=None):
    script = f'source "{posix(SH)}"\n{body}\n'
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


def quota_stub(used):
    return "myquota() { cat <<'EOF'" + MYQUOTA.format(used=used) + "EOF\n}\n"


@needs_bash
def test_quota_free_files_parses_myquota():
    result = run_helper(quota_stub("919.4K") + "quota_free_files")
    assert result.returncode == 0 and result.stdout.strip() == "80600", result.stderr


@needs_bash
@pytest.mark.parametrize("used, ok", [("919.4K", True), ("980.0K", True), ("980.1K", False), ("1.0M", False)])
def test_quota_guard_stops_under_20k_free(used, ok):
    result = run_helper(quota_stub(used) + "quota_guard fetch; echo PASSED")
    assert ("PASSED" in result.stdout) == ok, (result.stdout, result.stderr)
    assert (result.returncode == 0) == ok


@needs_bash
def test_quota_guard_fails_closed_without_myquota():
    result = run_helper("myquota() { return 1; }\nquota_guard fetch; echo PASSED")
    assert result.returncode == 1 and "PASSED" not in result.stdout


@needs_bash
@pytest.mark.parametrize("left, minutes", [("29:45", 29), ("1:02:03", 62), ("1-00:10:00", 1450), ("INVALID", 9999)])
def test_minutes_left_reads_squeue(left, minutes):
    result = run_helper(f'squeue() {{ echo "{left}"; }}\nminutes_left', env={"SLURM_JOB_ID": "1"})
    assert result.stdout.strip() == str(minutes), result.stderr


@needs_bash
def test_need_minutes_stops_cleanly_for_the_next_submission():
    result = run_helper('squeue() { echo "9:00"; }\nneed_minutes 15 canonical; echo RAN', env={"SLURM_JOB_ID": "1"})
    assert result.returncode == 0 and "RESUBMIT" in result.stdout and "RAN" not in result.stdout
    result = run_helper('squeue() { echo "20:00"; }\nneed_minutes 15 canonical; echo RAN', env={"SLURM_JOB_ID": "1"})
    assert "RAN" in result.stdout


@needs_bash
def test_pool_helpers_are_v1_xl_s_with_the_pool_tag():
    body = (f'source "{posix(POOL_SH)}"\n' + quota_stub("919.4K")
            + 'echo "SRC=$V1XL_SRC"\nquota_guard fetch\nsqueue() { echo "9:00"; }\nneed_minutes 50 canonical; echo RAN\n')
    result = subprocess.run([BASH, "-c", body], capture_output=True, text=True, env={**os.environ, "SLURM_JOB_ID": "1"})
    assert f"SRC={SRC}" in result.stdout, result.stderr
    assert "[v1-pool] quota guard (fetch): 80600 files free" in result.stdout
    assert "[v1-pool] RESUBMIT: canonical needs ~50 min" in result.stdout and "RAN" not in result.stdout
    assert "[v1-xl] quota guard" in run_helper(quota_stub("919.4K") + "quota_guard fetch").stdout  # v1-xl's unchanged


def pool_gate():
    """The pool sbatch's gate on setup's checks (``setup_passed`` and the loop over them), as a bash snippet."""
    text = read(POOL_SBATCH)
    start = text.index("setup_passed() {")
    return text[start:text.index("\ndone\n", start) + len("\ndone\n")]


def test_pool_sbatch_starts_only_after_setup_s_checks_passed():
    text = read(POOL_SBATCH)
    assert "for check in splits rules leakage; do" in pool_gate()  # every check setup runs
    assert text.index("setup_passed \"$check\" ||") < text.index("flock -n 9") < text.index("# --- step 4")
    sh = read(POOL_SH)  # what setup runs, and writes to <out>/verify/<check>.json
    for check in ("splits", "rules", "leakage"):
        assert f'python "$VERIFY" {check} --out "$OUT"' in sh, check


@needs_bash
def test_pool_gate_needs_passing_setup_reports_newer_than_the_inputs(tmp_path):
    out = tmp_path / "out"
    for folder in ("splits", "index", "verify"):
        (out / folder).mkdir(parents=True)
    inputs = [out / "splits" / "manifest.json", out / "index" / "candidates.parquet"]
    for path in inputs:
        path.write_text("{}")
        os.utime(path, (1_900_000_000, 1_900_000_000))

    def report(check, ok):
        path = out / "verify" / f"{check}.json"
        path.write_text(json.dumps({"command": check, "ok": ok}))
        os.utime(path, (2_000_000_000, 2_000_000_000))

    def run():
        script = (f'set -euo pipefail\npython() {{ "{posix(sys.executable)}" "$@"; }}\nOUT="{posix(str(out))}"\n'
                  f"{pool_gate()}echo PASSED\n")
        return subprocess.run([BASH, "-c", script], capture_output=True, text=True)

    report("splits", True)
    report("rules", True)
    result = run()  # setup stopped before its leakage check
    assert result.returncode == 1 and "PASSED" not in result.stdout
    assert "'verify leakage' has not passed" in result.stderr, result.stderr
    report("leakage", False)  # setup's leakage check failed
    result = run()
    assert result.returncode == 1 and "'verify leakage' has not passed" in result.stderr
    report("leakage", True)
    result = run()
    assert result.returncode == 0 and "PASSED" in result.stdout, result.stderr
    os.utime(inputs[1], (2_100_000_000, 2_100_000_000))  # an input changed after setup's checks
    result = run()
    assert result.returncode == 1 and "'verify splits' has not passed" in result.stderr


# --- the CLI the sbatch calls (spec I4) ----------------------------------------------------------


def test_pipeline_clis_take_the_sbatch_arguments(monkeypatch):
    pytest.importorskip("pyarrow")
    from data_pipeline.v1 import canonical, openscene_images, pilot_view, qa_report

    seen = {}

    def fetch(*args, **kwargs):
        seen["fetch"] = kwargs
        return {"failed": []}

    def build_canonical(out, sets, *args, **kwargs):
        seen["canonical"] = list(sets)
        return {"frames": 0, "sets": {}, "checks": [], "runtime_s": 0}

    def build_view(out, set_name, *args, **kwargs):
        seen.setdefault("views", []).append(set_name)
        seen.setdefault("view_args", []).append((args, kwargs))
        return {"samples": {}, "storage": kwargs.get("storage", args[3] if len(args) > 3 else None), "image": {},
                "images_zip": None, "constant_velocity_baseline": {}}

    monkeypatch.setattr(openscene_images, "fetch", fetch)
    monkeypatch.setattr(canonical, "build", build_canonical)
    monkeypatch.setattr(qa_report, "build", lambda *a, **k: {"frames": 0, "horizon_check_all_frames": {},
                                                             "canonical_checks_failed": []})
    monkeypatch.setattr(pilot_view, "build", build_view)
    with pytest.raises(SystemExit) as stop:
        openscene_images.main(["fetch", "--out", "o", "--full", "--archives", "200", "--source-dir", SRC,
                               "--storage", "pack", "--shard", "3", "--num-shards", "96"])
    assert stop.value.code == 0 and seen["fetch"]["shard"] == 3 and seen["fetch"]["num_shards"] == 96
    assert "pack" in seen["fetch"].values()
    with pytest.raises(SystemExit):
        canonical.main(["--out", "o", "--sets", "v1-xl", "v1-xl-nd", "v1-large", "--metadata", "m.tgz"])
    assert seen["canonical"] == ["v1-xl", "v1-xl-nd", "v1-large"]
    qa_report.main(["--out", "o", "--set", "v1-xl"])
    for name in ("v1-xl", "v1-xl-nd"):
        pilot_view.main(["build", "--out", "o", "--set", name, "--storage", "zip", "--workers", "64"])
    assert seen["views"] == ["v1-xl", "v1-xl-nd"]
    assert all("zip" in args or kwargs.get("storage") == "zip" for args, kwargs in seen["view_args"])
    assert all(64 in args or kwargs.get("workers") == 64 for args, kwargs in seen["view_args"])
    # the v1.0-pool job: its four sets, 16 view workers
    pool_sets = ["v1-nt", "v1-xl-mix", "v1-nt-mix", "v1-all"]
    with pytest.raises(SystemExit):
        canonical.main(["--out", "o", "--sets", *pool_sets, "--metadata", "m.tgz"])
    assert seen["canonical"] == pool_sets
    qa_report.main(["--out", "o", "--set", "v1-all"])
    seen["views"], seen["view_args"] = [], []
    for name in pool_sets:
        pilot_view.main(["build", "--out", "o", "--set", name, "--storage", "zip", "--workers", "16"])
    assert seen["views"] == pool_sets
    assert all(16 in args or kwargs.get("workers") == 16 for args, kwargs in seen["view_args"])


# --- verify_v1_xl.py on a tiny build -------------------------------------------------------------


def load_verify():
    spec = importlib.util.spec_from_file_location("verify_v1_xl", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def stored_zip(path, members):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name in sorted(members):
            z.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), members[name])


@pytest.fixture
def tiny(tmp_path):
    """Archive 7 holds log L: t1, t2 train and t3, t4 dev; v1-large has t1, t3, t4. Smoke-sized, so --smoke."""
    out, large = str(tmp_path / "xl"), str(tmp_path / "large")
    native = {t: f"native {t}".encode() for t in ("t1", "t2", "t3", "t4")}
    small = {t: f"small {t}".encode() for t in native}
    images = [{"token": t, "log": "L", "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
               "width": 1920, "height": 1080} for t, b in native.items()]
    record = {"archive": 7, "sha256_ok": True, "logs_match_map": True, "missing": [], "undecodable": [],
              "wanted": 4, "kept": 4, "images": images}
    for folder, tokens in ((out, {"train": ["t1", "t2"], "dev": ["t3", "t4"]}),
                           (large, {"train": ["t1"], "dev": ["t3", "t4"]})):
        write(os.path.join(folder, "splits", "manifest.json"), json.dumps({"archive_order": [7]}))
        write(os.path.join(folder, "images", "archives", "7.json"), json.dumps(record))
        for split, ts in tokens.items():
            name = "v1-xl" if folder == out else "v1-large"
            write(os.path.join(folder, "splits", f"{name}_tokens_{split}.txt"), "".join(f"{t}\n" for t in ts))
    stored_zip(os.path.join(out, "images", "packs", "7.zip"), {f"images/CAM_F0/L/{t}.jpg": b for t, b in native.items()})

    def lines(ts):
        return "".join(json.dumps({"id": f"openscene/L/{t}", "image": f"images/L/{t}.jpg"}) + "\n" for t in ts)

    view = os.path.join(out, "views", "pilot_v1", "v1-xl")
    zpath = os.path.join(view, "images.zip")
    stored_zip(zpath, {f"images/L/{t}.jpg": b for t, b in small.items()})
    with open(zpath, "rb") as f:
        zip_sha = hashlib.sha256(f.read()).hexdigest()
    write(os.path.join(view, "view.json"), json.dumps({
        "set": "v1-xl", "storage": "zip", "prompt_version": "pilot_v1.1",
        "images_zip": {"members": 4, "bytes": os.path.getsize(zpath), "sha256": zip_sha}}))
    write(os.path.join(view, "train.jsonl"), lines(["t1", "t2"]))
    write(os.path.join(view, "dev.jsonl"), lines(["t3", "t4"]))
    lview = os.path.join(large, "views", "pilot_v1", "v1-large")
    write(os.path.join(lview, "view.json"), json.dumps({"set": "v1-large", "prompt_version": "pilot_v1.1"}))
    write(os.path.join(lview, "dev.jsonl"), lines(["t3", "t4"]))
    for t in ("t3", "t4"):
        os.makedirs(os.path.join(lview, "images", "L"), exist_ok=True)
        with open(os.path.join(lview, "images", "L", f"{t}.jpg"), "wb") as f:
            f.write(small[t])
    return out, large, lview


def failed(checks):
    return [c["check"] for c in checks if not c["ok"]]


def test_verify_passes_on_a_consistent_build(tiny):
    v = load_verify()
    out, large, lview = tiny
    checks, complete = v.check_fetched(out, 1, full=False, deep=True)
    assert complete and not failed(checks)
    assert not failed(v.check_crosscheck(out, large, full=False))
    assert not failed(v.check_view(out, "v1-xl", lview, full=False, samples=50, seed=0, deep=True))


def test_verify_catches_a_wrong_pack_a_changed_frame_and_a_different_dev_image(tiny):
    v = load_verify()
    out, large, lview = tiny
    with open(os.path.join(lview, "images", "L", "t3.jpg"), "wb") as f:
        f.write(b"other bytes")
    assert any("byte-identical" in c for c in failed(v.check_view(out, "v1-xl", lview, False, 50, 0, False)))
    with zipfile.ZipFile(os.path.join(out, "images", "packs", "7.zip"), "a") as z:
        z.writestr("images/CAM_F0/L/extra.jpg", b"x")
    assert failed(v.check_fetched(out, 1, full=False)[0])
    record_path = os.path.join(large, "images", "archives", "7.json")
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    record["images"][0]["sha256"] = "0" * 64
    write(record_path, json.dumps(record))
    assert any("SHA-256" in c for c in failed(v.check_crosscheck(out, large, full=False)))


def test_verify_reports_a_missing_record_as_not_yet(tiny):
    v = load_verify()
    out, _, _ = tiny
    write(os.path.join(out, "splits", "manifest.json"), json.dumps({"archive_order": [7, 8]}))
    checks, complete = v.check_fetched(out, 2, full=False)
    assert not complete and failed(checks) == ["a record for each of the first 2 archives"]
    with pytest.raises(SystemExit) as stop:
        v.main(["fetched", "--out", out, "--archives", "2", "--smoke", "--quiet"])
    assert stop.value.code == v.NOT_YET


# --- verify_v1_xl.py: the warm-up exclusion ------------------------------------------------------


@pytest.fixture
def warm(tmp_path):
    """Logs T (train) and D (dev), anchors every 1 s; warm-up scenes at T 20 s and D 5 s."""
    pytest.importorskip("pyarrow")
    import numpy as np
    from data_pipeline.v1 import tables

    out = str(tmp_path / "xl")
    t = [f"T{k:02d}" for k in range(40)] + [f"D{k:02d}" for k in range(10)]
    cols = {"token": np.array(t, dtype=object), "log_name": np.array([x[0] for x in t], dtype=object),
            "t_us": np.array([int(x[1:]) * 1_000_000 for x in t], dtype=np.int64),
            "our_split": np.array(["train" if x[0] == "T" else "dev" for x in t], dtype=object),
            "guard_reason": np.array([""] * len(t), dtype=object)}
    os.makedirs(os.path.join(out, "index"))
    tables.write(os.path.join(out, "index", "candidates.parquet"), cols)
    tables.write(os.path.join(out, "index", "logs.parquet"),
                 {"log_name": np.array(["T", "D"], dtype=object), "our_split": np.array(["train", "dev"], dtype=object),
                  "guard_reason": np.array(["", ""], dtype=object), "nuplan_split": np.array(["train", "val"], dtype=object)})
    sets = {"v1-large": (["T10", "T13", "T20", "T27", "T28"], ["D00", "D05"]),  # T13, T20, T27 are within 7 s of T20
            "v1-xl": (["T00", "T05", "T10", "T28", "T35"], ["D00", "D05"]),
            "v1-xl-nd": (["T05", "T10", "T28"], ["D00", "D05"])}
    for name, (train, dev) in sets.items():
        for split, ts in (("train", train), ("dev", dev)):
            write(os.path.join(out, "splits", f"{name}_tokens_{split}.txt"), "".join(f"{x}\n" for x in sorted(ts)))
    write(os.path.join(out, "splits", "navsim_v1.1_warmup_test_e2e_tokens.txt"), "D05\nT20\n")
    write(os.path.join(out, "splits", "manifest.json"), json.dumps({"scene_exclusion": {"window_s": 7.0}}))
    return out


def test_verify_warmup_train_is_clean_and_dev_is_reported(warm):
    v = load_verify()
    checks = v.check_leakage(warm)
    assert not failed(checks)
    info = next(c for c in checks if c["check"].startswith("information"))["value"]
    assert info["tokens"]["v1-xl dev"] == 1 and info["within_window"]["v1-large train"] == 3
    left = next(c for c in checks if c["check"].startswith("every v1-large train row left out"))["value"]
    assert left == {"left_out": 3, "inside": 3, "near_frames_placed_only_with_metadata": 0}


@pytest.mark.parametrize("token, what", [("T13", "within"), ("T20", "is a")])
def test_verify_warmup_fails_on_a_train_anchor_near_or_at_a_scene(warm, token, what):
    v = load_verify()
    path = os.path.join(warm, "splits", "v1-xl_tokens_train.txt")
    write(path, "".join(f"{x}\n" for x in sorted(["T00", "T05", "T10", "T28", "T35", token])))
    names = failed(v.check_leakage(warm))
    assert any(what in n and "v1-xl" in n for n in names), names
    write(path, "T00\nT05\nT10\nT28\nT35\n")
    write(os.path.join(warm, "splits", "v1-xl_tokens_train.txt"), "T00\nT05\nT10\nT35\n")  # T28 left out, outside
    assert any(n.startswith("every v1-large train row left out") for n in failed(v.check_leakage(warm)))


def test_verify_locates_unindexed_warmup_frames_itself_and_fails_without_metadata(warm, tmp_path):
    """Review 2026-11-09: the frames are found here, not taken from the plan's manifest."""
    import pickle

    v = load_verify()
    write(os.path.join(warm, "splits", "navsim_v1.1_warmup_test_e2e_tokens.txt"), "D05\nT20\ngap\n")
    meta = tmp_path / "meta"
    meta.mkdir()
    with open(meta / "T.pkl", "wb") as f:  # "gap": a frame of log T at 1.5 s, not an index row; T00 is 1.5 s away
        pickle.dump([{"token": "gap", "log_name": "T", "timestamp": 1_500_000}], f)

    def unindexed(checks):
        return next(c for c in checks if "not index rows" in c["check"])

    checks = v.check_leakage(warm)  # no --metadata: the window cannot be fully checked, so fail
    assert not unindexed(checks)["ok"] and "NOT located" in unindexed(checks)["check"]
    checks = v.check_leakage(warm, metadata=str(meta))  # the plan recorded nothing: it skipped the metadata
    names = failed(checks)
    assert any("not index rows" in n for n in names) and any("within" in n and "v1-xl" in n for n in names), names
    assert unindexed(checks)["value"]["differ"] == [("gap", "T")]
    write(os.path.join(warm, "splits", "manifest.json"),
          json.dumps({"scene_exclusion": {"window_s": 7.0, "located_in_metadata": {"gap": "T"}}}))
    names = failed(v.check_leakage(warm, metadata=str(meta)))  # recorded now, but T00 and T05 are still near it
    assert not any("not index rows" in n for n in names) and any("within" in n for n in names), names
    for name in ("v1-xl", "v1-xl-nd"):  # T05 is 3.5 s away too; T10 is 8.5 s away
        path = os.path.join(warm, "splits", f"{name}_tokens_train.txt")
        write(path, "".join(f"{x}\n" for x in read(path).split() if x not in ("T00", "T05")))
    assert not failed(v.check_leakage(warm, metadata=str(meta)))


def test_verify_devkit_holds_warmup_by_scene_and_every_other_list_at_zero(warm, tmp_path):
    pytest.importorskip("yaml")
    from tests.data import v1_synthetic

    v = load_verify()
    root = v1_synthetic.write_navsim_root(tmp_path / "navsim", train=["T"], val=["D"], filters={
        "warmup_test_e2e": {"log_names": ["T", "D"], "tokens": ["D05", "T20"]},
        "navtest": {"log_names": [], "tokens": ["nothing_selected"]}})
    assert not failed(v.check_leakage(warm, str(root)))  # warm-up logs are not guarded as a whole
    root = v1_synthetic.write_navsim_root(tmp_path / "navsim2", train=["T"], val=["D"], filters={
        "warmup_test_e2e": {"log_names": ["T", "D"], "tokens": ["D05", "T20"]},
        "private_test_e2e": {"log_names": [], "tokens": ["D00"]}})  # a dev hit in another list fails
    assert any("other guarded lists" in n for n in failed(v.check_leakage(warm, str(root))))
    root = v1_synthetic.write_navsim_root(tmp_path / "navsim3", train=["T"], val=["D"], filters={
        "warmup_test_e2e": {"log_names": ["T"], "tokens": ["T20", "T05"]}})  # not the committed list, and T05 is train
    names = failed(v.check_leakage(warm, str(root)))
    assert any("= the committed list" in n for n in names) and any("devkit warmup_test_e2e" in n for n in names)
