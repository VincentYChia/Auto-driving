"""The v1-xl Anvil scripts (PRETRAIN_V1_PLAN.md §10): static safety checks, the bash helpers with stubbed
``myquota`` / ``squeue``, the CLI contract the sbatch relies on (spec I4), and ``verify_v1_xl.py`` on a tiny build."""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "scripts", "anvil", "data_pipeline")
SH = os.path.join(HERE, "build_v1_xl.sh")
SBATCH = os.path.join(HERE, "build_v1_xl.sbatch")
VERIFY = os.path.join(HERE, "verify_v1_xl.py")
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


@pytest.mark.parametrize("path", [SH, SBATCH])
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
@pytest.mark.parametrize("path", [SH, SBATCH])
def test_shell_scripts_parse(path):
    subprocess.run([BASH, "-n", posix(path)], check=True)


def test_source_dir_is_explicit_and_only_read():
    assert f"V1XL_SRC={SRC}\n" in read(SH)
    sbatch = read(SBATCH)
    assert 'SRC="$V1XL_SRC"' in sbatch and not re.search(r'SRC="\$OUT', sbatch)
    assert '--source-dir "$SRC" --storage pack' in sbatch


def test_sbatch_takes_account_from_the_command_line_and_stays_cpu_only():
    directives = [line for line in read(SBATCH).splitlines() if line.startswith("#SBATCH")]
    assert not any(re.search(r"--account|-A\b|--gpus|--gres", d) for d in directives), directives
    assert "#SBATCH --time=00:30:00" in directives and "#SBATCH --partition=debug" in directives
    assert "gpu|gpu-debug|ai)" in read(SBATCH)


def test_sbatch_runs_the_spec_steps_with_guards():
    text = read(SBATCH)
    assert "SETS=(v1-xl v1-xl-nd v1-large)" in text and '--sets "${SETS[@]}"' in text
    assert "VIEWS=(v1-xl v1-xl-nd)" in text and "--storage zip" in text
    steps = ["fetch", "repro", "crosscheck", "canonical", "qa", '"view $view"', "verify"]
    for step in steps:
        assert f"quota_guard {step}" in text, step
    assert text.count("quota_guard ") >= len(steps)


def test_sbatch_runs_one_build_at_a_time():
    text = read(SBATCH)
    assert "#SBATCH --dependency=singleton" in text.splitlines()  # Slurm holds a second v1-xl job
    assert 'exec 9>> "$OUT/.build_v1_xl.lock"' in text and "flock -n 9 ||" in text
    assert text.index("flock -n 9") < text.index("# --- step 4")  # before the first write


def test_verify_never_deletes_or_moves():
    text = read(VERIFY)
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
