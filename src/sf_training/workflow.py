"""Prepare, validate and explicitly submit isolated training jobs (dry-run default)."""

import argparse
import datetime
import json
import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from sf_training.utils import write_json, sha, accumulation
from sf_training.normalization import METHOD, bounds
from sf_training.validation import SMOKE_STEPS, SMOKE_RESUME_STEP, validate_metrics

ROOT = Path(__file__).resolve().parents[2]
SF_SHA = "372819b31ba9a2c8fc5989edc7d525cb187cecd5"
OPENPI_SHA = "215abfb217dbac7d5f1273282331b9b1866c0479"


def files():
    paths = list((ROOT / "src/sf_training").glob("*.py"))
    paths += [
        ROOT / p
        for p in (
            "scripts/sf_common.sh",
            "scripts/sf.sh",
            "slurm/sf_train.sbatch",
            "config/sf_training/recipe.json",
            "config/sf_training/assets.json",
            "requirements/sf_train.lock",
        )
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def sources():
    for name, rev in [("spatial_forcing", SF_SHA), ("openpi", OPENPI_SHA)]:
        repo = ROOT / ".runtime/repos" / name
        actual = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != rev:
            raise ValueError(f"{name} revision mismatch")
        if subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError(f"{name} contains local changes")


def assets():
    home = ROOT / ".runtime/training"
    manifest = ROOT / "config/sf_training/assets.json"
    verified = json.loads((home / "assets_verified.json").read_text())
    if verified["manifest_sha256"] != sha(manifest):
        raise ValueError("Asset verification missing/stale")
    for row in json.loads(manifest.read_text())["files"]:
        p = home / row["path"]
        if not p.exists() or p.stat().st_size != row["size"]:
            raise ValueError("Missing/wrong-sized asset: " + str(p))
    marker = home / "arena_l0/packed/complete.json"
    packed = json.loads(marker.read_text())
    if packed["source_manifest"] != verified["manifest_sha256"]:
        raise ValueError("Packed dataset identity mismatch")
    for name, value in packed["files"].items():
        if (marker.parent / name).stat().st_size != value["size"]:
            raise ValueError("Packed data size mismatch: " + name)
    stats = home / "norm_stats.json"
    data = json.loads(stats.read_text())
    if (
        data["source_manifest"] != verified["manifest_sha256"]
        or data["frames"] != 361736
        or data["episodes"] != 3018
    ):
        raise ValueError("Normalization identity mismatch")
    if data.get("method") != METHOD:
        raise ValueError("Legacy normalization; repair and audit before preparing a new run")
    for value in data["norm_stats"].values():
        bounds(value)
    audit_path = home / "normalization_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["norm_stats_sha256"] != sha(stats) or audit["packed_manifest_sha256"] != sha(marker):
        raise ValueError("Full-dataset normalization audit missing/stale")
    tokenizer = ROOT / ".runtime/cache/openpi/big_vision/paligemma_tokenizer.model"
    return dict(
        norm_stats=sha(stats),
        packed_manifest=sha(marker),
        tokenizer=sha(tokenizer),
        assets=sha(manifest),
        normalization_audit=sha(audit_path),
    )


def environment():
    import importlib.metadata as md
    import transformers
    from transformers.models.siglip import check

    if not check.check_whether_transformers_replace_is_installed_correctly():
        raise ValueError("Missing SF transformers patch")
    installed = {x.metadata["Name"].lower(): x.version for x in md.distributions()}
    patch = (
        ROOT
        / ".runtime/repos/spatial_forcing/openpi-SF/src/openpi/models_pytorch/transformers_replace"
    )
    destination = Path(transformers.__file__).parent
    patched = {
        str(p.relative_to(patch)): sha(destination / p.relative_to(patch))
        for p in patch.rglob("*.py")
    }
    for p in patch.rglob("*.py"):
        if sha(p) != patched[str(p.relative_to(patch))]:
            raise ValueError("Transformer overlay differs from pinned SF source")
    return dict(packages=installed, transformers_patch=patched)


def verify_plan(path):
    plan = json.loads(path.read_text())
    if plan["root"] != str(ROOT) or plan["files"] != files():
        raise ValueError("Training code/config changed since prepare; create a new run")
    sources()
    if plan["assets"] != assets() or plan["environment"] != environment():
        raise ValueError("Prepared assets/environment changed")
    accumulation(plan["recipe"], plan["gpus"])
    return plan


def prepare(a):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", a.run_id):
        raise ValueError("Unsafe run id")
    directory = ROOT / "runs" / a.run_id
    if directory.exists():
        raise FileExistsError("Run already exists; use a new id")
    sources()
    asset_ids = assets()
    cfg = json.loads((ROOT / "config/sf_training/recipe.json").read_text())
    if a.profile == "smoke":
        cfg.update(steps=SMOKE_STEPS, checkpoint_every=SMOKE_RESUME_STEP, log_every=1)
    accumulation(cfg, a.gpus)
    plan = dict(
        schema=1,
        root=str(ROOT),
        profile=a.profile,
        gpus=a.gpus,
        time_limit="03:00:00" if a.profile == "smoke" else "24:00:00",
        recipe=cfg,
        files=files(),
        assets=asset_ids,
        environment=environment(),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    directory.mkdir(parents=True)
    write_json(directory / "plan.json", plan)
    print(f"Prepared {directory}/plan.json; no job submitted.")


def smoke_gate(run_id, plan):
    path = ROOT / "runs" / run_id / "smoke_passed.json"
    passed = json.loads(path.read_text())
    if (
        passed["files"] != plan["files"]
        or passed["assets"] != plan["assets"]
        or passed["environment"] != plan["environment"]
    ):
        raise ValueError("Smoke evidence does not match code/assets/environment")
    return passed


def finish_smoke(a):
    path = ROOT / "runs" / a.run_id / "plan.json"
    plan = verify_plan(path)
    if plan["profile"] != "smoke":
        raise ValueError("Not a smoke plan")
    evidence = {}
    if plan["recipe"]["global_batch"] != 32 or plan["recipe"]["steps"] != SMOKE_STEPS:
        raise ValueError("Smoke must use the formal batch and expanded validation budget")
    for arm in ("baseline", "sf"):
        status = json.loads((path.parent / arm / "status.json").read_text())
        inference = json.loads((path.parent / arm / "inference_smoke.json").read_text())
        if not (
            status["completed"]
            and status["step"] == SMOKE_STEPS
            and status["resumed_from"] == SMOKE_RESUME_STEP
            and inference["finite"]
            and inference["shape"] == [1, 10, 32]
        ):
            raise ValueError("Smoke training/resume/inference incomplete")
        evidence[arm] = validate_metrics(path.parent / arm / "metrics.jsonl")
    write_json(
        path.parent / "smoke_passed.json",
        {**{k: plan[k] for k in ("files", "assets", "environment", "gpus")}, "metrics": evidence},
    )
    print("PASS: both arms completed 32 updates, sparse rotation batches, resume, and inference.")


def matching_full(smoke, full):
    if smoke["profile"] != "smoke" or full["profile"] != "full":
        raise ValueError("Expected a smoke plan and a fresh full plan")
    if any(smoke[k] != full[k] for k in ("files", "assets", "environment", "gpus")):
        raise ValueError("Smoke/full code, data, environment or GPU count differs")
    ignored = {"steps", "checkpoint_every", "log_every"}
    if {k: v for k, v in smoke["recipe"].items() if k not in ignored} != {k: v for k, v in full["recipe"].items() if k not in ignored}:
        raise ValueError("Smoke/full training recipes differ")


def launch_full(a):
    """Called on the allocated node only after finish-smoke and report succeed."""
    full = verify_plan(ROOT / "runs" / a.full_run / "plan.json")
    smoke = verify_plan(ROOT / "runs" / a.smoke_run / "plan.json")
    matching_full(smoke, full)
    smoke_gate(a.smoke_run, full)
    # Also wait for this smoke allocation to exit, keeping the total GPU cap.
    for arm in ("baseline", "sf"):
        submit(SimpleNamespace(run_id=a.full_run, arm=arm, submit=True,
                               smoke_run=a.smoke_run, after_smoke=True, resume=False))


def submit(a):
    path = ROOT / "runs" / a.run_id / "plan.json"
    plan = verify_plan(path)
    mode = a.arm
    skip_smoke = getattr(a, "skip_smoke", False)
    then_full = getattr(a, "then_full_run", None)
    if skip_smoke and (plan["profile"] != "full" or a.smoke_run or a.after_smoke or then_full):
        raise ValueError("--skip-smoke requires a full plan without smoke/dependency options")
    if then_full:
        if mode != "smoke":
            raise ValueError("--then-full-run is only valid for training smoke")
        full = verify_plan(ROOT / "runs" / then_full / "plan.json")
        matching_full(plan, full)
        full_dir = ROOT / "runs" / then_full
        if any(full_dir.glob("*/step-*.pt")) or any((full_dir / "submission").glob("*.json")):
            raise ValueError("Automatic launch requires a fresh full run from pi05_base")
    if plan["profile"] == "smoke" and mode != "smoke":
        raise ValueError("Smoke plan must run both arms via --arm smoke")
    if plan["profile"] == "full" and mode == "smoke":
        raise ValueError("Full plan needs baseline or sf")
    dependency = None
    if plan["profile"] == "full" and a.submit and not skip_smoke:
        if not a.smoke_run:
            raise ValueError("Full submission requires --smoke-run")
        if a.after_smoke:
            smoke_plan = verify_plan(ROOT / "runs" / a.smoke_run / "plan.json")
            if smoke_plan["profile"] != "smoke" or any(
                smoke_plan[k] != plan[k] for k in ("files", "assets", "environment")
            ):
                raise ValueError("Dependent smoke identity mismatch")
            evidence = json.loads(
                (ROOT / "runs" / a.smoke_run / "submission/smoke.json").read_text()
            )
            dependency = evidence.get("job_id")
            if (
                evidence["status"] != "submitted"
                or not dependency
                or not dependency.isdigit()
            ):
                raise ValueError("No confirmed smoke submission")
        else:
            smoke_gate(a.smoke_run, plan)
    directory = path.parent / "submission"
    directory.mkdir(exist_ok=True)
    # Do not silently retry uncertain sbatch results. Review squeue by job name first.
    receipt = directory / (mode + ".json")
    if receipt.exists() and a.submit and not a.resume:
        raise FileExistsError(
            "Submission already attempted. Inspect receipt/Slurm; --resume is only for a finished interrupted run."
        )
    if a.resume and a.submit:
        if not receipt.exists():
            raise ValueError("No previous submission to resume")
        old = json.loads(receipt.read_text())
        job = old.get("job_id")
        if not job:
            raise ValueError(
                "Previous submission status unknown; resolve it before resubmitting"
            )
        live = subprocess.check_output(["squeue", "-h", "-j", str(job)], text=True)
        if live.strip():
            raise ValueError("Previous job still active; refusing duplicate submission")
    name = "sf-" + a.run_id + "-" + mode
    cmd = [
        "sbatch",
        "--parsable",
        "--account=gts-jli741-paid",
        "--partition=gpu-a100",
        "--qos=inferno",
        f"--gres=gpu:a100:{plan['gpus']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={8 * plan['gpus']}",
        f"--mem={max(96, 48 * plan['gpus'])}G",
        f"--time={plan['time_limit']}",
        "--job-name=" + name,
        "--chdir=" + str(ROOT),
        "--output=" + str(ROOT / "logs" / (a.run_id + "-" + mode + "-%j.out")),
        str(ROOT / "slurm/sf_train.sbatch"),
        a.run_id,
        mode,
        "skip-smoke" if skip_smoke else (a.smoke_run or "-"),
    ]
    if dependency:
        cmd[1:1] = ["--dependency=afterok:" + dependency, "--kill-on-invalid-dep=yes"]
    if then_full:
        cmd.append(then_full)
    print(shlex.join(cmd), flush=True)
    if not a.submit:
        print("Dry run only. Add --submit to allocate paid GPUs.")
        return
    attempt = dict(
        command=cmd,
        gpu_smoke="explicitly_skipped" if skip_smoke else (a.smoke_run or mode),
        attempted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="unknown",
    )
    if receipt.exists():
        receipt.rename(
            directory
            / (mode + "-previous-" + str(int(__import__("time").time())) + ".json")
        )
    write_json(receipt, attempt)
    result = subprocess.run(cmd, text=True, capture_output=True)
    attempt.update(
        stdout=result.stdout, stderr=result.stderr, returncode=result.returncode
    )
    if result.returncode == 0:
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(
                "Unrecognized sbatch response; inspect queue before retrying"
            )
        attempt.update(job_id=job, status="submitted")
    write_json(receipt, attempt)
    print(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Submission uncertain/failed; inspect queue before retrying")


def main():
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="cmd", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("profile", choices=["smoke", "full"])
    p.add_argument("run_id")
    p.add_argument("--gpus", type=int, choices=[1, 2, 4], default=1)
    p = subs.add_parser("check")
    p.add_argument("run_id")
    p = subs.add_parser("finish-smoke")
    p.add_argument("run_id")
    p = subs.add_parser("verify-smoke")
    p.add_argument("run_id")
    p.add_argument("smoke_run")
    p = subs.add_parser("launch-full")
    p.add_argument("smoke_run")
    p.add_argument("full_run")
    p = subs.add_parser("submit")
    p.add_argument("run_id")
    p.add_argument("--arm", choices=["smoke", "baseline", "sf"], required=True)
    p.add_argument("--smoke-run")
    p.add_argument("--after-smoke", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--then-full-run")
    p.add_argument("--skip-smoke", action="store_true",
                   help="Explicitly submit full training without a GPU smoke; numerical and identity checks still apply")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a)
    elif a.cmd == "submit":
        submit(a)
    elif a.cmd == "finish-smoke":
        finish_smoke(a)
    elif a.cmd == "launch-full":
        launch_full(a)
    elif a.cmd == "verify-smoke":
        smoke_gate(a.smoke_run, verify_plan(ROOT / "runs" / a.run_id / "plan.json"))
        print("Matching training smoke passed.")
    else:
        verify_plan(ROOT / "runs" / a.run_id / "plan.json")
        print("PASS: local checks. GPU training smoke remains separate.")


if __name__ == "__main__":
    main()
