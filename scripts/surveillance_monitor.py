#!/usr/bin/env python3
"""Surveillance Monitor for 6-Band Kaggle Kernels.
Checks statuses of all 9 kernels, records readings to surveillance_log.jsonl,
and automatically triggers dependent evaluation runs once parent training finishes.
Supports self-healing staging reconstruction, robust parent-child dependency tracking,
and automated artifact synchronization.
"""

import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KERNELS = [
    {
        "slug": "cuq6-baseline-k01",
        "name": "Model_training.ipynb (Baseline)",
        "type": "training",
        "dependents": ["cuq6-uncertainty-k02", "cuq6-sacp-k09"]
    },
    {
        "slug": "cuq6-uncertainty-k02",
        "name": "Model_uncertainty_comparison.ipynb",
        "type": "evaluation",
        "parent": "cuq6-baseline-k01"
    },
    {
        "slug": "cuq6-credit-k03",
        "name": "Model_training_credit.ipynb",
        "type": "training/distillation",
        "parent": "cuq6-ensemble-k04",
        "dependents": []
    },
    {
        "slug": "cuq6-ensemble-k04",
        "name": "Model_training_ensembles.ipynb",
        "type": "training",
        "dependents": ["cuq6-crede-k05", "cuq6-credit-k03"]
    },
    {
        "slug": "cuq6-crede-k05",
        "name": "Model_uncertainty_CreDE.ipynb",
        "type": "evaluation",
        "parent": "cuq6-ensemble-k04"
    },
    {
        "slug": "cuq6-multicp-k06",
        "name": "Model_training_multihead.ipynb",
        "type": "training",
        "dependents": ["cuq6-multicp-uq-k07", "cuq6-multicp-sacp-k08"]
    },
    {
        "slug": "cuq6-multicp-uq-k07",
        "name": "Model_uncertainty_multicp.ipynb",
        "type": "evaluation",
        "parent": "cuq6-multicp-k06"
    },
    {
        "slug": "cuq6-multicp-sacp-k08",
        "name": "Model_uncertainty_multicp_sacp.ipynb",
        "type": "evaluation",
        "parent": "cuq6-multicp-k06"
    },
    {
        "slug": "cuq6-sacp-k09",
        "name": "Model_sacp_comparison.ipynb",
        "type": "evaluation",
        "parent": "cuq6-baseline-k01"
    },
    {
        "slug": "cuq6-mambahsi-k10",
        "name": "Model_training_mambahsi.ipynb",
        "type": "training",
        "dependents": ["cuq6-mambahsi-uq-k11"]
    },
    {
        "slug": "cuq6-mambahsi-uq-k11",
        "name": "Model_uncertainty_comparison_mambahsi.ipynb",
        "type": "evaluation",
        "parent": "cuq6-mambahsi-k10"
    },
    {
        "slug": "cuq6-dofa-k12",
        "name": "Model_training.ipynb (DOFA)",
        "type": "training",
        "dependents": ["cuq6-dofa-uq-k13"]
    },
    {
        "slug": "cuq6-dofa-uq-k13",
        "name": "Model_uncertainty_comparison.ipynb (DOFA)",
        "type": "evaluation",
        "parent": "cuq6-dofa-k12"
    }
]

NOTEBOOK_SOURCE_MAP = {
    "cuq6-baseline-k01": ("Classification/baseline/Model_training.ipynb", "Model_training.ipynb"),
    "cuq6-uncertainty-k02": ("Classification/baseline/Model_uncertainty_comparison.ipynb", "Model_uncertainty_comparison.ipynb"),
    "cuq6-credit-k03": ("Classification/credit/Model_training_credit.ipynb", "Model_training_credit.ipynb"),
    "cuq6-ensemble-k04": ("Classification/ensemble/Model_training_ensembles.ipynb", "Model_training_ensembles.ipynb"),
    "cuq6-crede-k05": ("Classification/ensemble/Model_uncertainty_CreDE.ipynb", "Model_uncertainty_CreDE.ipynb"),
    "cuq6-multicp-k06": ("Classification/multicp/Model_training_multihead.ipynb", "Model_training_multihead.ipynb"),
    "cuq6-multicp-uq-k07": ("Classification/multicp/Model_uncertainty_multicp.ipynb", "Model_uncertainty_multicp.ipynb"),
    "cuq6-multicp-sacp-k08": ("Classification/multicp_sacp/Model_uncertainty_multicp_sacp.ipynb", "Model_uncertainty_multicp_sacp.ipynb"),
    "cuq6-sacp-k09": ("Classification/sacp/Model_sacp_comparison.ipynb", "Model_sacp_comparison.ipynb"),
    "cuq6-mambahsi-k10": ("Classification/mambahsi/Model_training_mambahsi.ipynb", "Model_training_mambahsi.ipynb"),
    "cuq6-mambahsi-uq-k11": ("Classification/mambahsi/Model_uncertainty_comparison_mambahsi.ipynb", "Model_uncertainty_comparison_mambahsi.ipynb"),
    "cuq6-dofa-k12": ("Classification/dofa/Model_training.ipynb", "Model_training.ipynb"),
    "cuq6-dofa-uq-k13": ("Classification/dofa/Model_uncertainty_comparison.ipynb", "Model_uncertainty_comparison.ipynb"),
}

def get_kernel_status(slug: str) -> str:
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", f"daneshselwal/{slug}"],
            capture_output=True,
            text=True,
            timeout=30
        )
        out = res.stdout.strip()
        if "has status" in out:
            status = out.split('"')[1].replace("KernelWorkerStatus.", "")
            return status
        return out or "UNKNOWN"
    except Exception as e:
        return f"ERR: {e}"

def retrigger_kernel(slug: str, repo_root: Path) -> bool:
    """Stage and push kernel to Kaggle. Self-heals from repository metadata if /tmp is missing."""
    staging_dir = Path(f"/tmp/cuq6_staging/{slug}")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sync metadata from repository kaggle_staging
    repo_meta = repo_root / "kaggle_staging" / slug / "kernel-metadata.json"
    dest_meta = staging_dir / "kernel-metadata.json"
    if repo_meta.exists():
        shutil.copy2(repo_meta, dest_meta)

    # Ensure enable_gpu is true
    if dest_meta.exists():
        try:
            with open(dest_meta) as f:
                meta_json = json.load(f)
            if not meta_json.get("enable_gpu", False):
                meta_json["enable_gpu"] = True
                with open(dest_meta, "w") as f:
                    json.dump(meta_json, f, indent=2)
        except Exception:
            pass

    # 2. Sync notebook from repository source
    if slug in NOTEBOOK_SOURCE_MAP:
        rel_src, nb_name = NOTEBOOK_SOURCE_MAP[slug]
        src_nb = repo_root / rel_src
        dest_nb = staging_dir / nb_name
        if src_nb.exists():
            shutil.copy2(src_nb, dest_nb)

    # 3. Push kernel
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(staging_dir)],
            capture_output=True,
            text=True,
            timeout=60
        )
        out = res.stdout.strip() or res.stderr.strip()
        print(f"[{slug}] Kaggle push result: {out}")
        return "successfully pushed" in out.lower()
    except Exception as e:
        print(f"[{slug}] Push exception: {e}")
        return False

def sync_completed_artifacts(slug: str, repo_root: Path):
    """Sync results for newly completed evaluation kernels."""
    sync_targets = {
        "cuq6-sacp-k09": (repo_root / "Classification/sacp/results", "combined_window_sacp_summary.csv"),
        "cuq6-crede-k05": (repo_root / "Classification/ensemble/results", "crede_evaluation_summary.csv"),
        "cuq6-credit-k03": (repo_root / "Classification/credit/results", "credit_evaluation_summary.csv"),
        "cuq6-multicp-uq-k07": (repo_root / "Classification/multicp/results", "multicp_summary.csv"),
        "cuq6-multicp-sacp-k08": (repo_root / "Classification/multicp_sacp/results", "scmcp_summary.csv"),
        "cuq6-mambahsi-uq-k11": (repo_root / "Classification/mambahsi/results/uncertainty_results", "conformal_reports_all_models.xlsx"),
        "cuq6-dofa-uq-k13": (repo_root / "Classification/dofa/results/uncertainty_results", "conformal_reports_all_models.xlsx"),
    }
    if slug not in sync_targets:
        return

    dest_dir, check_file = sync_targets[slug]
    if (dest_dir / check_file).exists():
        return

    print(f"[{slug}] Kernel COMPLETE! Downloading outputs...")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            res = subprocess.run(
                ["kaggle", "kernels", "output", f"daneshselwal/{slug}", "-p", tmpdir],
                capture_output=True,
                text=True,
                timeout=180
            )
            if res.returncode == 0:
                dest_dir.mkdir(parents=True, exist_ok=True)
                for root, _, files in os.walk(tmpdir):
                    if ".git" in root:
                        continue
                    for f in files:
                        if f.endswith((".csv", ".xlsx", ".png", ".json")) and not f.endswith(".log"):
                            src_path = os.path.join(root, f)
                            shutil.copy2(src_path, dest_dir / f)
                            print(f"[{slug}] Synced {f} -> {dest_dir}")
        except Exception as e:
            print(f"[{slug}] Sync output exception: {e}")

def check_and_auto_trigger(readings: dict, repo_root: Path) -> list:
    """Evaluate parent completion rules and auto-trigger downstream runs with cycle state persistence."""
    state_file = repo_root / "scripts" / ".trigger_state.json"
    state = {}
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
        except Exception:
            state = {}

    triggered = []

    # Rules: (Parent slug, [dependent slugs])
    cascade_rules = [
        ("cuq6-baseline-k01", ["cuq6-uncertainty-k02", "cuq6-sacp-k09"]),
        ("cuq6-ensemble-k04", ["cuq6-crede-k05", "cuq6-credit-k03"]),
        ("cuq6-multicp-k06", ["cuq6-multicp-uq-k07", "cuq6-multicp-sacp-k08"]),
        ("cuq6-mambahsi-k10", ["cuq6-mambahsi-uq-k11"]),
        ("cuq6-dofa-k12", ["cuq6-dofa-uq-k13"]),
    ]

    for parent, deps in cascade_rules:
        parent_status = readings.get(parent, "UNKNOWN")
        if parent_status == "COMPLETE":
            for dep in deps:
                curr_status = readings.get(dep, "UNKNOWN")
                dep_state = state.get(dep, {})
                already_triggered_for_parent = (dep_state.get("triggered_for") == parent)

                # Special-case: k02 is already fully verified and stored in repo
                if dep == "cuq6-uncertainty-k02" and (repo_root / "Classification/baseline/results/uncertainty_results/summary_metrics.csv").exists():
                    state[dep] = {
                        "triggered_for": parent,
                        "status": "VERIFIED_COMPLETE",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    continue

                # Trigger if:
                # 1. Dependent is in ERROR or UNKNOWN
                # 2. Or dependent has not yet been triggered for the current parent's COMPLETE status
                #    (handles premature runs that completed without parent weights)
                should_trigger = False
                if curr_status in ["ERROR", "UNKNOWN"]:
                    should_trigger = True
                elif not already_triggered_for_parent and curr_status not in ["RUNNING", "TRIGGERED", "QUEUED"]:
                    should_trigger = True

                if should_trigger:
                    print(f"[AUTO-TRIGGER] Parent {parent} is COMPLETE! Retriggering {dep}...")
                    ok = retrigger_kernel(dep, repo_root)
                    if ok:
                        triggered.append(dep)
                        readings[dep] = "TRIGGERED"
                        state[dep] = {
                            "triggered_for": parent,
                            "status": "TRIGGERED",
                            "timestamp": datetime.datetime.now().isoformat()
                        }

    # Also check if any completed kernel needs artifact syncing
    for slug, status in readings.items():
        if status == "COMPLETE":
            sync_completed_artifacts(slug, repo_root)

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    return triggered

def take_reading(repo_root: Path):
    now = datetime.datetime.now().isoformat()
    readings = {}

    print(f"=== Surveillance Reading at {now} ===")
    print(f"{'Kernel Slug':<24} | {'Type':<18} | {'Status':<15} | {'Notebook'}")
    print("-" * 80)

    for k in KERNELS:
        status = get_kernel_status(k["slug"])
        readings[k["slug"]] = status
        print(f"{k['slug']:<24} | {k['type']:<18} | {status:<15} | {k['name']}")

    triggered = check_and_auto_trigger(readings, repo_root)

    # Append to surveillance log
    log_file = repo_root / "surveillance_log.jsonl"
    entry = {
        "timestamp": now,
        "statuses": readings,
        "auto_triggered": triggered
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print("-" * 80)
    print(f"Logged reading to {log_file}")
    if triggered:
        print(f"Auto-triggered downstream evaluation kernels: {triggered}")
    return entry

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    take_reading(root)
