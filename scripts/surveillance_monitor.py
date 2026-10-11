#!/usr/bin/env python3
"""Surveillance Monitor for 6-Band Kaggle Kernels.
Checks statuses of all 9 kernels, records readings to surveillance_log.jsonl,
and automatically triggers dependent evaluation runs once parent training finishes.
"""

import datetime
import json
import subprocess
import sys
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
        "dependents": []
    },
    {
        "slug": "cuq6-ensemble-k04",
        "name": "Model_training_ensembles.ipynb",
        "type": "training",
        "dependents": ["cuq6-crede-k05"]
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
    }
]

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

def retrigger_kernel(slug: str) -> bool:
    staging_dir = Path(f"/tmp/cuq6_staging/{slug}")
    if not staging_dir.exists():
        return False
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(staging_dir)],
            capture_output=True,
            text=True,
            timeout=45
        )
        return "successfully pushed" in res.stdout
    except Exception:
        return False

def take_reading(repo_root: Path):
    now = datetime.datetime.now().isoformat()
    readings = {}
    triggered = []

    print(f"=== Surveillance Reading at {now} ===")
    print(f"{'Kernel Slug':<24} | {'Type':<18} | {'Status':<15} | {'Notebook'}")
    print("-" * 80)

    for k in KERNELS:
        status = get_kernel_status(k["slug"])
        readings[k["slug"]] = status
        print(f"{k['slug']:<24} | {k['type']:<18} | {status:<15} | {k['name']}")

    # Check auto-trigger rules for completed parents
    # 1. Baseline k01 complete -> retrigger k02, k09
    if readings.get("cuq6-baseline-k01") == "COMPLETE":
        for dep in ["cuq6-uncertainty-k02", "cuq6-sacp-k09"]:
            if readings.get(dep) in ["ERROR", "UNKNOWN"]:
                print(f"[AUTO-TRIGGER] Baseline trained! Pushing {dep}...")
                ok = retrigger_kernel(dep)
                if ok:
                    triggered.append(dep)
                    readings[dep] = "TRIGGERED"

    # 2. MultiCP k06 complete -> retrigger k07, k08
    if readings.get("cuq6-multicp-k06") == "COMPLETE":
        for dep in ["cuq6-multicp-uq-k07", "cuq6-multicp-sacp-k08"]:
            if readings.get(dep) in ["ERROR", "UNKNOWN"]:
                print(f"[AUTO-TRIGGER] MultiCP trained! Pushing {dep}...")
                ok = retrigger_kernel(dep)
                if ok:
                    triggered.append(dep)
                    readings[dep] = "TRIGGERED"

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
