#!/usr/bin/env python3
"""Checkpoint-preserving reprioritization of GPU 3, then restore its sweep job."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
SWEEP = PROJECT / "outputs/cmts_tau_sweep_20260922_a8_v05_lam100"
SELECTED = SWEEP / "tau1.25"
PAUSED = SWEEP / "tau1.5"
JOB = SWEEP / "extend_tau1.25_sim000_T1000"


def status(stage, **extra):
    data = dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra)
    temporary = JOB / "status.tmp"
    temporary.write_text(json.dumps(data,indent=2))
    temporary.replace(JOB / "status.json")
    print(json.dumps(data),flush=True)


def command(run, scale, first, last, rounds, partial):
    return ["conda","run","-n","diverse","--no-capture-output","python","scripts/73_cmts_dreamsim.py",
            "--model_path","models/stabilityai/stable-diffusion-3.5-large","--device","cuda:0",
            "--B_word","bright","--B_seed","18","--dim","16","--k","10","--B","8","--n0","24",
            "--v","0.5","--S","8","--lam","100","--alpha","8","--ref_seed","1810772",
            "--batch_size","8","--save_img_every","20","--partial_id",str(partial),"--tau_scale",str(scale),
            "--out_root",str(run),"--seed_start",str(first),"--seed_end",str(last),"--T",str(rounds)]


def worker():
    extension_error = None
    try:
        for rounds in [400,600,1000]:
            status("extending_selected",target_rounds=rounds)
            subprocess.run(command(SELECTED,1.25,0,1,rounds,200),cwd=PROJECT,check=True)
            out = PROJECT / f"outputs/continuous_diagnostics/continuous_tau1.25_sim000_T{rounds}"
            status("evaluating_selected",target_rounds=rounds)
            subprocess.run(["conda","run","-n","diverse","--no-capture-output","python",
                            "scripts/125_evaluate_tau_trajectory.py","--run",str(SELECTED),"--sim","0",
                            "--reference-json",str(JOB/"fixed_reference.json"),"--output",str(out)],cwd=PROJECT,check=True)
            subprocess.run(["conda","run","-n","diverse","--no-capture-output","python",
                            "scripts/126_replot_variance_normalized_mse.py","--evaluation",str(out)],cwd=PROJECT,check=True)
    except Exception as exc:
        extension_error = repr(exc)
        status("extension_error",error=extension_error)
    # Restore the earlier experiment even if the extension/evaluation fails.
    status("resuming_tau1.5",extension_error=extension_error)
    with (PAUSED / "resume_after_extension.log").open("a") as log:
        result = subprocess.run(command(PAUSED,1.5,0,5,200,201),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
    (PAUSED / "exit_code.txt").write_text(str(result.returncode)+"\n")
    status("finished" if result.returncode == 0 and extension_error is None else "finished_with_errors",
           extension_error=extension_error,resumed_sweep_exit_code=result.returncode)


def launch():
    if (JOB / "launcher.json").exists():
        raise RuntimeError("Continuation already launched; refusing duplicate GPU worker")
    with (SWEEP / "launch_manifest.tsv").open() as f:
        entry = next(row for row in csv.DictReader(f,delimiter="\t") if row["gpu"]=="3")
    pid = int(entry["pid"])
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
    if "tau-worker" not in cmd or "tau1.5" not in cmd or os.getpgid(pid)!=pid:
        raise RuntimeError("GPU 3 process identity differs from launch manifest; refusing to signal")
    # Only the currently assigned seed may be interrupted; its atomic checkpoint
    # is resumed later. Completed seed 0 of the selected setting has no active writer.
    if not (SELECTED / "sim000/summary.json").exists():
        raise RuntimeError("Selected seed has not finished its original run")
    import pickle
    with (SELECTED / "sim000/_ckpt.pkl").open("rb") as f:
        selected_checkpoint = pickle.load(f)
    if selected_checkpoint["t_done"] != 200:
        raise RuntimeError("Selected checkpoint is not at the expected 200-round boundary")
    before=[]
    for checkpoint in sorted(PAUSED.glob("sim*/_ckpt.pkl")):
        with checkpoint.open("rb") as f: saved=pickle.load(f)
        before.append(dict(sim=checkpoint.parent.name,rounds=saved["t_done"]))
    JOB.mkdir(parents=True,exist_ok=True)
    reference = PROJECT / "outputs/continuous_diagnostics/continuous_tau1.25_sim000/evaluation.json"
    (JOB / "fixed_reference.json").write_text(reference.read_text())
    (JOB / "paused_checkpoints.json").write_text(json.dumps(before,indent=2))
    os.killpg(pid,signal.SIGTERM)
    # Confirm the original worker group no longer contains running processes.
    for _ in range(60):
        entries=subprocess.check_output(["ps","-eo","pgid=,stat="],text=True).splitlines()
        live=[row for row in entries if len(row.split())>=2 and row.split()[0]==str(pid) and not row.split()[1].startswith("Z")]
        if not live:break
        time.sleep(1)
    else:
        raise RuntimeError("Original GPU 3 worker group has not stopped; no new job launched")
    old_exit = PAUSED / "exit_code.txt"
    if old_exit.exists(): old_exit.replace(JOB / "paused_worker_exit_code.txt")
    (PAUSED / "PAUSED_FOR_EXTENSION.txt").write_text(f"Will resume automatically after {JOB}.\n")
    env=os.environ.copy()
    env.pop("LD_LIBRARY_PATH",None)
    env.update(CUDA_VISIBLE_DEVICES="3",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
               MPLCONFIGDIR="/tmp/cmts_matplotlib",PYTHONUNBUFFERED="1")
    with (JOB / "worker.log").open("w") as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker"],cwd=PROJECT,
                               env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=child.pid,gpu=3,paused_launcher_pid=pid,paused_checkpoints=before,
              milestones=[400,600,1000],selected_run=str(SELECTED),selected_sim=0,
              fixed_oracle_probability=json.loads(reference.read_text())["oracle"]["probability"])
    (JOB / "launcher.json").write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker",action="store_true")
    args=ap.parse_args()
    worker() if args.worker else launch()
