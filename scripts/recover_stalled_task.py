"""Preserve and release a stalled Generator3 task; never restart it automatically.

Stop the original parallel controller with Ctrl+C before invoking this helper.
Requires the existing installed deta_generator3 package and psutil.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import psutil
from deta_generator3.monitor import collect_status


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_snapshot(job, process_api):
    layout = read(job/"parallel_plan.json")
    launch = read(job/"parallel_launch.json") if (job/"parallel_launch.json").is_file() else {}
    snapshot = collect_status(job, launch.get("requested_concurrency", layout["workers"]), process_api=process_api)
    live = [r for r in snapshot["workers"] if r["alive"] is True]
    if not live or any(r["alive"] is None or r["issues"] for r in snapshot["workers"]):
        raise RuntimeError("No confirmed live workers, or process observations are incomplete; inspect instead of stopping")
    if any(not r["identity_verified"] for r in live):
        raise RuntimeError("Worker creation identity is missing; refusing to stop processes")
    if snapshot["native_engine_count"]:
        raise RuntimeError("A task FDTD engine exists; refusing stalled-task recovery")
    # Include engines whose launcher detached from the recorded worker tree.
    for process in process_api.process_iter():
        try:
            if process.name().lower().startswith("fdtd-engine"):
                raise RuntimeError("A native FDTD engine exists on this machine; inspect before recovery")
        except (process_api.NoSuchProcess, process_api.ZombieProcess):
            continue
    samples = snapshot["active_samples"]
    if ({r["shard_index"] for r in samples} != {r["index"] for r in live}
            or len(samples) != len(live)
            or any(r["worker_alive"] is not True or r["log_read_error"]
                   or r["log_age_seconds"] is None or r["log_age_seconds"] < 3600
                   or r["percent_complete"] is None or not 0 <= r["percent_complete"] < 100 for r in samples)):
        raise RuntimeError("Need one unfinished, >=1-hour stale solver log per confirmed worker; refusing to stop")
    return snapshot, live


def confirmed_process(row, process_api):
    process = process_api.Process(row["pid"])
    name = process.name().lower().removesuffix(".exe")
    if (name != row["process_name"].lower().removesuffix(".exe")
            or abs(process.create_time()-row["started_at_unix"]) > 2):
        raise RuntimeError(f"PID {row['pid']} identity changed; refusing to stop it")
    return process


def recover(job_dir, process_api=psutil):
    job = Path(job_dir).resolve()
    # The original PS controller holds this file with FileShare.None. Opening
    # it fails while that controller is active and also prevents a new one.
    with (job/".parallel-controller.lock").open("r+b"):
        before, live = checked_snapshot(job, process_api)
        workers = [confirmed_process(r, process_api) for r in live]
        owned_cads = {}
        for worker in workers:
            for child in worker.children(recursive=True):
                if child.name().lower() == "fdtd-solutions.exe":
                    owned_cads[child.pid] = dict(pid=child.pid, process_name=child.name(),
                                                started_at_unix=child.create_time())
        for sample in before["active_samples"]:
            path = job/f"shard{sample['shard_index']}"/"samples"/f"sample_{sample['sample_id']:06d}.npz"
            if path.exists():
                raise RuntimeError(f"Uncommitted numeric sample exists:{path}; inspect it before recovery")
        archive = Path(tempfile.mkdtemp(prefix="recovery_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") , dir=job))
        copies = []

        def preserve(source):
            source = source.resolve()
            relative = source.relative_to(job)
            target = archive/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha(source)
            if sha(target) != digest:
                raise RuntimeError("Backup SHA mismatch; no process stopped")
            copies.append(dict(path=str(relative), sha256=digest, bytes=source.stat().st_size))

        for name in ("job.json", "parallel_plan.json", "parallel_workers.json", "parallel_launch.json", "parallel_status.json"):
            if (job/name).is_file():
                preserve(job/name)
        for path in (job/"parallel_logs").glob("*.log"):
            preserve(path)
        for index in range(before["shard_count"]):
            path = job/f"shard{index}"/"manifest.json"
            if path.is_file():
                preserve(path)
        for sample in before["active_samples"]:
            shard = job/f"shard{sample['shard_index']}"
            entry = next(e for e in read(shard/"manifest.json")["samples"] if e["id"] == sample["sample_id"])
            directory = (shard/"projects"/entry.get("project_subdir", "")).resolve()
            directory.relative_to(job)
            for path in directory.glob(f"sample_{sample['sample_id']:06d}*"):
                if path.is_file():
                    preserve(path)
            if (shard/"generation.log").is_file():
                preserve(shard/"generation.log")
        after, still_live = checked_snapshot(job, process_api)
        # Log ages naturally advance; compare source identities and progress.
        keys = ("shard_index", "sample_id", "worker_pid", "percent_complete", "auto_shutoff", "log_file")
        if (after["totals"] != before["totals"] or
                [[r.get(k) for k in keys] for r in after["active_samples"]] !=
                [[r.get(k) for k in keys] for r in before["active_samples"]] or
                [(r["pid"], r["started_at_unix"]) for r in still_live] != [(r["pid"], r["started_at_unix"]) for r in live]):
            raise RuntimeError("Task progressed or identities changed during backup; no process stopped")
        receipt = dict(state="backed_up_before_stop", job_dir=str(job), before=before, copied_files=copies,
                       owned_cads=list(owned_cads.values()), stopped_pids=[])
        receipt_path = archive/"recovery.json"
        receipt_path.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        targets = live+list(owned_cads.values())
        stopped = []
        for row in targets:
            try:
                process = confirmed_process(row, process_api)
                process.terminate()
                stopped.append(process)
                receipt["stopped_pids"].append(row["pid"])
                receipt_path.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
            except process_api.NoSuchProcess:
                continue
        _, alive = process_api.wait_procs(stopped, timeout=10)
        if alive:
            raise RuntimeError("Some recorded task processes remain alive; do not restart the task")
        receipt["state"] = "preserved_and_released_no_restart"
        receipt_path.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(dict(state=receipt["state"], archive=str(archive),
                              stopped_pids=receipt["stopped_pids"], counts=before["totals"])), flush=True)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        recover(args.job_dir)
    except (OSError, ValueError, RuntimeError, psutil.Error) as error:
        parser.exit(1, f"Recovery stopped ({type(error).__name__}): {error}\nTask not restarted automatically.\n")
