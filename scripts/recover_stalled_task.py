"""Preserve and release a Generator3 task; never restart it automatically.

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


def checked_snapshot(job, process_api, stop_active=False):
    layout = read(job/"parallel_plan.json")
    launch = read(job/"parallel_launch.json") if (job/"parallel_launch.json").is_file() else {}
    snapshot = collect_status(job, launch.get("requested_concurrency", layout["workers"]), process_api=process_api)
    live = [r for r in snapshot["workers"] if r["alive"] is True]
    if (not live and not stop_active) or any(r["alive"] is None or r["issues"] for r in snapshot["workers"]):
        raise RuntimeError("No confirmed live workers, or process observations are incomplete; inspect instead of stopping")
    if any(not r["identity_verified"] for r in live):
        raise RuntimeError("Worker creation identity is missing; refusing to stop processes")
    if stop_active:
        return snapshot, live
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


def recover(job_dir, process_api=psutil, stop_active=False):
    job = Path(job_dir).resolve()
    # The original PS controller holds this file with FileShare.None. Opening
    # it fails while that controller is active and also prevents a new one.
    with (job/".parallel-controller.lock").open("r+b"):
        before, live = checked_snapshot(job, process_api, stop_active)
        workers = [confirmed_process(r, process_api) for r in live]
        owned_children = {}
        for worker in workers:
            for child in worker.children(recursive=True):
                name = child.name().lower()
                if name == "fdtd-solutions.exe" or (stop_active and name.startswith("fdtd-engine")):
                    owned_children[child.pid] = dict(pid=child.pid, process_name=child.name(),
                                                    started_at_unix=child.create_time())
        for sample in before["active_samples"]:
            path = job/f"shard{sample['shard_index']}"/"samples"/f"sample_{sample['sample_id']:06d}.npz"
            if path.exists() and not stop_active:
                raise RuntimeError(f"Uncommitted numeric sample exists:{path}; inspect it before recovery")
        archive = Path(tempfile.mkdtemp(prefix="recovery_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") , dir=job))
        copies = []
        receipt = dict(state="stopping_user_requested_batch" if stop_active else "preparing_stalled_recovery",
                       reason="user_requested_stop_active" if stop_active else "verified_stalled_task",
                       job_dir=str(job), before=before, copied_files=copies,
                       owned_children=list(owned_children.values()), stopped_pids=[])
        receipt_path = archive/"recovery.json"

        def save_receipt():
            receipt_path.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")

        def stop_owned():
            # Stop Python first so it cannot advance to another geometry.
            # Native child identities were captured from its verified tree.
            targets = live+sorted(owned_children.values(), key=lambda r: r["process_name"].lower() == "fdtd-solutions.exe")
            stopped = []
            for row in targets:
                try:
                    process = confirmed_process(row, process_api)
                    process.terminate()
                    stopped.append(process)
                    receipt["stopped_pids"].append(row["pid"])
                    save_receipt()
                except process_api.NoSuchProcess:
                    continue
            _, alive = process_api.wait_procs(stopped, timeout=10)
            if alive:
                raise RuntimeError("Some recorded task processes remain alive; do not restart the task")

        save_receipt()
        if stop_active:
            # An explicit stop is permitted while solver logs are changing.
            # Release all owned writers before copying their evidence.
            stop_owned()
            receipt["state"] = "user_requested_processes_stopped_preserving_files"
            save_receipt()

        def preserve(source):
            source = source.resolve()
            relative = source.relative_to(job)
            target = archive/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha(source)
            if sha(target) != digest:
                raise RuntimeError("Backup SHA mismatch; inspect recovery.json before any restart")
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
        samples = {(s["shard_index"], s["sample_id"]) for s in before["active_samples"]}
        # A sample may have finished while the stop request was being observed.
        # Include the now-current interrupted row without changing any manifest.
        if stop_active:
            for index in range(before["shard_count"]):
                path = job/f"shard{index}"/"manifest.json"
                if path.is_file():
                    samples.update((index, e["id"]) for e in read(path)["samples"] if e["state"] == "running")
        for index, sample_id in sorted(samples):
            shard = job/f"shard{index}"
            entry = next(e for e in read(shard/"manifest.json")["samples"] if e["id"] == sample_id)
            directory = (shard/"projects"/entry.get("project_subdir", "")).resolve()
            directory.relative_to(job)
            for path in directory.glob(f"sample_{sample_id:06d}*"):
                if path.is_file():
                    preserve(path)
            if (shard/"generation.log").is_file():
                preserve(shard/"generation.log")
            numeric = shard/"samples"/f"sample_{sample_id:06d}.npz"
            if stop_active and entry["state"] != "success" and numeric.is_file():
                preserve(numeric)
        if not stop_active:
            after, still_live = checked_snapshot(job, process_api)
            # Log ages naturally advance; compare source identities and progress.
            keys = ("shard_index", "sample_id", "worker_pid", "percent_complete", "auto_shutoff", "log_file")
            if (after["totals"] != before["totals"] or
                    [[r.get(k) for k in keys] for r in after["active_samples"]] !=
                    [[r.get(k) for k in keys] for r in before["active_samples"]] or
                    [(r["pid"], r["started_at_unix"]) for r in still_live] != [(r["pid"], r["started_at_unix"]) for r in live]):
                raise RuntimeError("Task progressed or identities changed during backup; no process stopped")
            receipt["state"] = "backed_up_before_stop"
            save_receipt()
            stop_owned()
        receipt["state"] = "preserved_and_released_no_restart"
        save_receipt()
        print(json.dumps(dict(state=receipt["state"], archive=str(archive),
                              stopped_pids=receipt["stopped_pids"], counts=before["totals"])), flush=True)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--stop-active", action="store_true",
                        help="Explicitly stop this task's live solvers, then preserve interrupted evidence; no restart")
    args = parser.parse_args()
    try:
        recover(args.job_dir, stop_active=args.stop_active)
    except (OSError, ValueError, RuntimeError, psutil.Error) as error:
        parser.exit(1, f"Recovery stopped ({type(error).__name__}): {error}\nTask not restarted automatically.\n")
