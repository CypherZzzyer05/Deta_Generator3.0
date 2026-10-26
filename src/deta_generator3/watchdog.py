"""Scheduling-only observations of stalled native solves; never stop a process.

A fresh successful-attempt receipt exempts result collection. A log from an
older attempt cannot establish a current solve, so initialization without a
current log remains an observation rather than a wall-clock kill timeout.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import psutil

from .monitor import _open_shared_read, _read_json, collect_status


STATE_NAME = "watchdog_status.json"
SCHEMA_VERSION = 1
MAX_OBSERVATION_GAP_SECONDS = 90.


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sample_start(entry):
    text = entry.get("started_at")
    if not isinstance(text, str):
        return None
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Current sample started_at needs an explicit time zone")
    return stamp.timestamp()


def _successful_attempt(shard, entry, started):
    path = shard/"projects"/entry.get("project_subdir", "")/f"sample_{entry['id']:06d}.attempts.json"
    try:
        with _open_shared_read(path, binary=True) as stream:
            contents = stream.read()
            modified = os.fstat(stream.fileno()).st_mtime
    except FileNotFoundError:
        return False
    receipt = json.loads(contents.decode("utf-8-sig"))
    return (modified >= started and receipt.get("status") == 2
            and receipt.get("geometry_values") == entry.get("geometry_values")
            and "geometry_values" in entry)


def _write_state(job, state):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=job,
                                         prefix=".watchdog-status-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, job/STATE_NAME)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _base_state(now, timeout, identity=None):
    return dict(schema_version=SCHEMA_VERSION, task_identity=identity, observed_at_unix=now,
                timeout_seconds=timeout, observation_error=None, workers=[], stalled_workers=[])


def _same_progress(left, right):
    return all(left.get(key) == right.get(key) for key in
               ("log_file", "percent_complete", "auto_shutoff", "log_modified_at_unix"))


def observe(job_dir, expected_jobs, timeout_seconds=600., now=None, process_api=psutil):
    """Persist only independent watchdog state, returning no process side effects."""
    job = Path(job_dir).resolve()
    now = time.time() if now is None else now
    if not _finite(timeout_seconds) or timeout_seconds <= 0 or not _finite(now):
        raise ValueError("Watchdog timeout must be finite/positive and observation time finite")
    previous_path = job/STATE_NAME
    previous = _read_json(previous_path) if previous_path.is_file() else {}
    snapshot = collect_status(job, expected_jobs, now=now, process_api=process_api)
    identity = snapshot["task_identity"]
    state = _base_state(now, timeout_seconds, identity)
    incomplete = (any(row.get("issues") or row.get("alive") is None
                      or (row.get("alive") is True and not row.get("identity_verified"))
                      for row in snapshot["workers"])
                  or any(row.get("issues") for row in snapshot["native_engines"]))
    if incomplete:
        state["observation_error"] = "Process identity/visibility observations are incomplete; watchdog timers reset"
        _write_state(job, state)
        return state
    previous_time = previous.get("observed_at_unix")
    continuous = (previous.get("schema_version") == SCHEMA_VERSION
                  and previous.get("task_identity") == identity
                  and previous.get("timeout_seconds") == timeout_seconds
                  and _finite(previous_time) and 0 <= now-previous_time <= MAX_OBSERVATION_GAP_SECONDS
                  and not previous.get("observation_error"))
    old_rows = {row["index"]: row for row in previous.get("workers", [])} if continuous else {}
    active = {row["shard_index"]: row for row in snapshot["active_samples"]}
    for worker in snapshot["workers"]:
        if worker["alive"] is not True:
            continue
        row = dict(index=worker["index"], pid=worker["pid"], started_at_unix=worker["started_at_unix"],
                   phase="no_current_sample", sample_id=None, sample_started_at_unix=None,
                   no_progress_observed_since_unix=None)
        sample = active.get(worker["index"])
        if sample is None or sample.get("worker_pid") != worker["pid"] or sample.get("worker_alive") is not True:
            state["workers"].append(row)
            continue
        row["sample_id"] = sample["sample_id"]
        shard = job/f"shard{worker['index']}"
        manifest = _read_json(shard/"manifest.json")
        entry = next((e for e in manifest["samples"] if e["id"] == sample["sample_id"]), None)
        if entry is None or entry["state"] != "running":
            row["phase"] = "sample_changed_during_observation"
        elif worker.get("engine_pids"):
            row["phase"] = "native_engine_observed"
        else:
            started = _sample_start(entry)
            row["sample_started_at_unix"] = started
            percent, age = sample.get("percent_complete"), sample.get("log_age_seconds")
            if started is not None and _successful_attempt(shard, entry, started):
                row["phase"] = "successful_solve_result_collection"
            elif (started is None or sample.get("log_read_error") or not _finite(percent)
                  or not 0 <= percent < 100 or not _finite(age) or age < 0):
                row["phase"] = "current_solver_progress_unavailable"
            else:
                modified = now-age
                row.update(log_file=sample.get("log_file"), percent_complete=percent,
                           auto_shutoff=sample.get("auto_shutoff"), log_modified_at_unix=round(modified, 3),
                           log_age_seconds=age)
                if modified < max(started, worker["started_at_unix"]):
                    row["phase"] = "older_attempt_log_ignored"
                else:
                    old = old_rows.get(worker["index"], {})
                    same_attempt = all(old.get(key) == row[key] for key in
                        ("pid", "started_at_unix", "sample_id", "sample_started_at_unix"))
                    since = old.get("no_progress_observed_since_unix")
                    if not (same_attempt and _same_progress(old, row) and _finite(since) and since <= now):
                        since = now
                    row.update(phase="waiting_without_observed_engine",
                               no_progress_observed_since_unix=since,
                               no_progress_observed_seconds=now-since)
                    if now-since >= timeout_seconds and age >= timeout_seconds:
                        row["phase"] = "stalled_observations"
                        state["stalled_workers"].append(dict(row))
        state["workers"].append(row)
    _write_state(job, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--expected-jobs", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.)
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        # Reserve exit2 exclusively for a completed stalled-observation check.
        return 0 if error.code == 0 else 1
    try:
        result = observe(args.job_dir, args.expected_jobs, args.timeout_seconds)
        if result["observation_error"]:
            print("Watchdog observation unavailable: "+result["observation_error"], file=sys.stderr, flush=True)
            return 1
        if result["stalled_workers"]:
            details = "; ".join(f"shard{r['index']} sample_{r['sample_id']:06d} PID{r['pid']} "
                                f"unchanged for{r['no_progress_observed_seconds']:.0f}s"
                                for r in result["stalled_workers"])
            print("WATCHDOG_STALLED: "+details+". Stop scheduling new shards; preserve current data/workers. "
                  "Inspect the local license client (ansyscl), native diagnostics and resources; these observations "
                  "do not identify the cause. Use task-scoped stop/recovery before restarting.", flush=True)
            return 2
        return 0
    except (OSError, ValueError, KeyError, TypeError, psutil.Error) as error:
        # An observation error breaks continuity even if the next poll is soon.
        if args.job_dir.is_dir():
            try:
                state = _base_state(time.time(), args.timeout_seconds)
                state["observation_error"] = type(error).__name__
                _write_state(args.job_dir, state)
            except (OSError, ValueError):
                pass
        print(f"Watchdog observation unavailable ({type(error).__name__}): {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
