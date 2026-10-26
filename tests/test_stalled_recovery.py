"""Recovery fixtures never start/stop real solver or Python processes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/recover_stalled_task.py"
spec = importlib.util.spec_from_file_location("stalled_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class ProcessFixture:
    def __init__(self, pid, name="python.exe", children=()):
        self.pid, self.label, self.kids, self.created, self.alive = pid, name, children, 1000., True

    def name(self): return self.label
    def create_time(self): return self.created
    def children(self, recursive=True): return self.kids
    def terminate(self): self.alive = False


class StalledRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cad = ProcessFixture(303, "fdtd-solutions.exe")
        self.workers = [ProcessFixture(101, children=[self.cad]), ProcessFixture(102)]
        self.other = [ProcessFixture(202), ProcessFixture(6176, "hydra_service.exe")]
        processes = {p.pid: p for p in [*self.workers, self.cad, *self.other]}
        self.api = SimpleNamespace(Process=lambda pid: processes[pid], process_iter=lambda: list(processes.values()),
            NoSuchProcess=psutil.NoSuchProcess, ZombieProcess=psutil.ZombieProcess,
            wait_procs=lambda procs, timeout: ([p for p in procs if not p.alive], [p for p in procs if p.alive]))
        self.snapshot = dict(totals=dict(success=1, failed=1, running=2, pending=0), shard_count=2,
            native_engine_count=0, workers=[dict(index=i, pid=101+i, process_name="python.exe",
                started_at_unix=1000., alive=True, issues=[], identity_verified=True) for i in range(2)],
            active_samples=[dict(shard_index=i, sample_id=i, worker_pid=101+i, worker_alive=True,
                log_read_error=None, log_age_seconds=79000., percent_complete=40., auto_shutoff=.001,
                log_file=f"sample_{i:06d}_p0.log") for i in range(2)])
        (self.root/".parallel-controller.lock").touch()
        (self.root/"parallel_plan.json").write_text(json.dumps(dict(workers=2)))
        (self.root/"job.json").write_text(json.dumps(dict(sample_count=4)))
        (self.root/"parallel_workers.json").write_text(json.dumps(dict(workers=[])))
        (self.root/"parallel_logs").mkdir()
        (self.root/"parallel_logs/shard0.stdout.log").write_text("TEST-ONLY old worker output")
        for i in range(2):
            shard = self.root/f"shard{i}"
            (shard/"projects").mkdir(parents=True)
            (shard/"samples").mkdir()
            (shard/"projects"/f"sample_{i:06d}.fsp").write_bytes(b"TEST-ONLY interrupted project")
            (shard/"projects"/f"sample_{i:06d}_p0.log").write_text("TEST-ONLY solver diagnostic")
            (shard/"manifest.json").write_text(json.dumps(dict(samples=[dict(id=i, state="running"),
                dict(id=i+2, state="success" if i == 0 else "failed") ])))
            (shard/"generation.log").write_text("TEST-ONLY interrupted call")
        (self.root/"shard0/samples/sample_000002.npz").write_bytes(b"TEST-ONLY committed sample; never copied/modified")

    def test_preserves_partial_evidence_and_stops_only_registered_processes(self):
        originals = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(recovery, "collect_status", return_value=self.snapshot):
            result = recovery.recover(self.root, self.api)
        self.assertEqual(set(result["stopped_pids"]), {101, 102, 303})
        self.assertTrue(all(p.alive for p in self.other))
        archive = next(self.root.glob("recovery_*"))
        self.assertEqual((archive/"shard0/projects/sample_000000.fsp").read_bytes(), b"TEST-ONLY interrupted project")
        self.assertFalse((archive/"shard0/samples").exists())
        for path, value in originals.items():
            self.assertEqual(path.read_bytes(), value)
        for copy in result["copied_files"]:
            self.assertEqual(recovery.sha(archive/copy["path"]), copy["sha256"])

    def test_refuses_recent_log_or_completed_solver_phase(self):
        for key, value in (("log_age_seconds", 30), ("percent_complete", 100)):
            snapshot = deepcopy(self.snapshot)
            snapshot["active_samples"][0][key] = value
            with patch.object(recovery, "collect_status", return_value=snapshot):
                with self.assertRaisesRegex(RuntimeError, "unfinished"):
                    recovery.recover(self.root, self.api)
            self.assertTrue(all(p.alive for p in self.workers))

    def test_refuses_native_engine_or_reused_worker_pid(self):
        snapshot = deepcopy(self.snapshot)
        snapshot["native_engine_count"] = 1
        with patch.object(recovery, "collect_status", return_value=snapshot):
            with self.assertRaisesRegex(RuntimeError, "engine"):
                recovery.recover(self.root, self.api)
        self.workers[0].created = 2000.
        with patch.object(recovery, "collect_status", return_value=self.snapshot):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                recovery.recover(self.root, self.api)
        self.assertTrue(all(p.alive for p in self.workers))

    def test_rejects_uncommitted_numeric_sample_and_progress_during_backup(self):
        pending = self.root/"shard0/samples/sample_000000.npz"
        pending.write_bytes(b"TEST-ONLY uncommitted numeric data")
        with patch.object(recovery, "collect_status", return_value=self.snapshot):
            with self.assertRaisesRegex(RuntimeError, "Uncommitted numeric sample"):
                recovery.recover(self.root, self.api)
        pending.unlink()
        changed = deepcopy(self.snapshot)
        changed["active_samples"][0]["percent_complete"] = 41.
        with patch.object(recovery, "collect_status", side_effect=[self.snapshot, changed]):
            with self.assertRaisesRegex(RuntimeError, "progressed"):
                recovery.recover(self.root, self.api)
        self.assertTrue(all(p.alive for p in self.workers))


if __name__ == "__main__":
    unittest.main()
