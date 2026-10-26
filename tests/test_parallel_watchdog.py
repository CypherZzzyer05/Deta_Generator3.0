"""Finite observation fixtures; no native solver or process termination."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deta_generator3 import watchdog


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.job = Path(self.temp.name)
        self.identity = "TEST-ONLY-task"
        self.log_mtime = 1001.
        self.percent = 40.
        self.shutoff = .001
        self.engine = False
        self.incomplete = False
        self.pid_offset = 0
        self.worker_start = 900.
        self.sample_start = 1000.
        self.sample_offset = 0
        self.write_manifests()

    def write_manifests(self):
        for index in range(8):
            folder = self.job/f"shard{index}"
            folder.mkdir(exist_ok=True)
            entry = dict(id=index+self.sample_offset, state="running", started_at=stamp(self.sample_start),
                         geometry_values=[700., 350., 350., 250., 180., 0., 0.])
            (folder/"manifest.json").write_text(json.dumps(dict(samples=[entry])))

    def snapshot(self, now):
        workers, samples = [], []
        for index in range(8):
            pid = 100+index+self.pid_offset
            workers.append(dict(index=index, pid=pid, started_at_unix=self.worker_start,
                alive=True, identity_verified=True, issues=["AccessDenied"] if self.incomplete and index == 0 else [],
                engine_pids=[900+index] if self.engine else []))
            samples.append(dict(shard_index=index, sample_id=index+self.sample_offset, worker_pid=pid,
                worker_alive=True, log_read_error=None, percent_complete=self.percent,
                auto_shutoff=self.shutoff, log_age_seconds=now-self.log_mtime,
                log_file=f"sample_{index+self.sample_offset:06d}_p0.log"))
        return dict(task_identity=self.identity, workers=workers, active_samples=samples,
                    native_engines=[dict(pid=900, issues=[])] if self.engine else [])

    def observe(self, now, timeout=600.):
        with patch.object(watchdog, "collect_status", return_value=self.snapshot(now)):
            return watchdog.observe(self.job, 10, timeout, now=now, process_api=object())

    def receipt(self, modified, status=2, geometry=None):
        folder = self.job/"shard0/projects"
        folder.mkdir(exist_ok=True)
        path = folder/f"sample_{self.sample_offset:06d}.attempts.json"
        path.write_text(json.dumps(dict(status=status, geometry_values=(geometry if geometry is not None
            else [700., 350., 350., 250., 180., 0., 0.]))))
        os.utime(path, (modified, modified))
        return path

    def test_eight_live_workers_no_engines_need_continuous_observations_before_trip(self):
        originals = {p: p.read_bytes() for p in self.job.rglob("manifest.json")}
        first = self.observe(2000.)
        self.assertEqual(first["stalled_workers"], [])
        for now in range(2030, 2601, 30):
            result = self.observe(float(now))
        self.assertEqual(len(result["stalled_workers"]), 8)
        self.assertTrue(all(row["no_progress_observed_seconds"] == 600 for row in result["stalled_workers"]))
        self.assertTrue(all(path.read_bytes() == content for path, content in originals.items()))
        self.assertEqual({p.name for p in self.job.iterdir() if p.is_file()}, {watchdog.STATE_NAME})

    def test_healthy_engine_and_new_progress_each_reset_the_timer(self):
        for now in range(2000, 2300, 30):
            self.observe(float(now))
        self.engine = True
        observed = self.observe(2300.)
        self.assertTrue(all(r["no_progress_observed_since_unix"] is None for r in observed["workers"]))
        self.engine = False
        for now in range(2330, 2601, 30):
            result = self.observe(float(now))
        self.assertEqual(result["stalled_workers"], [])
        self.percent = 41.
        progressed = self.observe(2630.)
        self.assertEqual(progressed["workers"][0]["no_progress_observed_since_unix"], 2630.)
        self.shutoff = .0001
        changed = self.observe(2660.)
        self.assertEqual(changed["workers"][0]["no_progress_observed_since_unix"], 2660.)
        self.log_mtime = 1002.
        changed = self.observe(2690.)
        self.assertEqual(changed["workers"][0]["no_progress_observed_since_unix"], 2690.)

    def test_old_interrupted_log_on_a_new_worker_is_never_current_solve_evidence(self):
        self.observe(2000.)
        self.pid_offset = 1000
        self.worker_start = 2050.
        self.sample_start = 2060.
        self.write_manifests()
        for now in range(2070, 2701, 30):
            result = self.observe(float(now))
        self.assertEqual(result["stalled_workers"], [])
        self.assertTrue(all(r["phase"] == "older_attempt_log_ignored" for r in result["workers"]))
        self.assertTrue(all(r["no_progress_observed_since_unix"] is None for r in result["workers"]))

    def test_identity_sample_and_start_time_changes_do_not_inherit_old_timer(self):
        self.observe(2000.)
        self.pid_offset = 1000
        self.worker_start = 901.
        changed = self.observe(2030.)
        self.assertEqual(changed["workers"][0]["no_progress_observed_since_unix"], 2030.)
        self.sample_offset = 8
        self.write_manifests()
        changed = self.observe(2060.)
        self.assertEqual(changed["workers"][0]["no_progress_observed_since_unix"], 2060.)
        self.sample_start = 1000.5
        self.write_manifests()
        changed = self.observe(2090.)
        self.assertEqual(changed["workers"][0]["no_progress_observed_since_unix"], 2090.)

    def test_observation_gap_task_or_timeout_change_resets_continuity(self):
        self.observe(2000.)
        self.assertEqual(self.observe(2300.)["workers"][0]["no_progress_observed_since_unix"], 2300.)
        self.identity = "another-TEST-task"
        self.assertEqual(self.observe(2330.)["workers"][0]["no_progress_observed_since_unix"], 2330.)
        self.assertEqual(self.observe(2360., 300.)["workers"][0]["no_progress_observed_since_unix"], 2360.)
        self.assertEqual(self.observe(2350., 300.)["workers"][0]["no_progress_observed_since_unix"], 2350.)

    def test_incomplete_observations_reset_all_timers(self):
        self.observe(2000.)
        self.incomplete = True
        result = self.observe(2030.)
        self.assertIsNotNone(result["observation_error"])
        self.assertEqual(result["workers"], [])
        self.incomplete = False
        result = self.observe(2060.)
        self.assertEqual(result["workers"][0]["no_progress_observed_since_unix"], 2060.)

    def test_fresh_successful_receipt_exempts_result_collection_but_old_or_other_geometry_does_not(self):
        self.receipt(1002.)
        result = self.observe(2000.)
        self.assertEqual(result["workers"][0]["phase"], "successful_solve_result_collection")
        self.receipt(999.)
        result = self.observe(2030.)
        self.assertEqual(result["workers"][0]["phase"], "waiting_without_observed_engine")
        self.receipt(1002., geometry=[1., 2.])
        result = self.observe(2060.)
        self.assertEqual(result["workers"][0]["phase"], "waiting_without_observed_engine")

    def test_recent_current_log_and_finished_percent_do_not_immediately_trip(self):
        self.log_mtime = 1995.
        result = self.observe(2000.)
        self.assertEqual(result["stalled_workers"], [])
        self.percent = 100.
        result = self.observe(2030.)
        self.assertTrue(all(r["no_progress_observed_since_unix"] is None for r in result["workers"]))

    def test_cli_reserves_exit_two_for_stall_and_resets_on_observation_error(self):
        args = ["--job-dir", str(self.job), "--expected-jobs", "10", "--timeout-seconds", "600"]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for result, expected in ((dict(observation_error=None, stalled_workers=[]), 0),
                    (dict(observation_error="incomplete", stalled_workers=[]), 1),
                    (dict(observation_error=None, stalled_workers=[dict(index=0, sample_id=0, pid=100,
                        no_progress_observed_seconds=600.)]), 2)):
                with patch.object(watchdog, "observe", return_value=result):
                    self.assertEqual(watchdog.main(args), expected)
            self.assertEqual(watchdog.main(["v"]), 1)
            with patch.object(watchdog, "observe", side_effect=OSError("TEST-only visibility error")):
                self.assertEqual(watchdog.main(args), 1)
        reset = json.loads((self.job/watchdog.STATE_NAME).read_text())
        self.assertEqual(reset["workers"], [])
        self.assertEqual(reset["observation_error"], "OSError")


if __name__ == "__main__":
    unittest.main()
