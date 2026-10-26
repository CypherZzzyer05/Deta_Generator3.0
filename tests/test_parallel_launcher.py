"""Real Windows subprocess/shard integration with explicit TEST-ONLY fields."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import psutil

ROOT = Path(__file__).resolve().parents[1]
SHELL = shutil.which("powershell") or shutil.which("pwsh")


@unittest.skipUnless(sys.platform == "win32" and SHELL, "Windows controller integration")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = self.root/"job with spaces"
        site = self.root/"site"
        site.mkdir()
        (self.root/"trace").mkdir()
        (site/"sitecustomize.py").write_text('''
import json, os, pathlib, sys, time
from test_generator import TestOnlySimulator
import deta_generator3.cli as cli
import deta_generator3.solver as solver
def native_check(config, preflight=False):
    report = {"recommended_ram_gib": .00001, "test_fixture_only": True}
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    (config.output_dir.parent/"preflight.json").write_text(json.dumps(report))
    return report
cli.native_check = native_check
class TimedSimulator(TestOnlySimulator):
    entered = False
    def run(self, geometry, project, starting_attempt=3):
        trace = pathlib.Path(os.environ["GENERATOR3_TEST_TRACE"])
        if not self.entered:
            self.entered = True
            (trace/f"{os.getpid()}.ready").write_text("TEST-ONLY")
            deadline = time.monotonic()+30
            while len(list(trace.glob("*.ready"))) < 2:
                if time.monotonic() > deadline:
                    raise RuntimeError("TEST-ONLY: two sample.run calls did not overlap")
                time.sleep(.02)
        start = time.time()
        time.sleep(.15)
        try:
            if os.environ.get("GENERATOR3_TEST_FAIL") and "--shard-index" in sys.argv:
                if sys.argv[sys.argv.index("--shard-index")+1] == "1":
                    raise ValueError("TEST-ONLY rejected spectrum")
            return super().run(geometry, project, starting_attempt)
        finally:
            (trace/f"{os.getpid()}_{pathlib.Path(project).stem}.json").write_text(
                json.dumps({"start": start, "end": time.time(), "test_fixture_only": True}))
solver.PairedSimulator = TimedSimulator
''', encoding="utf-8")
        self.env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(site), str(ROOT/"src"), str(ROOT/"tests")]),
            GENERATOR3_TEST_TRACE=str(self.root/"trace"), PYTHONIOENCODING="utf-8",
            OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def invoke(self, *extra, env=None):
        return subprocess.run([sys.executable, "-m", "deta_generator3", "parallel", "--job-dir", str(self.job),
            "--samples", "4", "--workers", "2", "--jobs", "2", "--threads", "1", *extra],
            env=env or self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", timeout=120)

    def test_real_controller_reaches_concurrent_runs_merges_and_resumes(self):
        run = self.invoke()
        self.assertEqual(run.returncode, 0, run.stdout)
        summary = json.loads((self.job/"merged/summary.json").read_text())
        self.assertEqual(summary["success"], 4)
        intervals = [json.loads(p.read_text()) for p in (self.root/"trace").glob("*.json")]
        self.assertEqual(len(intervals), 4)
        self.assertTrue(any(a["start"] < b["start"] < a["end"] for a in intervals for b in intervals))
        self.assertIn("native_engines=0", run.stdout)
        before = {p: p.read_bytes() for p in (self.job/"merged/samples").glob("*.npz")}
        resumed = self.invoke()
        self.assertEqual(resumed.returncode, 0, resumed.stdout)
        self.assertEqual(len(list((self.root/"trace").glob("*.json"))), 4)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        status = self.invoke("--parallel-action", "status")
        self.assertEqual(status.returncode, 0, status.stdout)
        self.assertIn("success=4/4", status.stdout)

    def test_quality_failures_are_merged_retained_and_not_automatically_retried(self):
        run = self.invoke(env=dict(self.env, GENERATOR3_TEST_FAIL="1"))
        self.assertEqual(run.returncode, 0, run.stdout)
        summary = json.loads((self.job/"merged/summary.json").read_text())
        self.assertEqual((summary["success"], summary["failed"]), (2, 2))
        resumed = self.invoke()
        self.assertEqual(resumed.returncode, 0, resumed.stdout)
        summary = json.loads((self.job/"merged/summary.json").read_text())
        self.assertEqual((summary["success"], summary["failed"]), (2, 2))
        self.assertEqual(len(list((self.root/"trace").glob("*.json"))), 4)

    def test_watchdog_stall_is_not_swallowed_as_a_monitor_warning(self):
        site = self.root/"site/sitecustomize.py"
        with site.open("a", encoding="utf-8") as stream:
            stream.write('''
if os.environ.get("GENERATOR3_TEST_WATCHDOG_TRIP") and "deta_generator3.watchdog" in sys.orig_argv:
    print("TEST-ONLY injected watchdog stall", flush=True)
    os._exit(2)
''')
        run = self.invoke(env=dict(self.env, GENERATOR3_TEST_WATCHDOG_TRIP="1"))
        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("Verified worker stall", run.stdout)
        # The controller intentionally leaves workers untouched. These test
        # workers finish their four analytic fixtures; wait before temp cleanup.
        runtime = json.loads((self.job/"parallel_workers.json").read_text(encoding="utf-8-sig"))
        waiting = []
        for row in runtime["workers"]:
            try:
                process = psutil.Process(row["pid"])
                if abs(process.create_time()-row["started_at_unix"]) <= 2:
                    waiting.append(process)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(waiting, timeout=30)
        self.assertFalse(alive)


if __name__ == "__main__":
    unittest.main()
