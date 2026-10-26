"""Stopped tasks can merge started shards while later shards remain uncreated."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deta_generator3.config import load_config
from deta_generator3.pipeline import generate_dataset
from test_generator import TestOnlySimulator

ROOT = Path(__file__).resolve().parents[1]
SHELL = shutil.which("powershell") or shutil.which("pwsh")


@unittest.skipUnless(sys.platform == "win32" and SHELL, "Windows controller integration")
class PartialParallelMergeTests(unittest.TestCase):
    def test_unstarted_shards_do_not_block_partial_export(self):
        with TemporaryDirectory() as directory:
            job = Path(directory)/"partial task"
            env = dict(os.environ, PYTHONPATH=str(ROOT/"src"))
            base = [sys.executable, "-m", "deta_generator3", "parallel", "--job-dir", str(job),
                    "--samples", "4", "--workers", "4", "--jobs", "2", "--threads", "1"]
            prepared = subprocess.run(base+["--parallel-action", "prepare"], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(prepared.returncode, 0, prepared.stdout)
            config = replace(load_config(job/"job.json"), output_dir=job/"shard0")
            with patch("deta_generator3.solver.PairedSimulator", TestOnlySimulator):
                generate_dataset(config, sample_ids=[0])
            merged = subprocess.run(base+["--parallel-action", "merge"], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(merged.returncode, 0, merged.stdout)
            summary = json.loads((job/"merged/summary.json").read_text())
            self.assertEqual((summary["success"], summary["pending"]), (1, 3))
            self.assertFalse((job/"shard1").exists())
