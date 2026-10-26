"""Parallel seed handling must preserve old plans and support fresh geometries."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deta_generator3 import cli, parallel
from deta_generator3.config import load_config, geometry_group_key


class ParallelSeedTests(unittest.TestCase):
    def test_new_task_uses_explicit_seed_and_preserves_legacy_default(self):
        with TemporaryDirectory() as root:
            self.assertEqual(parallel.resolve_seed(root), 20261023)
            self.assertEqual(parallel.resolve_seed(root, 20261026), 20261026)

    def test_existing_task_keeps_seed_and_rejects_reassignment_without_writing(self):
        with TemporaryDirectory() as root:
            path = Path(root)/"job.json"
            original = json.dumps(dict(seed=20261026, sample_count=1000)).encode()
            path.write_bytes(original)
            self.assertEqual(parallel.resolve_seed(root), 20261026)
            self.assertEqual(parallel.resolve_seed(root, 20261026), 20261026)
            with self.assertRaisesRegex(ValueError, "seed is fixed"):
                parallel.resolve_seed(root, 20261023)
            self.assertEqual(path.read_bytes(), original)

    def test_cli_passes_explicit_seed_and_omission_distinctly_to_parallel(self):
        with patch.object(parallel, "run_parallel", return_value=0) as run:
            self.assertEqual(cli.main(["parallel", "--seed", "20261026"]), 0)
            self.assertEqual(run.call_args.kwargs["seed"], 20261026)
            self.assertEqual(cli.main(["parallel"]), 0)
            self.assertIsNone(run.call_args.kwargs["seed"])

    def test_cli_init_without_seed_retains_original_default(self):
        with patch.object(cli, "initialize", return_value={}) as initialize:
            self.assertEqual(cli.main(["init"]), 0)
            self.assertEqual(initialize.call_args.args[4], 20261023)
            self.assertEqual(cli.main(["init", "--seed", "20261026"]), 0)
            self.assertEqual(initialize.call_args.args[4], 20261026)

    def test_requested_new_seed_produces_a_distinct_1000_geometry_plan(self):
        with TemporaryDirectory() as root:
            groups = []
            for seed in (20261023, 20261026):
                job = Path(root)/str(seed)
                cli.initialize(job, samples=1000, seed=seed)
                config = load_config(job/"job.json")
                groups.append({geometry_group_key(g.array()) for g in config.sample_geometries()})
            self.assertEqual(len(groups[0]), 1000)
            self.assertEqual(len(groups[1]), 1000)
            self.assertFalse(groups[0].intersection(groups[1]))

    def test_parallel_prepare_passes_seed_to_real_powershell_command(self):
        resources = dict(workers=4, concurrency=2, threads=4, cpu_inventory=dict(budget_cpus=64))
        with TemporaryDirectory() as root, \
                patch.object(parallel.sys, "platform", "win32"), \
                patch.object(parallel, "resolve_resources", return_value=resources), \
                patch.object(parallel.shutil, "which", return_value="powershell.exe"), \
                patch.object(parallel.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as invoke:
            self.assertEqual(parallel.run_parallel(root, action="prepare", seed=20261026), 0)
            command = invoke.call_args.args[0]
            self.assertEqual(command[command.index("-Seed")+1], "20261026")
            self.assertEqual(command[command.index("-Command")+1], "prepare")


if __name__ == "__main__":
    unittest.main()
