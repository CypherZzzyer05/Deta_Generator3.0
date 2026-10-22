"""Test-only analytic fields exercise data flow; they are not FDTD results."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from deta_generator3.cli import initialize
from deta_generator3.config import Geometry, NAMES, load_config
from deta_generator3.fixed_fov import render_fixed_observations
from deta_generator3.merge import merge_shards
from deta_generator3.pipeline import (generate_dataset, directory_lock, prepare_manifest,
                                      checksum, load_sample, pack_dataset, atomic_json)


class TestOnlySimulator:
    calls = 0
    fail_height = None

    def __init__(self, settings, lumapi_path=None):
        self.settings = settings
        self.version = "TEST-ONLY-NOT-FDTD"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run(self, geometry, project, starting_attempt=3):
        type(self).calls += 1
        if type(self).fail_height == geometry.h:
            raise ValueError("TEST-ONLY rejected power spectrum")
        s = self.settings
        data = dict(status=2, fdtd_version=self.version, elapsed_seconds=.1,
            native_layout_verified=True,
            wavelength_nm=s.wavelength_grid(), reflectance=np.full(1000, .5),
            transmittance=np.full(1000, .1), absorptance=np.full(1000, .4),
            field_wavelength_nm=np.asarray(s.field_wavelengths_nm), field_z_nm=np.asarray(s.field_z_nm),
            material_identity_json=json.dumps({"fixture": {"material": "TEST-ONLY",
                "fitted_index_ri": [[1., 0.], [1., 0.]]}}))
        for plane in range(2):
            for axis in ("x", "y"):
                data[f"native_{axis}_nm_{plane}"] = np.linspace(-geometry.p/2, geometry.p/2, 5)
            data[f"native_E_{plane}"] = np.ones((5, 5, 9, 3), np.complex64)
            data[f"native_H_{plane}"] = np.full((5, 5, 9, 3), .001, np.complex64)
        observation, _ = render_fixed_observations(data, s)
        data.update(observation)
        return data


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        initialize(self.root/"job", samples=4, threads=1)
        self.config = load_config(self.root/"job/job.json")
        TestOnlySimulator.calls = 0
        TestOnlySimulator.fail_height = None
        self.patch = patch("deta_generator3.solver.PairedSimulator", TestOnlySimulator)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_fixed_center_distances_angles_and_overlap_rejection(self):
        g = Geometry(800, 350, 380, 250, 180, 30, -20)
        np.testing.assert_array_equal(g.centers(), [[175, 190], [-175, 190], [-175, -190], [175, -190]])
        self.assertEqual(g.angles(), (30, -20, 30, -20))
        with self.assertRaises(ValueError):
            Geometry(800, 100, 100, 250, 180, 0, 0)
        with self.assertRaises(ValueError):
            Geometry(800, 350, 380, 250, 180, 45, 0)

    def test_resume_and_model_inputs_remain_identical_without_field_pack_duplication(self):
        first = generate_dataset(self.config, limit=1)
        self.assertEqual(first["success"], 1)
        moved_runtime = replace(self.config, lumapi_path="X:/other/lumapi.py",
            simulation=replace(self.config.simulation, threads=16))
        result = generate_dataset(moved_runtime)
        self.assertEqual(result["success"], 4)
        self.assertEqual(TestOnlySimulator.calls, 4)
        before = {p: checksum(p) for p in (self.config.output_dir/"samples").glob("*.npz")}
        generate_dataset(moved_runtime)
        self.assertEqual(TestOnlySimulator.calls, 4)
        self.assertEqual(before, {p: checksum(p) for p in before})
        training = json.loads((self.config.output_dir/"training_manifest.json").read_text())
        self.assertTrue(all(e["state"] == "complete" for e in training["samples"]))
        self.assertEqual(training["parameter_names"], list(NAMES))
        with np.load(self.config.output_dir/"dataset.npz", allow_pickle=False) as packed:
            self.assertEqual(packed["spectra"].shape, (4, 1000))
            self.assertNotIn("intensity", packed.files)
            self.assertEqual(len(packed["field_sample_files"]), 4)
        with np.load(next((self.config.output_dir/"samples").glob("*.npz"))) as data:
            np.testing.assert_allclose(data["intensity"], 3, rtol=1e-7)
            self.assertEqual(data["intensity"].shape, (2, 9, 128, 128))
            self.assertIn("native_E_0", data.files)
            self.assertNotIn("electric", data.files)

    def test_failure_is_preserved_and_skipped_until_explicit_retry(self):
        TestOnlySimulator.fail_height = self.config.sample_geometries()[0].h
        result = generate_dataset(self.config)
        self.assertEqual((result["success"], result["failed"]), (3, 1))
        calls = TestOnlySimulator.calls
        generate_dataset(self.config)
        self.assertEqual(TestOnlySimulator.calls, calls)
        TestOnlySimulator.fail_height = None
        result = generate_dataset(self.config, retry_failed=True)
        self.assertEqual(result["success"], 4)
        manifest = json.loads((self.config.output_dir/"manifest.json").read_text())
        self.assertIn("TEST-ONLY", manifest["samples"][0]["previous_failures"][0]["error"])
        self.assertEqual(manifest["samples"][0]["project_subdir"], "retry_001")

    def test_corrupted_success_is_rejected_and_not_silently_overwritten(self):
        generate_dataset(self.config, limit=1)
        path = next((self.config.output_dir/"samples").glob("*.npz"))
        path.write_bytes(b"TEST corrupted sample")
        with self.assertRaisesRegex(ValueError, "checksum"):
            generate_dataset(self.config)
        self.assertEqual(path.read_bytes(), b"TEST corrupted sample")

    def test_two_shards_merge_failures_and_successes_without_recomputing_or_copying_fields(self):
        shards = [replace(self.config, output_dir=self.root/f"shard{i}") for i in range(2)]
        TestOnlySimulator.fail_height = self.config.sample_geometries()[0].h
        for i, shard in enumerate(shards):
            generate_dataset(shard, sample_ids=range(i, 4, 2))
        merged = replace(self.config, output_dir=self.root/"merged")
        result = merge_shards(merged, [s.output_dir for s in shards])
        self.assertEqual((result["success"], result["failed"]), (3, 1))
        manifest = json.loads((merged.output_dir/"manifest.json").read_text())
        self.assertEqual(manifest["samples"][0]["state"], "failed")
        source = shards[1].output_dir/"samples/sample_000001.npz"
        destination = merged.output_dir/"samples/sample_000001.npz"
        self.assertTrue(source.samefile(destination))
        self.assertEqual(TestOnlySimulator.calls, 4)
        TestOnlySimulator.fail_height = None
        generate_dataset(shards[0], sample_ids=[0], retry_failed=True)
        repaired = merge_shards(merged, [s.output_dir for s in shards])
        self.assertEqual(repaired["success"], 4)
        manifest = json.loads((merged.output_dir/"manifest.json").read_text())
        self.assertTrue(manifest["samples"][0]["previous_failures"])

    def test_changed_plan_material_or_solver_provenance_is_rejected(self):
        generate_dataset(self.config, limit=1)
        changed = replace(self.config, simulation=replace(self.config.simulation, mesh_nm=8))
        with self.assertRaisesRegex(ValueError, "Configuration"):
            generate_dataset(changed)
        path = self.config.output_dir/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest["samples"][0]["geometry_values"][3] += 1
        atomic_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "plan changed"):
            generate_dataset(self.config)

    def test_field_axis_mismatch_cannot_enter_dataset(self):
        original = TestOnlySimulator.run
        def wrong_plane(sim, *args, **kwargs):
            result = original(sim, *args, **kwargs)
            result["field_z_nm"] = np.array([100, 350])
            return result
        with patch.object(TestOnlySimulator, "run", wrong_plane):
            result = generate_dataset(self.config, limit=1)
        self.assertEqual((result["success"], result["failed"]), (0, 1))
        self.assertFalse(list((self.config.output_dir/"samples").glob("*.npz")))

    def test_concurrent_writer_is_rejected(self):
        with directory_lock(self.config.output_dir):
            with self.assertRaisesRegex(RuntimeError, "Another generator"):
                generate_dataset(self.config)


if __name__ == "__main__":
    unittest.main()
