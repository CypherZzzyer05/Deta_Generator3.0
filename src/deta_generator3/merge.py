"""Verify shards, preserve failures and link immutable native sample files."""
from contextlib import ExitStack
import errno
import json
import os
from pathlib import Path
import shutil
import numpy as np

from .pipeline import (directory_lock, prepare_manifest, plan_checksum, checksum,
                       load_sample, atomic_json, pack_dataset)


def same_material_identity(left, right):
    return (set(left) == set(right) and all(
        left[name]["material"] == right[name]["material"]
        and np.shape(left[name]["fitted_index_ri"]) == np.shape(right[name]["fitted_index_ri"])
        and np.allclose(left[name]["fitted_index_ri"], right[name]["fitted_index_ri"], rtol=1e-7, atol=1e-7)
        for name in left))


def merge_shards(config, directories):
    directories = [Path(p).resolve() for p in directories]
    if not directories or len(set(directories)) != len(directories) or config.output_dir.resolve() in directories:
        raise ValueError("Supply distinct shard directories and a separate merge destination")
    with ExitStack() as stack:
        for folder in sorted([*directories, config.output_dir.resolve()]):
            stack.enter_context(directory_lock(folder))
        target = prepare_manifest(config)
        incoming = {}
        versions, identities = set(), []
        for folder in directories:
            manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("config_fingerprint") != target["config_fingerprint"]
                    or plan_checksum(manifest) != target["sampling_plan_sha256"]
                    or manifest.get("sampling_plan_sha256") != target["sampling_plan_sha256"]):
                raise ValueError("Shard physics/implementation or geometry/split plan differs")
            if "fdtd_version" in manifest:
                versions.add(manifest["fdtd_version"])
            if "material_identity" in manifest:
                identities.append(manifest["material_identity"])
            for entry in manifest["samples"]:
                if entry["state"] not in ("success", "failed"):
                    continue
                expected = target["samples"][entry["id"]]
                if any(entry[k] != expected[k] for k in ("geometry_values", "split")):
                    raise ValueError("Shard geometry or split changed")
                prior = incoming.get(entry["id"])
                if prior and any(prior[1].get(k) != entry.get(k) for k in ("state", "sha256", "error")):
                    raise ValueError("Conflicting duplicate outcome for sample "+str(entry["id"]))
                recovered = (expected["state"] == "failed" and entry["state"] == "success"
                    and any(all(old.get(k) == expected.get(k) for k in ("error", "finished_at"))
                            for old in entry.get("previous_failures", [])))
                if expected["state"] in ("success", "failed") and not recovered and any(
                        expected.get(k) != entry.get(k) for k in ("state", "sha256", "error")):
                    raise ValueError("Existing merge outcome differs; original preserved")
                if entry["state"] == "success":
                    load_sample(folder/"samples"/f"sample_{entry['id']:06d}.npz", entry, config, target["config_fingerprint"])
                incoming[entry["id"]] = (folder, entry)
        if "fdtd_version" in target:
            versions.add(target["fdtd_version"])
        if len(versions) > 1:
            raise ValueError("Mixed native solver versions")
        if "material_identity" in target:
            identities.append(target["material_identity"])
        if identities and any(not same_material_identity(identities[0], i) for i in identities[1:]):
            raise ValueError("Actual fitted material indices differ")
        if versions:
            target["fdtd_version"] = versions.pop()
        if identities:
            target["material_identity"] = identities[0]
        (config.output_dir/"samples").mkdir(exist_ok=True)
        for i, (folder, entry) in sorted(incoming.items()):
            if entry["state"] == "success":
                source = folder/"samples"/f"sample_{i:06d}.npz"
                destination = config.output_dir/"samples"/source.name
                if destination.exists():
                    if checksum(destination) != entry["sha256"]:
                        raise ValueError("Merge destination conflicts with source sample")
                else:
                    try:
                        os.link(source, destination)
                    except OSError as error:
                        if error.errno != errno.EXDEV and getattr(error, "winerror", None) != 17:
                            raise
                        temporary = destination.with_suffix(".npz.tmp")
                        shutil.copyfile(source, temporary)
                        if checksum(temporary) != entry["sha256"]:
                            raise ValueError("Cross-volume copy checksum differs")
                        os.replace(temporary, destination)
            target["samples"][i] = entry
        target["merged_shards"] = [str(p) for p in directories]
        atomic_json(config.output_dir/"manifest.json", target)
        return pack_dataset(config, target)
