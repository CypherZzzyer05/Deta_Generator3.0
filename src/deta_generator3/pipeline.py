"""Immutable plans, paired sample integrity, resume and compact indexing."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import traceback
import numpy as np
from .config import Geometry, NAMES, UNITS, geometry_group_key
from .native import validate_spectrum
LOG = logging.getLogger(__name__)
SCHEMA = "as2se3_au_glass_paired_v1"

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 临时文件写完后再替换目标文件，使磁盘上的进度记录尽量保持完整。
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)

def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 使用文件句柄，避免 NumPy 自动追加 .npz 后缀，导致临时文件名与替换目标不一致。
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)

def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

@contextmanager
def generation_log(directory):
    handler = logging.FileHandler(directory / "generation.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    try:
        yield
    finally:
        LOG.removeHandler(handler)
        handler.close()

@contextmanager
def directory_lock(directory):
    """OS-held lock is released even if the process is killed in PyCharm."""
    # 使用操作系统文件锁，防止两个进程同时更新同一目录；进程退出会释放锁。
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".generation.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another generator is using {directory}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)

def split_assignments(count, seed):
    """Assign before simulation so later successes never move between splits."""
    # 在仿真前划分，避免按仿真成功顺序重新分组；100 个样本时得到 80/10/10。
    order = np.random.default_rng(seed + 1).permutation(count)
    n_val = max(1, round(count * .1)) if count >= 3 else 0
    n_test = max(1, round(count * .1)) if count >= 3 else 0
    n_train = count - n_val - n_test
    result = ["train"] * count
    for i in order[n_train:n_train+n_val]:
        result[int(i)] = "val"
    for i in order[n_train+n_val:]:
        result[int(i)] = "test"
    return result


def plan_checksum(manifest):
    plan = [{k: e[k] for k in ("id", "geometry_values", "split")} for e in manifest["samples"]]
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def prepare_manifest(config):
    metadata = config.metadata()
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = config.output_dir/"manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("schema") != SCHEMA or saved.get("config_fingerprint") != fingerprint:
            raise ValueError("Configuration/material/implementation changed; choose a new output directory")
        if (saved.get("sampling_plan_sha256") != plan_checksum(saved)
                or [e["id"] for e in saved["samples"]] != list(range(config.sample_count))):
            raise ValueError("Saved sampling/split plan changed")
        return saved
    if any(p.name != ".generation.lock" for p in config.output_dir.iterdir()):
        raise ValueError("Nonempty output directory without a manifest")
    geometries = config.sample_geometries()
    manifest = dict(schema=SCHEMA, schema_version=4, created_at=utc_now(), config=metadata,
        config_fingerprint=fingerprint, parameter_names=list(NAMES), parameter_units=list(UNITS),
        dx_dy_definition="center_to_center", positive_angle="CCW as viewed from +z",
        ambient="air above AND below finite stack", backend="real Lumerical FDTD",
        samples=[dict(id=i, geometry_values=g.array().tolist(), split=split, state="pending")
                 for i, (g, split) in enumerate(zip(geometries, split_assignments(len(geometries), config.seed)))])
    manifest["sampling_plan_sha256"] = plan_checksum(manifest)
    atomic_json(path, manifest)
    return manifest


def validate_pair(result, config):
    s = config.simulation
    validate_spectrum(result["wavelength_nm"], result["reflectance"], result["transmittance"], s, int(result["status"]))
    shape = (len(s.field_z_nm), len(s.field_wavelengths_nm), s.image_size, s.image_size)
    for key in ("intensity", "hotspot_mask", "hotspot_labels"):
        if np.asarray(result[key]).shape != shape:
            raise ValueError(f"Wrong paired field shape: {key}")
    if (not np.isfinite(result["intensity"]).all() or np.any(result["intensity"] < 0)
            or not np.isin(result["hotspot_mask"], (0, 1)).all()):
        raise ValueError("Invalid absolute intensity or hotspot mask")
    flux = result["poynting_incident_normalized"]
    if np.shape(flux) != (*shape, 3) or not np.isfinite(flux).all():
        raise ValueError("Invalid paired Poynting field")
    for key, expected in (("field_wavelength_nm", s.field_wavelengths_nm), ("field_z_nm", s.field_z_nm)):
        if not np.array_equal(result[key], expected):
            raise ValueError(f"Field observation axis differs: {key}")
    pixels = ((np.arange(s.image_size)+.5)/s.image_size-.5)*s.field_xy_span_nm
    for key in ("pixel_x_nm", "pixel_y_nm"):
        if np.shape(result[key]) != pixels.shape or not np.allclose(result[key], pixels, rtol=1e-8, atol=1e-6):
            raise ValueError("Fixed laboratory pixel coordinates differ")
    for plane in range(len(s.field_z_nm)):
        x, y = [np.asarray(result[f"native_{axis}_nm_{plane}"]) for axis in ("x", "y")]
        if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in (x, y)):
            raise ValueError("Native field coordinates must be finite and ascending")
        for key in ("E", "H"):
            field = result[f"native_{key}_{plane}"]
            if (np.shape(field) != (len(y), len(x), len(s.field_wavelengths_nm), 3)
                    or not np.iscomplexobj(field) or not np.isfinite(field).all()):
                raise ValueError("Invalid native co-located complex field")


def load_sample(path, entry, config, fingerprint):
    if checksum(path) != entry.get("sha256"):
        raise ValueError(f"Sample checksum differs: {path}; original preserved")
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    if (str(result["config_fingerprint"].item()) != fingerprint
            or not np.array_equal(result["geometry_values"], entry["geometry_values"])
            or not np.allclose(result["geometry_01"], config.normalize(entry["geometry_values"]), atol=1e-7)
            or not bool(result["native_layout_verified"])):
        raise ValueError("Sample identity/geometry/layout differs")
    validate_pair(result, config)
    return result


def pack_dataset(config, manifest):
    entries = [e for e in manifest["samples"] if e["state"] == "success"]
    spectra, transmission, versions = [], [], set()
    for entry in entries:
        sample = load_sample(config.output_dir/"samples"/f"sample_{entry['id']:06d}.npz",
                             entry, config, manifest["config_fingerprint"])
        spectra.append(sample["reflectance"].astype(np.float32))
        transmission.append(sample["transmittance"].astype(np.float32))
        versions.add(str(sample["fdtd_version"].item()))
    if len(versions) > 1 or (versions and versions != {manifest.get("fdtd_version")}):
        raise ValueError("Cannot pack mixed solver versions")
    count = len(entries)
    values = np.array([e["geometry_values"] for e in entries], dtype=float).reshape(count, 7)
    splits = np.array([e["split"] for e in entries], dtype="U5")
    training = json.loads(json.dumps(manifest))
    for entry in training["samples"]:
        if entry["state"] == "success":
            entry["state"] = "complete"
    atomic_json(config.output_dir/"training_manifest.json", training)
    arrays = dict(spectra=np.asarray(spectra, dtype=np.float32).reshape(count, 1000),
        transmittance=np.asarray(transmission, dtype=np.float32).reshape(count, 1000),
        geometry_values=values, geometry_01=config.normalize(values).astype(np.float32),
        lower_values=config.lower, upper_values=config.upper,
        parameter_names=np.asarray(NAMES), parameter_units=np.asarray(UNITS),
        wavelength_nm=config.simulation.wavelength_grid(), sample_ids=np.array([e["id"] for e in entries], dtype=np.int64),
        split_names=splits, status=np.full(count, 2, dtype=np.int64),
        field_sample_files=np.array([f"samples/sample_{e['id']:06d}.npz" for e in entries], dtype="U64"),
        metadata_json=json.dumps(manifest, ensure_ascii=False))
    for split in ("train", "val", "test"):
        arrays[split+"_indices"] = np.flatnonzero(splits == split)
    atomic_npz(config.output_dir/"dataset.npz", **arrays)
    summary = dict(planned=config.sample_count, success=count,
        failed=sum(e["state"] == "failed" for e in manifest["samples"]),
        pending=sum(e["state"] not in ("success", "failed") for e in manifest["samples"]),
        splits={k: int(np.sum(splits == k)) for k in ("train", "val", "test")},
        spectra_shape=[count, 1000], field_shape=[2, 9, 128, 128],
        dataset=str(config.output_dir/"dataset.npz"), training_manifest=str(config.output_dir/"training_manifest.json"),
        quality_level=config.quality_level, field_storage="per-sample NPZ; no duplicated stacked field cache")
    summary["dataset_ready"] = all(summary["splits"][k] > 0 for k in ("train", "val", "test"))
    summary["all_planned_samples_passed"] = count == config.sample_count
    atomic_json(config.output_dir/"summary.json", summary)
    return summary


def generate_dataset(config, limit=None, fail_fast=False, sample_ids=None, pack_every=25, retry_failed=False):
    from .solver import PairedSimulator
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if type(pack_every) is not int or pack_every < 1:
        raise ValueError("pack_every must be positive")
    selected = set(range(config.sample_count) if sample_ids is None else sample_ids)
    if not selected <= set(range(config.sample_count)):
        raise ValueError("Sample IDs are outside the original plan")
    with directory_lock(config.output_dir):
        manifest = prepare_manifest(config)
        with generation_log(config.output_dir):
            todo = []
            for entry in manifest["samples"]:
                if entry["id"] not in selected:
                    continue
                if entry["state"] == "success":
                    load_sample(config.output_dir/"samples"/f"sample_{entry['id']:06d}.npz", entry, config, manifest["config_fingerprint"])
                elif entry["state"] != "failed" or retry_failed:
                    todo.append(entry)
            if limit is not None:
                todo = todo[:limit]
            if not todo:
                return pack_dataset(config, manifest)
            (config.output_dir/"samples").mkdir(exist_ok=True)
            try:
                with PairedSimulator(config.simulation, config.lumapi_path) as simulator:
                    if manifest.get("fdtd_version") not in (None, simulator.version):
                        raise ValueError("Solver version changed; use a new output directory")
                    manifest["fdtd_version"] = simulator.version
                    for processed, entry in enumerate(todo, 1):
                        i = entry["id"]
                        if entry["state"] == "failed":
                            entry.setdefault("previous_failures", []).append({k: entry[k] for k in ("error", "finished_at", "project_subdir") if k in entry})
                        retries = len(entry.get("previous_failures", []))
                        entry["project_subdir"] = f"retry_{retries:03d}" if retries else ""
                        entry.update(state="running", started_at=utc_now())
                        atomic_json(config.output_dir/"manifest.json", manifest)
                        LOG.info("Sample %d/%d (%s): %s", i+1, config.sample_count, entry["split"], entry["geometry_values"])
                        try:
                            path = config.output_dir/"samples"/f"sample_{i:06d}.npz"
                            if path.exists():
                                raise ValueError("Uncommitted sample exists; preserve and inspect it before resuming")
                            project = config.output_dir/"projects"/entry["project_subdir"]/f"sample_{i:06d}.fsp"
                            result = simulator.run(Geometry(*entry["geometry_values"]), project, starting_attempt=3)
                            validate_pair(result, config)
                            identity = json.loads(result["material_identity_json"])
                            from .merge import same_material_identity
                            if ("material_identity" in manifest
                                    and not same_material_identity(manifest["material_identity"], identity)):
                                raise ValueError("Actual material fits changed; result rejected")
                            manifest["material_identity"] = identity
                            # Keep native E/H and model inputs, omit duplicated reconstructed E/H and SI flux.
                            stored = {k: v for k, v in result.items() if k not in ("electric", "magnetic", "poynting_w_m2")}
                            atomic_npz(path, **stored, geometry_values=np.array(entry["geometry_values"]),
                                geometry_01=config.normalize(entry["geometry_values"]).astype(np.float32), sample_id=i,
                                config_fingerprint=np.asarray(manifest["config_fingerprint"]))
                            entry.update(state="success", sha256=checksum(path), status=2,
                                elapsed_seconds=float(result["elapsed_seconds"]), finished_at=utc_now())
                            entry.pop("error", None)
                            if not config.simulation.save_projects:
                                try:
                                    project.unlink(missing_ok=True)
                                except OSError:
                                    LOG.warning("Successful sample committed; scratch project could not be removed: %s", project)
                        except Exception as error:
                            entry.update(state="failed", error=str(error), finished_at=utc_now())
                            with (config.output_dir/"errors.jsonl").open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(dict(id=i, time=utc_now(), error=str(error), traceback=traceback.format_exc()))+"\n")
                            LOG.exception("Sample %d rejected; excluded from training", i)
                            if fail_fast:
                                raise
                        finally:
                            atomic_json(config.output_dir/"manifest.json", manifest)
                            if processed % pack_every == 0:
                                pack_dataset(config, manifest)
            finally:
                summary = pack_dataset(config, manifest)
            return summary
