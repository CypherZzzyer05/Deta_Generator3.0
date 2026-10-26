"""CPU field/spectrum acquisition with Deta2-style jobs and shard scheduling."""
import argparse
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import psutil

from . import __version__
from .config import load_config
from .pipeline import (atomic_json, directory_lock, prepare_manifest, generate_dataset,
                       pack_dataset, checksum)
from .merge import merge_shards
from .native import locate_lumapi

ASSETS = Path(__file__).parent/"assets"


def initialize(directory, samples=1000, threads=4, lumapi=None, seed=20261023, keep_projects=False):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use an empty job directory; existing jobs are never overwritten")
    if (type(samples) is not int or samples < 1 or type(threads) is not int or threads < 1
            or type(seed) is not int or seed < 0):
        raise ValueError("Positive sample count and CPU thread count required")
    data = json.loads((ASSETS/"configs/paired.json").read_text(encoding="utf-8"))
    data.update(output_dir="data", sample_count=samples, seed=seed, lumapi_path=lumapi)
    data["simulation"].update(threads=threads, save_projects=keep_projects,
                             as2se3_nk_csv="materials/as2se3_joseph_400nm.csv")
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"materials").mkdir()
    for file in (ASSETS/"materials").iterdir():
        if file.is_file():
            shutil.copyfile(file, directory/"materials"/file.name)
    path = directory/"job.json"
    atomic_json(path, data)
    load_config(path)
    return dict(config=str(path), samples=samples, seed=seed,
                parameter_definition="p,dx,dy,h,l,alpha_1and3,alpha_2and4; dx/dy center-to-center",
                layer_order_bottom_to_top=["air", "glass350", "Au100", "As2Se3 tetramer", "air"],
                substrate_termination="finite", field_shape=[2, 9, 128, 128])


def configured(args):
    config = load_config(args.config)
    settings = config.simulation
    for key in ("threads", "processes"):
        value = getattr(args, key, None)
        if value is not None:
            settings = replace(settings, **{key: value})
    if args.keep_projects:
        settings = replace(settings, save_projects=True)
    return replace(config, simulation=settings,
                   lumapi_path=args.lumapi or os.environ.get("LUMERICAL_LUMAPI_PATH") or config.lumapi_path,
                   output_dir=Path(args.output).resolve() if args.output else config.output_dir)


def native_check(config, preflight=False):
    from .solver import PairedSimulator
    from .materials import validate_material_fit
    api = locate_lumapi(config.lumapi_path)
    report = dict(python=sys.executable, lumapi=str(api), solver_run_performed=False,
        available_ram_gib=psutil.virtual_memory().available/2**30, threads=config.simulation.threads,
        implementation_sha256=config.metadata()["implementation_sha256"],
        material_table_sha256=config.metadata()["material_table_sha256"])
    with PairedSimulator(config.simulation, str(api)) as simulator:
        geometry = config.sample_geometries()[0]
        simulator.build(geometry)
        report.update(fdtd_version=simulator.version, native_layout=simulator.layout_report,
                      native_layout_verified=True,
                      geometry_values=geometry.array().tolist(),
                      as2se3_fit=validate_material_fit(simulator.fdtd, config.simulation, simulator.optical_table),
                      material_identity=simulator.material_identity())
        if preflight:
            grids = {axis: np.asarray(simulator.fdtd.getresult("FDTD", axis)).reshape(-1)
                     for axis in ("x", "y", "z")}
            report["mesh"] = {a: dict(nodes=len(v), minimum_step_nm=float(np.diff(v).min()/1e-9),
                                      maximum_step_nm=float(np.diff(v).max()/1e-9)) for a, v in grids.items()}
            info = simulator.fdtd.runsystemcheck()
            recommended = info.get("Memory_Recommended_Bytes", info.get("Memory_Recommended"))
            if recommended is None or not np.isfinite(float(recommended)) or float(recommended) <= 0:
                raise ValueError("Native memory recommendation unavailable")
            report["recommended_ram_gib"] = float(recommended)/2**30
    path = config.output_dir.parent/("preflight.json" if preflight else "doctor.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, report)
    return {k: v for k, v in report.items() if k not in ("material_identity", "as2se3_fit")} | {
        "report": str(path), "as2se3_max_complex_index_error": report["as2se3_fit"]["max_complex_index_error"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=("init", "doctor", "preflight", "plan", "generate", "pack", "merge", "status", "parallel"))
    parser.add_argument("--config", type=Path, default=Path("job.json"))
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--preset", choices=("paired",), default="paired")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--lumapi")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-projects", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-ids", type=int, nargs="+")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry recorded failures; previous errors remain recorded")
    parser.add_argument("--pack-every", type=int, default=25)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", nargs="+", type=Path)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--parallel-action", choices=("start", "prepare", "run", "status", "merge"), default="start")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "parallel":
            if args.output or args.limit or args.processes or args.sample_ids or args.retry_failed or args.keep_projects:
                raise ValueError("Use init/custom serial jobs for these overrides; parallel uses the fixed paired preset")
            from .parallel import run_parallel
            samples = args.samples if args.samples is not None else 1000
            return run_parallel(args.job_dir or Path(f"jobs/paired_{samples}"), samples,
                args.workers, args.concurrency, args.threads, args.lumapi, args.parallel_action, args.pack_every,
                jobs=args.jobs, seed=args.seed)
        if any(v is not None for v in (args.jobs, args.workers, args.concurrency)):
            raise ValueError("--jobs/--workers/--concurrency require parallel")
        if args.command == "init":
            result = initialize(args.job_dir or "jobs/paired", args.samples if args.samples is not None else 1000,
                                args.threads if args.threads is not None else 4, args.lumapi,
                                args.seed if args.seed is not None else 20261023, args.keep_projects)
        else:
            config = configured(args)
            if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
                raise ValueError("Require shard_count>=1 and 0<=shard_index<shard_count")
            if args.command in ("doctor", "preflight"):
                result = native_check(config, args.command == "preflight")
            elif args.command == "merge":
                result = merge_shards(config, args.shards or [])
            elif args.command == "generate":
                if args.shard_count > 1 and args.output is None:
                    raise ValueError("Each shard requires a separate --output")
                ids = list(range(args.shard_index, config.sample_count, args.shard_count))
                if args.sample_ids is not None:
                    if not set(args.sample_ids) <= set(ids):
                        raise ValueError("Selected IDs are outside this shard")
                    ids = args.sample_ids
                result = generate_dataset(config, args.limit, args.fail_fast, ids, args.pack_every, args.retry_failed)
            else:
                with directory_lock(config.output_dir):
                    manifest = prepare_manifest(config)
                    result = pack_dataset(config, manifest) if args.command == "pack" else dict(
                        planned_samples=len(manifest["samples"]), config_fingerprint=manifest["config_fingerprint"],
                        output_dir=str(config.output_dir), parameter_names=manifest["parameter_names"],
                        counts={k: sum(e["state"] == k for e in manifest["samples"])
                                for k in ("pending", "running", "success", "failed")})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; committed samples are preserved. Repeat the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        logging.error("%s", error)
        return 1
