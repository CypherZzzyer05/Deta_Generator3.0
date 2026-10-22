"""Build the small standalone wheel and verify an independently installed copy."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

root = Path(__file__).resolve().parents[1]
artifacts = root/"artifacts"
artifacts.mkdir(exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir",
                ".", "--wheel-dir", str(artifacts)], cwd=root, check=True)
wheel = artifacts/"deta_generator3-3.0.0-py3-none-any.whl"
installed = root/"tmp/installed-wheel"
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--upgrade",
                "--target", str(installed), str(wheel)], check=True)
with zipfile.ZipFile(wheel) as archive:
    files = [n for n in archive.namelist() if n.startswith("deta_generator3/") and not n.endswith("/")]
    for name in files:
        if archive.read(name) != (root/"src"/name).read_bytes() or archive.read(name) != (installed/name).read_bytes():
            raise ValueError("Wheel/source/installed file differs: "+name)
check_dir = root/"tmp/isolated-check"
check_dir.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=str(installed), PYTHONIOENCODING="utf-8")
def call(*args):
    return subprocess.check_output([sys.executable, "-B", *args], env=env, cwd=check_dir,
                                   text=True, encoding="utf-8")
origin = call("-c", "import deta_generator3; print(deta_generator3.__file__)").strip()
if not Path(origin).is_relative_to(installed):
    raise ValueError("Verification imported a source checkout rather than the installed wheel")
if call("-m", "deta_generator3", "--version").strip() != "3.0.0":
    raise ValueError("Installed version differs")
with tempfile.TemporaryDirectory(prefix="wheel-job-", dir=check_dir) as job_name:
    job = Path(job_name)
    call("-m", "deta_generator3", "init", "--job-dir", str(job), "--samples", "4", "--threads", "1")
    call("-m", "deta_generator3", "plan", "--config", str(job/"job.json"))
    config = json.loads((job/"job.json").read_text())
    if (config["simulation"]["gold_thickness_nm"], config["simulation"]["glass_thickness_nm"],
            config["simulation"]["field_xy_span_nm"], config["simulation"]["image_size"]) != (100, 350, 1600, 128):
        raise ValueError("Installed preset physical contract differs")
report = dict(distribution="deta-generator3", version="3.0.0", wheel=wheel.name,
    sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), bytes=wheel.stat().st_size,
    package_files_verified=len(files), installed_origin_verified=True,
    installed_init_and_plan_verified=True, native_time_solve_in_this_check=False)
(artifacts/"checksums.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, indent=2))
