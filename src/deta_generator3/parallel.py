"""PyCharm-friendly entry to the packaged Windows shard controller."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
from .resources import cpu_inventory


def resolve_resources(job_dir, samples=100, workers=None, concurrency=None, threads=None, jobs=None):
    if samples < 1:
        raise ValueError('samples must be positive')
    if jobs is not None and concurrency is not None and jobs != concurrency:
        raise ValueError('--jobs and --concurrency specify the same active-job limit; use one value')
    requested = jobs if jobs is not None else concurrency
    if requested is not None and not 1 <= requested <= min(samples,64):
        raise ValueError('Require 1<=jobs<=min(samples,64)')
    layout_path = Path(job_dir)/'parallel_plan.json'
    previous = json.loads(layout_path.read_text(encoding='utf-8-sig')) if layout_path.is_file() else None
    if previous and previous['samples'] != samples:
        raise ValueError('Sample count is fixed for this task; use a new --job-dir')
    active = requested if requested is not None else min(samples,4)
    if workers is None:
        workers = int(previous['workers']) if previous else min(samples,active*4,64)
    if not 1 <= workers <= min(samples,64):
        raise ValueError('Require 1<=workers<=min(samples,64)')
    if previous and workers != previous['workers']:
        raise ValueError('Existing shard count is fixed; use a new --job-dir to change it')
    if requested is None:
        active = min(active,workers)
    if active > workers:
        raise ValueError(f'Task has {workers} shards but {active} jobs were requested; use a new task with more shards')
    inventory = cpu_inventory()
    budget = inventory['budget_cpus']
    threads = max(1,budget//active) if threads is None else threads
    if not 1 <= threads <= 1024 or active*threads > budget:
        raise ValueError(f'{active} jobs x {threads} threads exceeds the usable CPU budget {budget}; omit --threads for automatic allocation')
    return dict(workers=workers,concurrency=active,threads=threads,cpu_inventory=inventory)


def run_parallel(job_dir, samples=100, workers=None, concurrency=None, threads=None,
                 lumapi=None, action='start', pack_every=25, jobs=None):
    if sys.platform != 'win32':
        raise ValueError('The automatic parallel controller is currently for Windows')
    resources = resolve_resources(job_dir,samples,workers,concurrency,threads,jobs)
    workers,concurrency,threads = (resources[key] for key in ('workers','concurrency','threads'))
    if not 1 <= pack_every <= 10000:
        raise ValueError('Pack interval must be between 1 and 10000')
    shell = shutil.which('powershell') or shutil.which('pwsh')
    if shell is None:
        raise FileNotFoundError('Windows PowerShell is required for the parallel controller')
    script = Path(__file__).parent/'assets/scripts/run-parallel.ps1'
    if not script.is_file():
        raise FileNotFoundError('Packaged controller missing; reinstall deta-generator3 3.0.0 or newer')
    job_dir = Path(job_dir).resolve()
    api = lumapi or os.environ.get('LUMERICAL_LUMAPI_PATH')
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    base = [shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
            '-Python', sys.executable, '-JobDir', str(job_dir), '-Samples', str(samples),
            '-Workers', str(workers), '-Concurrency', str(concurrency), '-Threads', str(threads),
            '-PackEvery', str(pack_every)]
    if api:
        base += ['-Lumapi', api]

    def invoke(command):
        # Inherit stdout/stderr so progress appears in PyCharm's Run window.
        return subprocess.run(base+['-Command', command], env=env).returncode

    print(f'PARALLEL CONFIG ({action}): capacity={concurrency} independent samples x {threads} threads each; {workers} queued shards; {samples} samples total.', flush=True)
    print('CPU inventory: '+json.dumps(resources['cpu_inventory']),flush=True)
    if action not in ('start', 'run'):
        return invoke(action)
    if action == 'start':
        code = invoke('prepare')
        if code:
            return code
    print('Checking the first native layout and mesh; no FDTD engine run in preflight.', flush=True)
    preflight = [sys.executable, '-m', 'deta_generator3', 'preflight', '--config', str(job_dir/'job.json'),
                 '--threads', str(threads)]
    if api:
        preflight += ['--lumapi', api]
    code = subprocess.run(preflight, env=env).returncode
    if code:
        return code
    import psutil
    preflight_report = json.loads((job_dir/'preflight.json').read_text(encoding='utf-8'))
    needed = preflight_report['recommended_ram_gib'] * concurrency
    available = psutil.virtual_memory().available/2**30
    if needed > available * .85:
        raise ValueError(f'Estimated solver memory {needed:.2f} GiB for {concurrency} jobs exceeds '
                         f'the available {available:.2f} GiB budget; lower --jobs')
    print('Starting CPU dataset generation; successful shards are merged automatically.', flush=True)
    return invoke('run')
