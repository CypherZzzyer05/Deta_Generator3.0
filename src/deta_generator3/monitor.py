"""Read-only observations of shard workers and their native FDTD engines.

No simulator is imported, started or stopped here. A native-engine count is an
observed lower bound: process permissions and short-lived children can hide jobs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import psutil


LOG_TAIL_BYTES = 32768
STATES = ('success', 'running', 'failed', 'pending')
PROGRESS_RE = re.compile(r'(\d+(?:\.\d+)?)\s*%\s*complete', re.IGNORECASE)
SHUTOFF_RE = re.compile(r'Auto\s+Shutoff\s*:\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)', re.IGNORECASE)


def _open_shared_read(path, *, binary=False, encoding='utf-8'):
    """Read with Windows share-read/write/delete access.

    Python's ordinary Windows file open does not share deletion, which can
    prevent native writers from renaming/deleting their files. Windows
    MoveFileEx still cannot replace an *open destination*, so callers must
    close promptly and the atomic writer must tolerate a brief read overlap.
    The CRT descriptor owns the native handle after open_osfhandle succeeds;
    fdopen closes both.
    """
    path = Path(path)
    if os.name != 'nt':
        return path.open('rb') if binary else path.open('r', encoding=encoding)

    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create_file = kernel.CreateFileW
    create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    # GENERIC_READ; FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    # OPEN_EXISTING; FILE_ATTRIBUTE_NORMAL. No write/delete access is requested.
    native_handle = create_file(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if native_handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(native_handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close_handle(native_handle)
        raise
    try:
        return os.fdopen(fd, 'rb') if binary else os.fdopen(fd, 'r', encoding=encoding)
    except BaseException:
        # fd now owns the native handle, including failure to create a wrapper.
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _read_json(path):
    # Parsing a large manifest must not extend the native file-handle lifetime.
    with _open_shared_read(path, binary=True) as handle:
        contents = handle.read()
    value = json.loads(contents.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path.name}')
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _process_name(name):
    # PowerShell ProcessName omits .exe, whereas psutil.name() includes it.
    name = str(name).replace('\\', '/').rsplit('/', 1)[-1].lower()
    return name[:-4] if name.endswith('.exe') else name


def read_progress(project_dir, sample_id, now):
    """Parse only a bounded log tail; never return raw solver-log contents."""
    result = {'sample_id': sample_id, 'percent_complete': None,
              'auto_shutoff': None, 'log_age_seconds': None, 'log_file': None,
              'log_read_error': None}
    for name in (f'sample_{sample_id:06d}_p0.log', f'sample_{sample_id:06d}.log'):
        path = project_dir / name
        try:
            with _open_shared_read(path, binary=True) as handle:
                size = handle.seek(0, os.SEEK_END)
                start = max(0, size - LOG_TAIL_BYTES)
                handle.seek(start)
                tail = handle.read(LOG_TAIL_BYTES)
                modified = os.fstat(handle.fileno()).st_mtime
        except FileNotFoundError:
            continue
        except OSError as error:
            # One inaccessible/deleted log must not hide the other shards or
            # discard process observations. Do not expose raw exception text.
            result.update(log_file=name, log_read_error=type(error).__name__)
            return result
        text = tail.decode('utf-8', errors='replace')
        progress = PROGRESS_RE.findall(text)
        shutoff = SHUTOFF_RE.findall(text)
        if progress and 0 <= float(progress[-1]) <= 100:
            result['percent_complete'] = float(progress[-1])
        if shutoff and _finite_number(float(shutoff[-1])):
            result['auto_shutoff'] = float(shutoff[-1])
        result.update(log_file=name, log_age_seconds=max(0., now-modified))
        return result
    return result


def _metrics(process, issues, process_api):
    result = {'rss_bytes': None, 'cpu_seconds': None, 'created_at_unix': None}
    for field, method in (('rss_bytes', 'memory_info'), ('cpu_seconds', 'cpu_times'),
                          ('created_at_unix', 'create_time')):
        try:
            value = getattr(process, method)()
            if field == 'rss_bytes':
                value = value.rss
            elif field == 'cpu_seconds':
                value = value.user + value.system
            result[field] = value
        except (process_api.NoSuchProcess, process_api.AccessDenied, process_api.ZombieProcess) as error:
            issues.append(f'{method}: {type(error).__name__}')
    return result


def _observe_worker(record, engines, process_api):
    index = record.get('index')
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError('Worker index must be a nonnegative integer')
    pid = _positive_int(record.get('pid'), 'Worker PID')
    expected_name = record.get('process_name')
    if not isinstance(expected_name, str) or not expected_name.strip():
        raise ValueError('Worker process_name is required')
    started = record.get('started_at_unix')
    if started is not None and not _finite_number(started):
        raise ValueError('Worker started_at_unix must be a finite timestamp')
    issues = []
    row = {'index': index, 'pid': pid, 'process_name': expected_name,
           'started_at_unix': started, 'alive': False, 'identity_verified': False,
           'issues': issues, 'engine_pids': []}
    try:
        process = process_api.Process(pid)
        if not process.is_running() or process.status() == process_api.STATUS_ZOMBIE:
            return row
        if _process_name(process.name()) != _process_name(expected_name):
            row['identity_mismatch'] = 'process_name'
            return row
        if started is not None:
            created = process.create_time()
            if abs(created-started) > 2.:
                row['identity_mismatch'] = 'creation_time'
                return row
            row['identity_verified'] = True
        else:
            issues.append('Creation time absent in runtime record; identity checked by name only')
        row['alive'] = True
        row.update(_metrics(process, issues, process_api))
        try:
            children = process.children(recursive=True)
        except (process_api.NoSuchProcess, process_api.AccessDenied, process_api.ZombieProcess) as error:
            issues.append(f'children: {type(error).__name__}')
            children = []
        for child in children:
            try:
                if (not child.is_running() or child.status() == process_api.STATUS_ZOMBIE or
                        not _process_name(child.name()).startswith('fdtd-engine')):
                    continue
                if child.pid not in engines:
                    engine_issues = []
                    engines[child.pid] = {'pid': child.pid, 'issues': engine_issues,
                                         **_metrics(child, engine_issues, process_api)}
                row['engine_pids'].append(child.pid)
            except (process_api.NoSuchProcess, process_api.AccessDenied, process_api.ZombieProcess) as error:
                issues.append(f'child observation: {type(error).__name__}')
    except process_api.AccessDenied:
        row['alive'] = None
        issues.append('Worker identity observation: AccessDenied')
    except (process_api.NoSuchProcess, process_api.ZombieProcess):
        row['alive'] = False
    return row


def collect_status(job_dir, expected_jobs, now=None, process_api=psutil):
    """Return a JSON-serializable status without changing any task files."""
    expected_jobs = _positive_int(expected_jobs, 'Expected jobs')
    now = time.time() if now is None else now
    job_dir = Path(job_dir).resolve()
    config = _read_json(job_dir/'job.json')
    layout = _read_json(job_dir/'parallel_plan.json')
    samples = _positive_int(config.get('sample_count'), 'Sample count')
    shards = _positive_int(layout.get('workers'), 'Shard count')
    if layout.get('samples') != samples:
        raise ValueError('job.json and parallel_plan.json have different sample counts')
    if expected_jobs > shards:
        raise ValueError('Expected jobs cannot exceed the fixed shard count')
    identity = hashlib.sha256(json.dumps({'config': config, 'samples': samples, 'shards': shards},
                                        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    previous = {}
    warnings = []
    previous_path = job_dir/'parallel_status.json'
    if previous_path.exists():
        try:
            previous = _read_json(previous_path)
        except (ValueError, OSError):
            warnings.append('Previous status snapshot could not be read; observations start afresh')
        if previous.get('task_identity') != identity:
            previous = {}

    totals = {state: 0 for state in STATES}
    active_samples = []
    for index in range(shards):
        path = job_dir/f'shard{index}'/'manifest.json'
        rows = {}
        if path.exists():
            manifest = _read_json(path)
            for entry in manifest['samples']:
                sample_id = entry['id']
                if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id in rows:
                    raise ValueError(f'Invalid or duplicate sample ID in shard{index}')
                rows[sample_id] = entry
        # Each manifest contains the full plan. Only this shard's assigned IDs
        # contribute, even when unassigned rows are marked success or running.
        for sample_id in range(index, samples, shards):
            state = rows.get(sample_id, {}).get('state', 'pending')
            if state not in totals:
                raise ValueError(f'Unknown sample state in shard{index}: {state}')
            totals[state] += 1
            if state == 'running':
                active_samples.append({'shard_index': index,
                    **read_progress(job_dir/f'shard{index}'/'projects'/rows[sample_id].get('project_subdir', ''), sample_id, now)})

    runtime_path = job_dir/'parallel_workers.json'
    runtime = _read_json(runtime_path) if runtime_path.exists() else {'workers': []}
    workers, engines = [], {}
    for record in runtime.get('workers', []):
        if not isinstance(record, dict):
            raise ValueError('Invalid worker runtime record')
        row = _observe_worker(record, engines, process_api)
        if row['index'] >= shards or any(old['index'] == row['index'] for old in workers):
            raise ValueError('Duplicate or out-of-range worker runtime identity')
        # Windows can recycle an exited worker's PID for a later shard. Its
        # historical record is harmless once creation-time checks reject it.
        if row['alive'] is True and any(old['alive'] is True and old['pid'] == row['pid'] for old in workers):
            raise ValueError('Multiple live shard records use the same worker PID')
        workers.append(row)
    alive = sum(row['alive'] is True for row in workers)
    for sample in active_samples:
        matching = next((row for row in workers if row['index'] == sample['shard_index']), None)
        sample['worker_pid'] = matching['pid'] if matching else None
        sample['worker_alive'] = matching['alive'] if matching else False
    engine_rows = sorted(engines.values(), key=lambda row: row['pid'])
    previous_engines = {row['pid']: row for row in previous.get('native_engines', [])}
    for row in engine_rows:
        old = previous_engines.get(row['pid'], {})
        same_process = (row['created_at_unix'] is not None and old.get('created_at_unix') is not None and
                        abs(row['created_at_unix']-old['created_at_unix']) <= 2.)
        row['cpu_seconds_since_previous'] = None
        if same_process and row['cpu_seconds'] is not None and old.get('cpu_seconds') is not None:
            delta = row['cpu_seconds']-old['cpu_seconds']
            if delta >= 0:
                row['cpu_seconds_since_previous'] = delta

    low_since = None
    if alive >= 2 and len(engine_rows) <= 1:
        low_since = previous.get('low_engine_observed_since_unix')
        previous_time = previous.get('observed_at_unix')
        # A missing/stale snapshot does not establish continuous low concurrency.
        if (not _finite_number(low_since) or not _finite_number(previous_time) or
                not 0 <= now-previous_time <= 90. or low_since > now):
            low_since = now
        if now-low_since >= 90.:
            warnings.append('At least 2 workers but at most 1 visible native engine for >=90s of observations; '
                            'check initialization, license availability, resources and process permissions. '
                            'This does not prove serial execution.')
    if any(row['issues'] for row in workers) or any(row['issues'] for row in engine_rows):
        warnings.append('Some process observations are incomplete or identity timestamps are unavailable')
    for sample in active_samples:
        if sample['log_read_error']:
            warnings.append(f"shard{sample['shard_index']} sample_{sample['sample_id']:06d} progress log "
                            f"unavailable ({sample['log_read_error']}); other observations remain available")
    return {'schema_version': 1, 'job_dir': str(job_dir), 'task_identity': identity,
            'observed_at_unix': now, 'requested_concurrency': expected_jobs,
            'sample_count': samples, 'shard_count': shards, 'totals': totals,
            'workers_alive': alive, 'native_engine_count': len(engine_rows),
            'workers': workers, 'native_engines': engine_rows, 'active_samples': active_samples,
            'observations_are_lower_bound': True,
            'peak_worker_processes_alive': max(alive, previous.get('peak_worker_processes_alive', 0)),
            'peak_native_engine_count': max(len(engine_rows), previous.get('peak_native_engine_count', 0)),
            'low_engine_observed_since_unix': low_since, 'warnings': warnings}


def write_snapshot(job_dir, snapshot):
    """Atomic replace with a private temporary file for concurrent status readers."""
    path = Path(job_dir)/'parallel_status.json'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.parallel-status-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(snapshot, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def print_status(snapshot):
    totals = snapshot['totals']
    print(f"success={totals['success']}/{snapshot['sample_count']}, "
          f"running={totals['running']}, failed={totals['failed']}, pending={totals['pending']}; "
          f"workers_alive={snapshot['workers_alive']}/{snapshot['requested_concurrency']}, "
          f"native_engines={snapshot['native_engine_count']} (observed lower bound)", flush=True)
    for row in snapshot['active_samples']:
        percent = '?' if row['percent_complete'] is None else f"{row['percent_complete']:.2f}%"
        age = '?' if row['log_age_seconds'] is None else f"{row['log_age_seconds']:.0f}s"
        shutoff = '?' if row['auto_shutoff'] is None else f"{row['auto_shutoff']:.3g}"
        print(f"  shard{row['shard_index']} sample_{row['sample_id']:06d}: {percent}, "
              f"Auto Shutoff={shutoff}, log_age={age}, worker_alive={row['worker_alive']}", flush=True)
    print('  Multiple Python workers do not establish multiple solvers; process permissions may limit visibility.', flush=True)
    for warning in snapshot['warnings']:
        print(f'  Note: {warning}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--expected-jobs', type=int, required=True)
    parser.add_argument('--write-snapshot', action='store_true')
    args = parser.parse_args(argv)
    try:
        snapshot = collect_status(args.job_dir, args.expected_jobs)
        if args.write_snapshot:
            write_snapshot(args.job_dir, snapshot)
        print_status(snapshot)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Monitor failed ({type(error).__name__}): {error}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
