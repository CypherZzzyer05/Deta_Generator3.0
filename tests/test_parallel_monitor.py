"""Process and file fixtures only; no solver or license is required."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil

from deta_generator3.monitor import (collect_status, main, read_progress, write_snapshot,
                                    LOG_TAIL_BYTES, _open_shared_read, _read_json)


class ProcessFixture:
    def __init__(self, pid, name='python.exe', created=1000., children=(), denied=(), alive=True):
        self.pid, self.process_name, self.created = pid, name, created
        self.descendants, self.denied, self.alive = children, denied, alive
        self.cpu = 7.

    def _get(self, key, value):
        if key in self.denied:
            raise psutil.AccessDenied(self.pid)
        return value

    def is_running(self): return self._get('is_running', self.alive)
    def status(self): return self._get('status', 'running')
    def name(self): return self._get('name', self.process_name)
    def create_time(self): return self._get('create_time', self.created)
    def children(self, recursive=False):
        assert recursive
        return self._get('children', self.descendants)
    def memory_info(self): return self._get('memory_info', SimpleNamespace(rss=1048576))
    def cpu_times(self): return self._get('cpu_times', SimpleNamespace(user=self.cpu, system=3.))
    def cmdline(self): raise AssertionError('Monitor must never inspect engine commands/licenses')


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.job = Path(self.temp.name)/'job with spaces'
        self.job.mkdir()
        self.write('job.json', {'sample_count': 8, 'simulation': {'mesh_policy': 'metal_only'}})
        self.write('parallel_plan.json', {'samples': 8, 'workers': 4}, bom=True)
        self.processes = {}
        def lookup(pid):
            if pid not in self.processes: raise psutil.NoSuchProcess(pid)
            return self.processes[pid]
        self.api = SimpleNamespace(Process=lookup, AccessDenied=psutil.AccessDenied,
            NoSuchProcess=psutil.NoSuchProcess, ZombieProcess=psutil.ZombieProcess,
            STATUS_ZOMBIE=psutil.STATUS_ZOMBIE)

    def write(self, name, data, bom=False):
        path = self.job/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8-sig' if bom else 'utf-8')

    def worker(self, index, pid, **kwargs):
        self.processes[pid] = ProcessFixture(pid, **kwargs)
        return {'index': index, 'pid': pid, 'process_name': 'python',
                'started_at_unix': 1000.}

    def collect(self, now=2000.):
        return collect_status(self.job, 4, now=now, process_api=self.api)

    def test_shards_only_count_their_ids_and_missing_shards_are_pending(self):
        self.write('shard0/manifest.json', {'samples': [
            {'id': i, 'state': 'success' if i != 4 else 'running'} for i in range(8)]})
        self.write('shard1/manifest.json', {'samples': [
            {'id': i, 'state': 'failed' if i == 1 else 'success'} for i in range(8)]}, bom=True)
        status = self.collect()
        self.assertEqual(status['totals'], {'success': 2, 'running': 1, 'failed': 1, 'pending': 4})
        self.assertEqual([row['sample_id'] for row in status['active_samples']], [4])
        self.assertEqual(status['workers_alive'], 0)
        self.assertFalse((self.job/'parallel_status.json').exists())
        self.assertFalse((self.job/'.parallel-controller.lock').exists())

    def test_real_engine_descendants_are_deduplicated_and_measured(self):
        engine = ProcessFixture(303, 'fdtd-engine-ompi-lcl.exe', created=1100.)
        other = ProcessFixture(304, 'fdtd-solutions.exe')
        records = [self.worker(0, 101, children=[engine, other]),
                   self.worker(1, 102, children=[engine])]
        self.write('parallel_workers.json', {'workers': records}, bom=True)
        status = self.collect()
        self.assertEqual(status['workers_alive'], 2)
        self.assertEqual(status['native_engine_count'], 1)
        self.assertTrue(status['observations_are_lower_bound'])
        self.assertEqual(status['native_engines'][0]['rss_bytes'], 1048576)
        self.assertEqual(status['native_engines'][0]['cpu_seconds'], 10.)
        self.assertTrue(all(row['identity_verified'] for row in status['workers']))

    def test_pid_reuse_name_or_creation_time_is_rejected(self):
        records = [self.worker(0, 101, created=1002.01),
                   self.worker(1, 102, name='another.exe'),
                   self.worker(2, 103, created=1002.)]
        self.write('parallel_workers.json', {'workers': records})
        status = self.collect()
        self.assertEqual(status['workers_alive'], 1)
        self.assertEqual(status['workers'][0]['identity_mismatch'], 'creation_time')
        self.assertEqual(status['workers'][1]['identity_mismatch'], 'process_name')
        self.assertTrue(status['workers'][2]['alive'])

    def test_recycled_pid_in_historical_shard_record_does_not_block_later_worker(self):
        old = self.worker(0, 101)
        current = dict(old, index=1, started_at_unix=1500.)
        self.processes[101].created = 1500.
        self.write('parallel_workers.json', {'workers': [old, current]})
        status = self.collect()
        self.assertEqual(status['workers_alive'], 1)
        self.assertFalse(status['workers'][0]['alive'])
        self.assertEqual(status['workers'][0]['identity_mismatch'], 'creation_time')
        self.assertTrue(status['workers'][1]['alive'])

    def test_duplicate_live_worker_identity_or_shard_index_is_still_rejected(self):
        first = self.worker(0, 101)
        self.write('parallel_workers.json', {'workers': [first, dict(first, index=1)]})
        with self.assertRaisesRegex(ValueError, 'Multiple live shard'):
            self.collect()
        self.processes.clear()
        self.write('parallel_workers.json', {'workers': [first, dict(first, pid=102)]})
        with self.assertRaisesRegex(ValueError, 'Duplicate or out-of-range'):
            self.collect()

    def test_access_denied_yields_explicit_unknown_or_lower_bound(self):
        engine = ProcessFixture(303, 'fdtd-engine.exe', denied=('memory_info', 'cpu_times'))
        hidden = ProcessFixture(304, 'fdtd-engine.exe', denied=('name',))
        records = [self.worker(0, 101, denied=('name',)),
                   self.worker(1, 102, denied=('children',)),
                   self.worker(2, 103, children=[engine, hidden])]
        self.write('parallel_workers.json', {'workers': records})
        status = self.collect()
        self.assertIsNone(status['workers'][0]['alive'])
        self.assertEqual(status['workers_alive'], 2)
        self.assertEqual(status['native_engine_count'], 1)
        self.assertIsNone(status['native_engines'][0]['rss_bytes'])
        self.assertIsNone(status['native_engines'][0]['cpu_seconds'])
        self.assertTrue(status['warnings'])

    def test_log_tail_is_bounded_latest_and_fallback_compatible(self):
        directory = self.job/'shard0/projects'
        directory.mkdir(parents=True)
        path = directory/'sample_000004_p0.log'
        path.write_text('99% complete. Auto Shutoff: 1e-9\n' + 'x'*LOG_TAIL_BYTES +
                        '\n12.1% complete. Auto Shutoff: 0.05\n22.1444% complete. Auto Shutoff: 8.36967e-05\n',
                        encoding='utf-8')
        os.utime(path, (1900., 1900.))
        result = read_progress(directory, 4, 2000.)
        self.assertEqual(result['percent_complete'], 22.1444)
        self.assertEqual(result['auto_shutoff'], 8.36967e-05)
        self.assertEqual(result['log_age_seconds'], 100.)
        self.assertNotIn('raw', result)
        path.rename(directory/'sample_000004.log')
        self.assertEqual(read_progress(directory, 4, 2000.)['percent_complete'], 22.1444)
        self.assertIsNone(read_progress(directory, 5, 2000.)['percent_complete'])

    def test_inaccessible_progress_log_keeps_process_and_other_status_observations(self):
        self.write('shard0/manifest.json', {'samples': [{'id': 0, 'state': 'running'}]})
        self.write('parallel_workers.json', {'workers': [self.worker(0, 101)]})
        original = _open_shared_read
        def deny_log(path, **kwargs):
            if path.name == 'sample_000000_p0.log':
                raise PermissionError('test-only secret that must never be printed')
            return original(path, **kwargs)
        with patch('deta_generator3.monitor._open_shared_read', side_effect=deny_log):
            status = self.collect()
        self.assertEqual(status['workers_alive'], 1)
        self.assertEqual(status['totals']['running'], 1)
        self.assertEqual(status['active_samples'][0]['log_read_error'], 'PermissionError')
        self.assertTrue(any('progress log unavailable' in warning for warning in status['warnings']))
        self.assertNotIn('test-only secret', json.dumps(status))

    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing regression')
    def test_windows_shared_read_allows_open_source_rename_and_closes_descriptor(self):
        # Share-delete permits moving the open source. Win32 MoveFileEx still
        # refuses replacing an open destination, even with share-delete, so
        # that different claim must not be encoded as a passing regression.
        for binary in (False, True):
            with self.subTest(binary=binary):
                name = 'sample_000000_p0.log' if binary else '\u4eff\u771f_manifest.json'
                path = self.job/name
                path.write_bytes(b'{"generation": 0}\r\n')
                with _open_shared_read(path, binary=binary) as reader:
                    descriptor = reader.fileno()
                    relocated = path.with_suffix('.relocated')
                    os.replace(path, relocated)
                    original = reader.read()
                    self.assertIn(b'"generation": 0' if binary else '"generation": 0', original)
                    self.assertEqual(_read_json(relocated)['generation'], 0)
                with self.assertRaises(OSError):
                    os.fstat(descriptor)

    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing regression')
    def test_json_file_is_closed_before_parsing_so_atomic_replace_is_not_blocked(self):
        path = self.job/'manifest_parse_overlap.json'
        path.write_text('{"generation": 0}', encoding='utf-8')
        original_loads = json.loads
        def replace_during_parse(contents, *args, **kwargs):
            replacement = self.job/'manifest_parse_overlap.tmp'
            replacement.write_text('{"generation": 1}', encoding='utf-8')
            os.replace(replacement, path)
            return original_loads(contents, *args, **kwargs)
        with patch('deta_generator3.monitor.json.loads', side_effect=replace_during_parse):
            self.assertEqual(_read_json(path)['generation'], 0)
        self.assertEqual(_read_json(path)['generation'], 1)

    def test_shared_json_open_missing_or_denied_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            _read_json(self.job/'missing.json')
        with patch('deta_generator3.monitor._open_shared_read', side_effect=PermissionError('test-only')):
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--job-dir', str(self.job), '--expected-jobs', '4']), 1)

    def test_snapshot_accumulates_peaks_cpu_delta_and_cautious_sustained_warning(self):
        engine = ProcessFixture(303, 'fdtd-engine.exe', created=1100.)
        records = [self.worker(0, 101, children=[engine]), self.worker(1, 102)]
        self.write('parallel_workers.json', {'workers': records})
        first = self.collect(now=2000.)
        write_snapshot(self.job, first)
        engine.cpu += 30.
        second = self.collect(now=2060.)
        self.assertEqual(second['native_engines'][0]['cpu_seconds_since_previous'], 30.)
        self.assertFalse(any('>=90s' in warning for warning in second['warnings']))
        write_snapshot(self.job, second)
        third = self.collect(now=2090.)
        self.assertTrue(any('does not prove serial' in warning for warning in third['warnings']))
        self.assertEqual(third['peak_worker_processes_alive'], 2)
        self.processes.clear()
        write_snapshot(self.job, third)
        finished = self.collect(now=2120.)
        self.assertEqual(finished['workers_alive'], 0)
        self.assertEqual(finished['peak_worker_processes_alive'], 2)
        self.assertEqual(finished['peak_native_engine_count'], 1)
        self.assertIsNone(finished['low_engine_observed_since_unix'])
        self.assertFalse(list(self.job.glob('.parallel-status-*.tmp')))

    def test_stale_or_different_task_snapshot_does_not_create_false_warning_or_peak(self):
        records = [self.worker(0, 101), self.worker(1, 102)]
        self.write('parallel_workers.json', {'workers': records})
        previous = self.collect(now=2000.)
        write_snapshot(self.job, previous)
        status = self.collect(now=3000.)
        self.assertEqual(status['low_engine_observed_since_unix'], 3000.)
        self.assertFalse(any('>=90s' in warning for warning in status['warnings']))
        previous.update(task_identity='another-task', peak_native_engine_count=10)
        write_snapshot(self.job, previous)
        self.assertEqual(self.collect(now=2030.)['peak_native_engine_count'], 0)

    def test_reused_engine_pid_does_not_generate_a_cpu_delta(self):
        engine = ProcessFixture(303, 'fdtd-engine.exe', created=1100.)
        self.write('parallel_workers.json', {'workers': [self.worker(0, 101, children=[engine])]})
        write_snapshot(self.job, self.collect())
        engine.created = 1500.
        engine.cpu = 1000.
        row = self.collect(now=2030.)['native_engines'][0]
        self.assertIsNone(row['cpu_seconds_since_previous'])

    def test_manifest_running_is_shown_with_latest_progress_and_worker_liveness(self):
        self.write('shard0/manifest.json', {'samples': [{'id': 0, 'state': 'running'}]})
        project = self.job/'shard0/projects'
        project.mkdir()
        (project/'sample_000000_p0.log').write_text('42.5% complete. Auto Shutoff: 0.002\n')
        self.write('parallel_workers.json', {'workers': [self.worker(0, 101)]})
        status = self.collect()
        self.assertEqual(status['active_samples'][0]['percent_complete'], 42.5)
        self.assertTrue(status['active_samples'][0]['worker_alive'])
        self.processes.clear()
        status = self.collect()
        self.assertEqual(status['totals']['running'], 1)
        self.assertFalse(status['active_samples'][0]['worker_alive'])

    def test_ten_registered_workers_count_ten_only_if_ten_engines_are_observed(self):
        self.write('job.json', {'sample_count': 100, 'simulation': {'mesh_policy': 'metal_only'}})
        self.write('parallel_plan.json', {'samples': 100, 'workers': 10})
        records = []
        for index in range(10):
            engine = ProcessFixture(300+index, 'fdtd-engine.exe')
            records.append(self.worker(index, 100+index, children=[engine]))
        self.write('parallel_workers.json', {'workers': records})
        status = collect_status(self.job, 10, now=2000., process_api=self.api)
        self.assertEqual(status['workers_alive'], 10)
        self.assertEqual(status['native_engine_count'], 10)
        self.assertEqual(status['totals']['pending'], 100)

    def test_child_vanishing_during_observation_does_not_break_status(self):
        engine = ProcessFixture(303, 'fdtd-engine.exe')
        with patch.object(engine, 'name', side_effect=psutil.NoSuchProcess(303)):
            self.write('parallel_workers.json', {'workers': [self.worker(0, 101, children=[engine])]})
            status = self.collect()
        self.assertEqual(status['workers_alive'], 1)
        self.assertEqual(status['native_engine_count'], 0)
        self.assertTrue(status['warnings'])

    def test_legacy_runtime_records_are_observable_but_marked_unverified(self):
        record = self.worker(0, 101)
        record.pop('started_at_unix')
        self.write('parallel_workers.json', {'workers': [record]})
        row = self.collect()['workers'][0]
        self.assertTrue(row['alive'])
        self.assertFalse(row['identity_verified'])
        self.assertTrue(row['issues'])

    def test_cli_is_read_only_unless_snapshot_requested_and_fails_on_broken_input(self):
        args = ['--job-dir', str(self.job), '--expected-jobs', '4']
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        self.assertIn('workers_alive=0/4', stdout.getvalue())
        self.assertIn('native_engines=0', stdout.getvalue())
        self.assertFalse((self.job/'parallel_status.json').exists())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args+['--write-snapshot']), 0)
        self.assertTrue((self.job/'parallel_status.json').exists())
        self.assertFalse((self.job/'.parallel-controller.lock').exists())
        (self.job/'shard0').mkdir()
        (self.job/'shard0/manifest.json').write_text('{broken', encoding='utf-8')
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(args), 1)
        self.assertIn('Monitor failed', stderr.getvalue())

    def test_atomic_replace_failure_preserves_previous_snapshot_and_cleans_temp(self):
        original = {'marker': 'previous'}
        self.write('parallel_status.json', original)
        with patch('deta_generator3.monitor.os.replace', side_effect=PermissionError('test')):
            with self.assertRaises(PermissionError):
                write_snapshot(self.job, {'marker': 'next'})
        self.assertEqual(json.loads((self.job/'parallel_status.json').read_text()), original)
        self.assertFalse(list(self.job.glob('.parallel-status-*.tmp')))


if __name__ == '__main__':
    unittest.main()
