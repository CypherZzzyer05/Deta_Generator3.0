"""Parallel JSON saves survive a brief Windows monitoring read handle."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from deta_generator3.worker import retry_json_write
from deta_generator3.pipeline import atomic_json


class WorkerStorageTests(unittest.TestCase):
    def test_transient_failure_retries_but_non_windows_and_other_errors_do_not(self):
        sharing = PermissionError('test sharing conflict')
        sharing.winerror = 32
        with patch('deta_generator3.worker.sys.platform','win32'),patch('deta_generator3.worker.time.sleep'):
            writer = unittest.mock.Mock(side_effect=[sharing,sharing,'saved'])
            self.assertEqual(retry_json_write(writer)('path',{'x':1}),'saved')
            self.assertEqual(writer.call_count,3)
            writer = unittest.mock.Mock(side_effect=sharing)
            with self.assertRaises(PermissionError): retry_json_write(writer,attempts=2)('path',{})
            self.assertEqual(writer.call_count,2)
        with patch('deta_generator3.worker.sys.platform','linux'):
            writer = unittest.mock.Mock(side_effect=sharing)
            with self.assertRaises(PermissionError): retry_json_write(writer)('path',{})
            self.assertEqual(writer.call_count,1)

    @unittest.skipUnless(sys.platform=='win32','Windows MoveFileEx read-lock regression')
    def test_real_windows_monitor_read_does_not_lose_saved_manifest(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            atomic_json(path,{'state':'running'})
            ready = threading.Event()
            release = threading.Event()
            def reader():
                with path.open('rb') as stream:
                    self.assertTrue(stream.read())
                    ready.set()
                    release.wait(3)
            thread = threading.Thread(target=reader)
            thread.start()
            self.assertTrue(ready.wait(2))
            calls = []
            def writer(path,data):
                calls.append(1)
                try:
                    return atomic_json(path,data)
                except PermissionError:
                    release.set()
                    raise
            try:
                retry_json_write(writer)(path,{'state':'success'})
                self.assertGreaterEqual(len(calls),2)
                self.assertEqual(json.loads(path.read_text()),{'state':'success'})
            finally:
                release.set()
                thread.join(3)


if __name__ == '__main__': unittest.main()
