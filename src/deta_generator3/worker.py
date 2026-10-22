"""Parallel child entry; tolerate brief Windows read locks during JSON replacement.

The underlying sampler, simulator and dataset implementation are unchanged.
Each child has its own output-directory lock and remains its only writer.
"""
import functools
import sys
import time


def retry_json_write(writer, attempts=40, delay=.05):
    @functools.wraps(writer)
    def write(path,data):
        for attempt in range(attempts):
            try:
                return writer(path,data)
            except PermissionError as error:
                # Retry Windows access/sharing/lock conflicts for <=1.95 s;
                # permanent permission failures still surface to the user.
                if (sys.platform != 'win32' or getattr(error,'winerror',None) not in (5,32,33)
                        or attempt+1 == attempts):
                    raise
                time.sleep(delay)
    return write


def main(argv=None):
    from . import cli,pipeline
    original_pipeline,original_cli = pipeline.atomic_json,cli.atomic_json
    wrapped = retry_json_write(original_pipeline)
    pipeline.atomic_json = wrapped
    cli.atomic_json = wrapped
    try:
        return cli.main(argv)
    finally:
        pipeline.atomic_json,cli.atomic_json = original_pipeline,original_cli


if __name__ == '__main__':
    raise SystemExit(main())
