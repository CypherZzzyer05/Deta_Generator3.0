"""Read CPU capacity without changing affinity or the native solver settings."""
import ctypes
import os
import sys
import psutil


def _windows_cpu_info():
    if sys.platform != 'win32':
        return None
    try:
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        count = kernel.GetActiveProcessorCount
        count.argtypes, count.restype = [ctypes.c_ushort],ctypes.c_uint
        groups = kernel.GetActiveProcessorGroupCount
        groups.argtypes, groups.restype = [],ctypes.c_ushort
        total, group_count = int(count(0xffff)),int(groups())
        version = sys.getwindowsversion()
        cross_group_default = (version.build >= 22000 or
                               (version.product_type != 1 and version.build >= 20348))
        return dict(total=total,groups=group_count,cross_group_default=cross_group_default)
    except (AttributeError,OSError):
        return None


def cpu_inventory():
    logical = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    try:
        affinity = len(psutil.Process().cpu_affinity())
    except (AttributeError,psutil.Error):
        affinity = None
    windows = _windows_cpu_info()
    total = max(logical,windows['total']) if windows else logical
    budget,notes = total,[]
    if windows and windows['groups'] > 1:
        # A legacy affinity mask describes only the primary group, not all
        # groups. Do not incorrectly cap modern Windows to its first 64 CPUs.
        if not windows['cross_group_default']:
            budget = min(total,affinity or logical,64)
            notes.append('Older Windows processor groups: conservative per-process CPU budget; total hardware count is reported separately.')
        elif affinity and affinity < min(64,total):
            budget = min(total,affinity)
            notes.append('CPU affinity appears restricted; automatic budget respects that restriction.')
    elif affinity:
        budget = min(total,affinity)
    return dict(psutil_logical_cpus=logical,windows_total_logical_cpus=windows['total'] if windows else None,
                processor_groups=windows['groups'] if windows else None,affinity_count=affinity,
                budget_cpus=max(1,budget),notes=notes)
