"""Installed Lumerical API discovery and unmodified signed spectrum checks."""
import importlib.util
import os
import sys
from pathlib import Path
import numpy as np

def locate_lumapi(explicit=None):
    """Prefer the installed v241 API; never silently use a newer PyLumerical package."""
    # 优先使用配置/环境变量明确指定的本机 API，保持与已安装 2024 R1 的版本一致。
    requested = explicit or os.environ.get("LUMERICAL_LUMAPI_PATH")
    if requested:
        path = Path(requested).expanduser()
        if path.is_dir():
            path = path / "lumapi.py"
        if not path.is_file():
            raise FileNotFoundError(f"Configured lumapi.py does not exist: {path}")
        return path.resolve()
    roots = [Path(r'D:\Program Files\Lumerical'), Path(r'C:\Program Files\Lumerical'),
             Path(r'D:\Program Files\ANSYS Inc'), Path(r'C:\Program Files\ANSYS Inc'),
             Path('/opt/lumerical'), Path('/opt/ansys_inc')]
    matches = {p.resolve() for root in roots if root.is_dir() for pattern in
               ('v*/api/python/lumapi.py', 'v*/Lumerical/api/python/lumapi.py') for p in root.glob(pattern)}
    if len(matches) == 1:
        return matches.pop()
    if len(matches) > 1:
        raise ValueError('Multiple Lumerical APIs found; select one with --lumapi or LUMERICAL_LUMAPI_PATH')
    raise FileNotFoundError('Lumerical API not found. Set --lumapi or LUMERICAL_LUMAPI_PATH')

def load_lumapi(explicit=None):
    path = locate_lumapi(explicit)
    existing = sys.modules.get("lumapi")
    if existing is not None:
        # 同一个 Python 进程已加载另一套 API 时拒绝混用，避免版本与 DLL 不对应。
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError("A different lumapi is already loaded; restart Python with the configured API")
        return existing
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("lumapi", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lumapi"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("lumapi", None)
        raise
    return module

def validate_spectrum(wavelength_nm, reflectance, transmittance, settings, status):
    """Reject bad physics; do not hide sign errors or energy errors by clipping."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64).reshape(-1)
    reflectance = np.asarray(reflectance, dtype=np.float64).reshape(-1)
    transmittance = np.asarray(transmittance, dtype=np.float64).reshape(-1)
    grid = settings.wavelength_grid()
    if any(x.shape != grid.shape for x in (wavelength_nm, reflectance, transmittance)):
        raise ValueError("Monitor must return the configured number of wavelength/R/T points")
    if not all(np.isfinite(x).all() for x in (wavelength_nm, reflectance, transmittance)):
        raise ValueError("Nonfinite monitor values")
    # 按波长统一升序，同步排列 R 和 T，使网络第 i 个输入始终对应同一波长。
    order = np.argsort(wavelength_nm)
    wavelength_nm, reflectance, transmittance = [x[order] for x in (wavelength_nm, reflectance, transmittance)]
    if not np.allclose(wavelength_nm, grid, atol=1e-5, rtol=1e-8):
        raise ValueError("Monitor wavelength grid does not match the configured grid")
    if status not in (1, 2):
        raise ValueError(f"FDTD failed or diverged (status={status})")
    # status=1 表示到达时间上限；status=2 表示自动关断，严格采样要求后者。
    if settings.require_auto_shutoff and status != 2:
        raise ValueError("FDTD reached its time limit before auto-shutoff; increase simulation_time_ps")
    tol = settings.quality_tolerance
    # 保留真实数值并检查功率/能量范围；数值容差允许少量离散误差。
    # 这里不靠裁剪或逐条光谱 min-max 缩放来掩盖符号、归一化或收敛错误。
    if min(reflectance.min(), transmittance.min()) < -tol:
        raise ValueError("Negative R/T beyond tolerance; check propagation/monitor directions")
    if max(reflectance.max(), transmittance.max()) > 1 + tol:
        raise ValueError("R/T exceeds unity beyond tolerance; check convergence/source normalization")
    if np.max(reflectance + transmittance) > 1 + tol:
        raise ValueError("R+T exceeds unity beyond tolerance; check convergence and PML")
    # R、T 由源功率归一化；1-R-T 包含传播到 T 监视器之前的材料吸收损耗。
    return {"wavelength_nm": grid, "reflectance": reflectance,
            "transmittance": transmittance, "absorptance": 1 - reflectance - transmittance}
