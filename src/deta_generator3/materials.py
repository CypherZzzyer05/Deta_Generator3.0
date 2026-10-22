"""Import documented As2Se3 n,k; never silently substitute silicon or As2S3."""
from pathlib import Path
import numpy as np

C0 = 299792458.
AS2SE3_NAME = "As2Se3 - documented sampled nk"


def optical_table(settings):
    if settings.as2se3_model == "nk_csv":
        table = np.loadtxt(Path(settings.as2se3_nk_csv), delimiter=",", skiprows=1)
        if table.ndim != 2 or table.shape[1] != 3 or len(table) < 8:
            raise ValueError("n,k CSV requires wavelength_nm,n,k and at least eight rows")
        if not np.isfinite(table).all() or np.any(table[:, :2] <= 0) or np.any(table[:, 2] < 0):
            raise ValueError("Invalid passive optical constants")
        if np.any(np.diff(table[:, 0]) <= 0):
            raise ValueError("Optical wavelengths must increase strictly")
        if table[0, 0] > settings.wavelength_start_nm or table[-1, 0] < settings.wavelength_stop_nm:
            raise ValueError("n,k table must cover the entire simulation band; no extrapolation")
        return table
    waves = np.linspace(settings.wavelength_start_nm, settings.wavelength_stop_nm, 301)
    um2 = (waves / 1000) ** 2
    # Purified S2, Table 4: optional MID-IR lossless approximation only.
    eps = -22.822 + 30.52834 * um2 / (um2 - .18238**2) + .00639 * um2 / (um2 - 11.44227**2)
    return np.column_stack((waves, np.sqrt(eps), np.zeros_like(waves)))


def install_material(fdtd, settings):
    table = optical_table(settings)
    sampled = np.column_stack((C0 / (table[:, 0] * 1e-9), (table[:, 1] + 1j * table[:, 2])**2))
    if not fdtd.materialexists(AS2SE3_NAME):
        fdtd.setmaterial(fdtd.addmaterial("Sampled 3D data"), "name", AS2SE3_NAME)
    fdtd.setmaterial(AS2SE3_NAME, "max coefficients", 8)
    fdtd.setmaterial(AS2SE3_NAME, "tolerance", 1e-4)
    fdtd.setmaterial(AS2SE3_NAME, "sampled data", sampled[np.argsort(sampled[:, 0].real)])
    return table


def validate_material_fit(fdtd, settings, table):
    # Fit bounds must match the actual source bandwidth. Striding the grid
    # omitted its stop wavelength and queried a slightly different fit.
    waves = settings.wavelength_grid()
    frequencies = C0 / (waves * 1e-9)
    fit = np.asarray(fdtd.getfdtdindex(AS2SE3_NAME, frequencies, frequencies.min(), frequencies.max())).reshape(-1)
    if fit.shape != waves.shape:
        raise ValueError("As2Se3 fit did not return every requested wavelength")
    expected = np.interp(waves, table[:, 0], table[:, 1]) + 1j * np.interp(waves, table[:, 0], table[:, 2])
    error = float(np.max(np.abs(fit - expected)))
    if not np.isfinite(fit).all() or np.min(fit.imag) < -1e-5 or error > .03:
        raise ValueError(f"As2Se3 material fit unacceptable: max |delta n+ik|={error:.5g}")
    return {"max_complex_index_error": error, "source": settings.as2se3_source,
            "fit_band_nm": [settings.wavelength_start_nm, settings.wavelength_stop_nm],
            "model": settings.as2se3_model, "fit_wavelength_nm": waves.tolist(),
            "fit_n": fit.real.tolist(), "fit_k": fit.imag.tolist()}
