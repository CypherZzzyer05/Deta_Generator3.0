"""Real 3D periodic FDTD of As2Se3/Au(100 nm)/glass(350 nm) in air."""
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .native import load_lumapi, validate_spectrum
from .pipeline import atomic_json
from .materials import AS2SE3_NAME, C0, install_material, validate_material_fit
from .vision import hotspots, poynting
from .fixed_fov import periodic_complex_sample


class PairedSimulator:
    def __init__(self, settings, lumapi_path=None, execution_device="CPU"):
        if execution_device not in ("CPU", "GPU"):
            raise ValueError("execution_device must be CPU or GPU")
        self.settings = settings
        self.execution_device = execution_device
        self.api = load_lumapi(lumapi_path)
        self.fdtd = None

    def __enter__(self):
        s = self.settings
        self.fdtd = self.api.FDTD(hide=s.hide_fdtd)
        try:
            self.version = str(self.fdtd.version())
            self.fdtd.setresource("FDTD", 1, "processes", str(s.processes))
            self.fdtd.setresource("FDTD", 1, "threads", str(s.threads))
            for name in (s.gold_material, s.glass_material):
                if not self.fdtd.materialexists(name):
                    raise ValueError(f"Required material missing: {name}")
            self.optical_table = install_material(self.fdtd, s)
        except BaseException:
            self.fdtd.close()
            self.fdtd = None
            raise
        return self

    def __exit__(self, *args):
        if self.fdtd is not None:
            self.fdtd.close()
            self.fdtd = None

    def build(self, g):
        f, s = self.fdtd, self.settings
        nm = 1e-9
        p, h, pad = g.p * nm, g.h * nm, s.air_padding_nm * nm
        glass_bottom = -(s.gold_thickness_nm + s.glass_thickness_nm) * nm
        f.switchtolayout()
        f.selectall()
        f.delete()
        f.addfdtd(properties=OrderedDict([
            ("dimension", "3D"), ("x", 0.), ("y", 0.), ("x span", p), ("y span", p),
            ("z min", glass_bottom - pad), ("z max", h + pad),
            ("x min bc", "Periodic"), ("x max bc", "Periodic"),
            ("y min bc", "Periodic"), ("y max bc", "Periodic"),
            ("z min bc", "PML"), ("z max bc", "PML"), ("pml profile", 3),
            ("index", s.background_index), ("mesh accuracy", s.mesh_accuracy),
            ("simulation time", s.simulation_time_ps * 1e-12), ("auto shutoff min", s.auto_shutoff_min)]))
        if s.mesh_refinement is not None:
            f.setnamed('FDTD','mesh refinement',s.mesh_refinement)
        # The installed v241 GPU engine explicitly requires its express-mode flag.
        if getattr(self, "execution_device", "CPU") == "GPU":
            f.setnamed("FDTD", "express mode", True)
        f.addrect(name="glass_350nm", x=0., y=0., x_span=3*p, y_span=3*p,
                  z_min=glass_bottom, z_max=-s.gold_thickness_nm*nm, material=s.glass_material)
        f.addrect(name="gold_100nm", x=0., y=0., x_span=3*p, y_span=3*p,
                  z_min=-s.gold_thickness_nm*nm, z_max=0., material=s.gold_material)
        # Explicit vertices avoid ambiguity in the solver's native rotation sign.
        for quadrant, vertices in enumerate(g.vertices(), 1):
            f.addpoly(name=f"as2se3_Q{quadrant}", x=0., y=0., vertices=vertices*nm,
                      z_min=0., z_max=h, material=AS2SE3_NAME)
        f.addmesh(name="structure_mesh", x=0., y=0., x_span=p, y_span=p,
                  z_min=glass_bottom - s.mesh_nm*nm, z_max=max(g.h, max(s.field_z_nm))*nm + s.mesh_nm*nm,
                  dx=s.mesh_nm*nm, dy=s.mesh_nm*nm, dz=s.mesh_nm*nm)
        f.addmesh(name="gold_mesh", x=0., y=0., x_span=p, y_span=p,
                  z_min=-(s.gold_thickness_nm+s.metal_mesh_nm)*nm, z_max=s.metal_mesh_nm*nm,
                  dx=s.mesh_nm*nm, dy=s.mesh_nm*nm, dz=s.metal_mesh_nm*nm)
        if s.air_mesh_nm is not None:
            # Keep the finer local metal/structure overrides. Overlapping
            # regions use the smallest step; this constrains the outer air too.
            f.addmesh(name="air_resolution_mesh", x=0., y=0., x_span=p, y_span=p,
                z_min=glass_bottom-pad, z_max=h+pad,
                dx=s.mesh_nm*nm, dy=s.mesh_nm*nm, dz=s.air_mesh_nm*nm)
        f.addplane(properties=OrderedDict([
            ("name", "source"), ("injection axis", "z"), ("direction", "Backward"),
            ("x", 0.), ("y", 0.), ("x span", p), ("y span", p), ("z", h+.5*pad),
            ("angle theta", 0.), ("polarization angle", 0.), ("amplitude", 1.),
            ("override global source settings", True),
            ("wavelength start", s.wavelength_start_nm*nm), ("wavelength stop", s.wavelength_stop_nm*nm)]))
        f.cwnorm()
        f.setglobalmonitor("use source limits", True)
        f.setglobalmonitor("use wavelength spacing", True)
        f.setglobalmonitor("frequency points", s.spectrum_points)
        for name, z in (("R", h+.75*pad), ("T", glass_bottom-.5*pad)):
            f.addpower(name=name, monitor_type="2D Z-normal", x=0., y=0., x_span=p, y_span=p, z=z)
            f.set("down sample x", s.power_downsample)
            f.set("down sample y", s.power_downsample)
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz"):
                f.set("output " + component, False)
            f.set("output power", True)
        for i, z in enumerate(s.field_z_nm):
            f.addprofile(name=f"field_xy_{i}", monitor_type="2D Z-normal", x=0., y=0.,
                         x_span=p, y_span=p, z=z*nm)
            f.set("override global monitor settings", True)
            f.set("sample spacing", "custom")
            f.set("custom frequency samples", C0 / (np.asarray(s.field_wavelengths_nm)*nm))
            f.set("spatial interpolation", "specified position")
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                f.set("output " + component, True)
        self.verify_layout(g)

    def verify_layout(self, geometry):
        f, s = self.fdtd, self.settings
        checks = {}
        for name, limits in (("glass_350nm", (-450, -100)), ("gold_100nm", (-100, 0))):
            actual = [float(f.getnamed(name, key))/1e-9 for key in ("z min", "z max")]
            if not np.allclose(actual, limits, atol=1e-6, rtol=0):
                raise ValueError("Native finite stack differs: "+name)
            checks[name] = actual
        for q, vertices in enumerate(geometry.vertices(), 1):
            name = f"as2se3_Q{q}"
            if (not np.allclose(np.asarray(f.getnamed(name, "vertices"))/1e-9, vertices, atol=1e-6, rtol=0)
                    or not np.allclose([f.getnamed(name, "z min"), f.getnamed(name, "z max")],
                                       [0, geometry.h*1e-9], atol=1e-15, rtol=0)):
                raise ValueError("Native tetramer vertices/height differ")
        for i, z in enumerate(s.field_z_nm):
            if not np.isclose(float(f.getnamed(f"field_xy_{i}", "z"))/1e-9, z, atol=1e-6, rtol=0):
                raise ValueError("Native observation plane moved")
        if not np.allclose([f.getnamed("FDTD", "x span"), f.getnamed("FDTD", "y span")],
                           [geometry.p*1e-9]*2, atol=1e-15, rtol=0):
            raise ValueError("Native periodic cell differs")
        self.layout_report = dict(verified=True, layer_z_nm=checks,
            tetramer_z_nm=[0, geometry.h], centers_nm=geometry.centers().tolist(),
            field_z_nm=list(s.field_z_nm), transmission_medium="air below finite glass")
        return self.layout_report

    def material_identity(self):
        waves = self.settings.wavelength_grid()
        frequencies = C0/(waves*1e-9)
        identity = {}
        for key, material in (("As2Se3", AS2SE3_NAME), ("Au", self.settings.gold_material),
                              ("glass", self.settings.glass_material)):
            fit = np.asarray(self.fdtd.getfdtdindex(material, frequencies,
                frequencies.min(), frequencies.max())).reshape(-1)
            if fit.shape != waves.shape or not np.isfinite(fit).all() or np.min(fit.imag) < -1e-5:
                raise ValueError("Invalid fitted material index: "+key)
            identity[key] = dict(material=material, fitted_index_ri=np.c_[fit.real, fit.imag].tolist(),
                                 minimum_k=float(fit.imag.min()))
        return identity

    def fields(self, g):
        s, f = self.settings, self.fdtd
        planes_e, planes_h, masks, labels, features, native = [], [], [], [], [], {}
        span=g.p if s.field_xy_span_nm is None else s.field_xy_span_nm
        pixel_nm = ((np.arange(s.image_size)+.5)/s.image_size-.5) * span
        yy, xx = np.meshgrid(pixel_nm*1e-9, pixel_nm*1e-9, indexing="ij")
        queries = np.stack((yy, xx), axis=-1)
        for plane, z in enumerate(s.field_z_nm):
            e, h = f.getresult(f"field_xy_{plane}", "E"), f.getresult(f"field_xy_{plane}", "H")
            for axis in ("x", "y", "z", "lambda"):
                if not np.allclose(e[axis], h[axis], rtol=1e-10, atol=1e-16):
                    raise ValueError("E and H coordinates differ; Poynting fields must be co-located")
            x, y = np.asarray(e["x"]).reshape(-1), np.asarray(e["y"]).reshape(-1)
            waves = np.asarray(e["lambda"]).reshape(-1)*1e9
            order = np.argsort(waves)
            if not np.allclose(waves[order], s.field_wavelengths_nm, atol=1e-6):
                raise ValueError("Field wavelength registration failed")
            if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
                raise ValueError("Native monitor axes must be ascending")
            fields = []
            for key, data in (("E", e), ("H", h)):
                array = np.asarray(data[key])
                if array.shape != (len(x), len(y), 1, len(waves), 3):
                    raise ValueError(f"Unexpected native {key} shape: {array.shape}")
                array = array[:, :, 0, order, :].transpose(1, 0, 2, 3)
                native[f"native_{key}_{plane}"] = array.astype(np.complex64)
                if s.field_xy_span_nm is None:
                    uniform = RegularGridInterpolator((y, x), array, bounds_error=True)(queries)
                else:
                    uniform,_=periodic_complex_sample(native[f'native_{key}_{plane}'],x*1e9,y*1e9,pixel_nm)
                fields.append(uniform.transpose(2, 0, 1, 3).astype(np.complex64))
            native[f"native_x_nm_{plane}"] = x*1e9
            native[f"native_y_nm_{plane}"] = y*1e9
            planes_e.append(fields[0])
            planes_h.append(fields[1])
            plane_masks, plane_labels, plane_features = [], [], []
            intensity = np.sum(np.abs(fields[0])**2, axis=-1)  # E_inc amplitude explicitly 1 V/m.
            for wave, image in zip(s.field_wavelengths_nm, intensity):
                mask, label, stats = hotspots(image, s.hotspot_seed_fraction,
                    s.hotspot_grow_fraction, s.hotspot_min_enhancement,periodic_xy=s.field_xy_span_nm is None)
                stats.update(z_nm=z, wavelength_nm=wave)
                plane_masks.append(mask)
                plane_labels.append(label)
                plane_features.append(stats)
            masks.append(plane_masks)
            labels.append(plane_labels)
            features.append(plane_features)
        e, h = np.stack(planes_e), np.stack(planes_h)
        # Fixed source amplitude; preserve absolute enhancement across samples/wavelengths.
        intensity = np.sum(np.abs(e)**2, axis=-1).astype(np.float32)
        flux = poynting(e, h).astype(np.float32)
        incident_flux = .5 * s.background_index * 8.8541878128e-12 * C0
        if not np.isfinite(intensity).all() or not np.isfinite(flux).all():
            raise ValueError("Nonfinite near-field values")
        return dict(electric=e, magnetic=h, intensity=intensity, poynting_w_m2=flux,
                    poynting_incident_normalized=flux/incident_flux,
                    hotspot_mask=np.asarray(masks, dtype=np.uint8), hotspot_labels=np.asarray(labels, dtype=np.int32),
                    hotspot_features_json=json.dumps(features, allow_nan=False),
                    pixel_x_nm=pixel_nm, pixel_y_nm=pixel_nm, **native)

    def run(self, g, project_path, starting_attempt=0):
        if type(starting_attempt) is not int or not 0 <= starting_attempt < 4:
            raise ValueError("starting_attempt must select one of the four existing time limits")
        self.build(g)
        fit = validate_material_fit(self.fdtd, self.settings, self.optical_table)
        path = Path(project_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        attempts, elapsed = [], 0.
        for attempt in range(starting_attempt, 4):
            limit = self.settings.simulation_time_ps * 2**attempt
            if attempt:
                self.fdtd.switchtolayout()
                self.fdtd.setnamed("FDTD", "simulation time", limit*1e-12)
            self.fdtd.save(str(path))
            start = time.monotonic()
            if self.execution_device == "GPU":
                self.fdtd.run("FDTD", "GPU")
            else:
                self.fdtd.run()
            wall = time.monotonic() - start
            elapsed += wall
            status = int(np.asarray(self.fdtd.getresult("FDTD", "status")).item())
            attempts.append(dict(time_limit_ps=limit, elapsed_seconds=wall, status=status))
            atomic_json(path.with_suffix(".attempts.json"), dict(attempts=attempts, status=status,
                starting_attempt=starting_attempt, geometry_values=g.array().tolist(), material_fit=fit))
            if status != 1:
                break
        # Keep diagnostics even if the final spectrum is rejected by a quality gate.
        self.fdtd.save(str(path))
        if status not in (1, 2):
            raise ValueError(f"FDTD solver did not produce valid results (status={status}); inspect {path.stem}_p0.log")
        if status != 2:
            raise ValueError("FDTD did not reach auto-shutoff within the maximum allowed time limit")
        r, t = self.fdtd.getresult("R", "T"), self.fdtd.getresult("T", "T")
        if not np.allclose(r["lambda"], t["lambda"], rtol=1e-10, atol=1e-16):
            raise ValueError("R/T wavelength grids differ")
        wave = np.asarray(r["lambda"])*1e9
        reflection, transmission = np.asarray(r["T"]), -np.asarray(t["T"])
        try:
            result = validate_spectrum(wave, reflection, transmission, self.settings, status)
        except ValueError as error:
            from .pipeline import atomic_npz
            atomic_npz(path.with_suffix(".rejected_spectrum.npz"), wavelength_nm=wave,
                       reflectance=reflection, transmittance=transmission, status=status,
                       rejection=str(error))
            raise
        result.update(self.fields(g))
        result.update(status=status, elapsed_seconds=elapsed, attempts_json=json.dumps(attempts),
                      starting_attempt=starting_attempt,
                      execution_device=self.execution_device,
                      fdtd_version=self.version, material_fit_json=json.dumps(fit),
                      field_wavelength_nm=np.array(self.settings.field_wavelengths_nm),
                      field_z_nm=np.array(self.settings.field_z_nm), native_layout_verified=True,
                      layout_report_json=json.dumps(self.layout_report),
                      material_identity_json=json.dumps(self.material_identity()))
        return result
