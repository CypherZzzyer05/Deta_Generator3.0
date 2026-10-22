"""Independent schema: do not reinterpret the legacy silicon datasets."""
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES = ("p", "dx", "dy", "h", "l", "alpha_1and3", "alpha_2and4")
UNITS = ("nm",) * 5 + ("deg",) * 2


def geometry_group_key(values):
    """Canonical original/y-mirror group for fixed normal x-polarized excitation."""
    row = np.asarray(values, dtype=np.float64)
    if row.shape != (7,) or not np.isfinite(row).all():
        raise ValueError("Expected seven finite geometry values")
    reflected = row.copy()
    reflected[5:] = (-row[[6, 5]]+45) % 90-45
    return min(tuple(np.round(row, 8)), tuple(np.round(reflected, 8)))


@dataclass(frozen=True)
class Geometry:
    p: float
    dx: float
    dy: float
    h: float
    l: float
    alpha_1and3: float
    alpha_2and4: float

    def __post_init__(self):
        values = self.array()
        if not np.isfinite(values).all() or np.any(values[:5] <= 0):
            raise ValueError("Lengths must be positive; all seven values must be finite")
        if np.any(values[5:] < -45) or np.any(values[5:] >= 45):
            raise ValueError("Square angles use the half-open canonical interval [-45,45) deg")
        polygons = self.vertices()
        if any(np.max(np.abs(v)) >= self.p / 2 - 1e-7 for v in polygons):
            raise ValueError("Squares must stay strictly inside their periodic cell")
        for i, a in enumerate(polygons):
            for b in polygons[:i]:
                # Separating axes of BOTH differently rotated polygons are required.
                separated = False
                for poly in (a, b):
                    for edge in np.diff(np.vstack((poly, poly[:1])), axis=0)[:2]:
                        axis = np.array([-edge[1], edge[0]]) / self.l
                        pa, pb = a @ axis, b @ axis
                        separated |= pa.max() < pb.min() - 1e-7 or pb.max() < pa.min() - 1e-7
                if not separated:
                    raise ValueError("Rotated squares overlap or touch")

    def array(self):
        return np.array([getattr(self, n) for n in NAMES], dtype=np.float64)

    def centers(self):
        # Quadrants I, II, III, IV viewed from +z; dx/dy are CENTER distances.
        return np.array([[self.dx, self.dy], [-self.dx, self.dy],
                         [-self.dx, -self.dy], [self.dx, -self.dy]]) / 2

    def angles(self):
        return (self.alpha_1and3, self.alpha_2and4) * 2

    def vertices(self):
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * self.l / 2
        result = []
        for center, angle in zip(self.centers(), self.angles()):
            a = np.deg2rad(angle)
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            result.append(corners @ rotation.T + center)
        return result


@dataclass(frozen=True)
class SimulationSettings:
    wavelength_start_nm: float = 600
    wavelength_stop_nm: float = 1400
    spectrum_points: int = 1000
    power_downsample: int = 1
    field_wavelengths_nm: tuple = (700, 1000, 1300)
    field_z_nm: tuple = (100, 400)  # Fixed LAB planes, independent of unknown h.
    field_xy_span_nm: float | None = None  # None retains the normalized single-cell view.
    image_size: int = 64
    gold_thickness_nm: float = 100
    glass_thickness_nm: float = 350
    gold_material: str = "Au (Gold) - Palik"
    glass_material: str = "SiO2 (Glass) - Palik"
    as2se3_model: str = "nk_csv"
    as2se3_nk_csv: str | None = str(Path(__file__).resolve().parent / "assets/materials/as2se3_joseph_400nm.csv")
    as2se3_source: str = "Joseph et al., DOI:10.1021/acsami.0c12525; author-provided 400-nm-film n,k table"
    mesh_nm: float = 20
    metal_mesh_nm: float = 10
    air_mesh_nm: float | None = None
    mesh_refinement: str | None = None
    mesh_accuracy: int = 3
    simulation_time_ps: float = 2
    auto_shutoff_min: float = 1e-5
    air_padding_nm: float = 1600
    background_index: float = 1
    quality_tolerance: float = .03
    require_auto_shutoff: bool = True
    processes: int = 1
    threads: int = 4
    hide_fdtd: bool = True
    save_projects: bool = True
    hotspot_seed_fraction: float = .5
    hotspot_grow_fraction: float = .2
    hotspot_min_enhancement: float = 2

    def __post_init__(self):
        if not 0 < self.wavelength_start_nm < self.wavelength_stop_nm:
            raise ValueError("Invalid wavelength band")
        if self.spectrum_points != 1000:
            raise ValueError("This schema records 1000 spectrum points")
        for name in ("mesh_nm", "metal_mesh_nm", "simulation_time_ps", "air_padding_nm",
                     "background_index", "gold_thickness_nm", "glass_thickness_nm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gold_thickness_nm != 100 or self.glass_thickness_nm != 350:
            raise ValueError("This experiment fixes Au=100 nm and glass=350 nm")
        if self.air_mesh_nm is not None and (type(self.air_mesh_nm) not in (int,float)
                or not np.isfinite(self.air_mesh_nm) or self.air_mesh_nm <= 0):
            raise ValueError("air_mesh_nm must be a finite positive number or None")
        if self.field_xy_span_nm is not None and (type(self.field_xy_span_nm) not in (int,float)
                or not np.isfinite(self.field_xy_span_nm) or self.field_xy_span_nm<=0):
            raise ValueError('field_xy_span_nm must be a finite positive physical span or None')
        if self.mesh_refinement is not None and self.mesh_refinement not in ('conformal variant 0','conformal variant 1'):
            raise ValueError('mesh_refinement must explicitly select conformal variant 0/1 or remain None')
        if not 0 < self.auto_shutoff_min < 1 or not 0 < self.quality_tolerance < .2:
            raise ValueError("Invalid quality thresholds")
        if self.require_auto_shutoff is not True:
            raise ValueError("Paired datasets require auto-shutoff convergence")
        for name in ("image_size", "mesh_accuracy", "threads", "processes", "power_downsample"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size < 16 or not 1 <= self.mesh_accuracy <= 8:
            raise ValueError("Invalid spatial resolution")
        waves = np.asarray(self.field_wavelengths_nm, dtype=float)
        zs = np.asarray(self.field_z_nm, dtype=float)
        if (waves.ndim != 1 or not len(waves) or not np.isfinite(waves).all()
                or np.any(np.diff(waves) <= 0) or waves[0] < self.wavelength_start_nm
                or waves[-1] > self.wavelength_stop_nm):
            raise ValueError("Field wavelengths must be unique, ascending and inside the source band")
        if zs.ndim != 1 or not len(zs) or not np.isfinite(zs).all() or np.any(np.diff(zs) <= 0):
            raise ValueError("Field planes must be finite, unique and ascending")
        if not 0 < self.hotspot_grow_fraction <= self.hotspot_seed_fraction <= 1:
            raise ValueError("Invalid hysteresis thresholds")
        if not np.isfinite(self.hotspot_min_enhancement) or self.hotspot_min_enhancement <= 0:
            raise ValueError("Invalid hotspot absolute floor")
        if self.as2se3_model != "nk_csv":
            raise ValueError("Generator3 requires the documented measured As2Se3 nk table")
        if self.as2se3_model == "published_s2_lossless" and not 1600 <= self.wavelength_start_nm < self.wavelength_stop_nm <= 10000:
            raise ValueError("Published S2 dispersion restricted to 1.6–10 um here")
        if self.as2se3_model == "nk_csv" and (not self.as2se3_nk_csv or not self.as2se3_source):
            raise ValueError("Measured nk_csv requires a file and provenance")

    def wavelength_grid(self):
        return np.linspace(self.wavelength_start_nm, self.wavelength_stop_nm, self.spectrum_points)


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: Path
    bounds: dict
    sample_count: int = 1000
    seed: int = 20261023
    lumapi_path: str | None = None
    quality_level: str = "pipeline_validation"
    parameter_range_source: str = "Project engineering domain; not validated fabrication bounds"
    simulation: SimulationSettings = field(default_factory=SimulationSettings)

    def __post_init__(self):
        if set(self.bounds) != set(NAMES):
            raise ValueError(f"bounds must contain exactly {NAMES}")
        for i, name in enumerate(NAMES):
            pair = self.bounds[name]
            if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] >= pair[1]:
                raise ValueError(f"Invalid bounds: {name}")
            if i < 5 and pair[0] <= 0:
                raise ValueError("Length bounds must be positive")
            if i >= 5 and list(pair) != [-45, 45]:
                raise ValueError("Angles must cover the canonical [-45,45) degree interval")
        if type(self.sample_count) is not int or self.sample_count < 1 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Positive sample count and nonnegative integer seed required")
        if self.quality_level != "pipeline_validation":
            raise ValueError("These engineering numerics are not globally certified research data")
        s = self.simulation
        if (s.wavelength_start_nm, s.wavelength_stop_nm) != (600, 1400):
            raise ValueError("This project uses the fixed 600–1400nm band")
        if tuple(s.field_wavelengths_nm) != tuple(range(600, 1401, 100)) or tuple(s.field_z_nm) != (100, 400):
            raise ValueError("Use the registered nine wavelengths and fixed z=100/400nm planes")
        if s.field_xy_span_nm != 1600 or s.image_size != 128:
            raise ValueError("Model observations require fixed1600nm/128px fields")
        if max(s.field_z_nm) >= self.lower[3]+s.air_padding_nm/2:
            raise ValueError("Every field plane must stay below the source for all planned heights")

    @property
    def parameter_names(self):
        return NAMES

    @property
    def parameter_units(self):
        return UNITS

    @property
    def lower(self):
        return np.array([self.bounds[n][0] for n in NAMES], dtype=float)

    @property
    def upper(self):
        return np.array([self.bounds[n][1] for n in NAMES], dtype=float)

    def normalize(self, values):
        return (np.asarray(values)-self.lower)/(self.upper-self.lower)

    def sample_geometries(self):
        rng = np.random.default_rng(self.seed)
        result = []
        seen = set()
        for _ in range(max(10000, self.sample_count*1000)):
            try:
                geometry = Geometry(*rng.uniform(self.lower, self.upper))
            except ValueError:
                continue
            key = geometry_group_key(geometry.array())
            if key in seen:
                continue
            seen.add(key)
            result.append(geometry)
            if len(result) == self.sample_count:
                return result
        raise ValueError("Could not sample enough nonoverlapping, distinct geometry groups")

    def metadata(self):
        data = asdict(self)
        data.pop("output_dir")
        data.pop("lumapi_path")
        for key in ("threads", "processes", "hide_fdtd", "save_projects"):
            data["simulation"].pop(key)
        table = Path(self.simulation.as2se3_nk_csv)
        data["material_table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
        data["simulation"]["as2se3_nk_csv"] = "content-addressed:material_table_sha256"
        data["generator"] = "deta-generator3/3.0.0"
        data["implementation_sha256"] = {name: hashlib.sha256(
            (Path(__file__).parent/name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
            for name in ("config.py", "solver.py", "native.py", "materials.py", "fixed_fov.py", "vision.py")}
        return data


def load_config(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    unknown = set(data)-{f.name for f in fields(PipelineConfig)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    s = data.get("simulation", {})
    if s.get("as2se3_nk_csv"):
        s["as2se3_nk_csv"] = str((path.parent/s["as2se3_nk_csv"]).resolve())
    data["simulation"] = SimulationSettings(**s)
    data["output_dir"] = (path.parent/data.get("output_dir", "data")).resolve()
    return PipelineConfig(**data)
