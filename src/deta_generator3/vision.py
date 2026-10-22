"""Hotspot segmentation and morphology; intensity is NOT dispersive energy density."""
import numpy as np
from scipy import ndimage


def periodic_components(mask):
    """8-connected components on a periodic XY cell, including diagonal seams."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    parent = np.arange(count + 1)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a, b):
        if a and b:
            parent[root(a)] = root(b)

    height, width = mask.shape
    for offset in (-1, 0, 1):
        for x in range(width):
            join(labels[0, x], labels[-1, (x + offset) % width])
        for y in range(height):
            join(labels[y, 0], labels[(y + offset) % height, -1])
    lookup = np.array([root(i) for i in range(count + 1)])
    unique = np.unique(lookup[1:])
    remap = np.zeros(count + 1, dtype=np.int32)
    for new, old in enumerate(unique, 1):
        remap[lookup == old] = new
    return remap[labels], len(unique)


def hotspots(intensity, seed_fraction=.5, grow_fraction=.2, min_enhancement=2.,periodic_xy=True):
    """Hysteresis on |E|²/|E_inc|², with a physical floor; no forced bright fraction."""
    image = np.asarray(intensity, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("Expected finite nonnegative 2D field enhancement")
    if not 0 < grow_fraction <= seed_fraction <= 1 or not np.isfinite(min_enhancement) or min_enhancement <= 0:
        raise ValueError("Invalid hotspot thresholds")
    if type(periodic_xy) is not bool:
        raise ValueError('Declare whether the observed image edges are periodic')
    def connected_components(mask):
        return periodic_components(mask) if periodic_xy else ndimage.label(mask,structure=np.ones((3,3),dtype=int))
    high = max(min_enhancement, image.max() * seed_fraction)
    low = max(min_enhancement, image.max() * grow_fraction)
    regions, _ = connected_components(image >= low)
    keep = np.unique(regions[image >= high])
    keep = keep[keep != 0]
    mask = np.isin(regions, keep) & (regions != 0)
    labels, count = connected_components(mask)
    components = []
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    for index in range(1, count + 1):
        selected = labels == index
        weights = image[selected]
        # Circular centroid and shortest displacements keep seam-spanning hotspots intact.
        u, v = (xx[selected] + .5) / width, (yy[selected] + .5) / height
        if periodic_xy:
            cu = np.sum(weights * np.exp(2j * np.pi * u))
            cv = np.sum(weights * np.exp(2j * np.pi * v))
            center = np.mod(np.angle([cu, cv]) / (2 * np.pi), 1)
            offsets = (np.column_stack((u, v)) - center + .5) % 1 - .5
        else:
            center=np.average(np.column_stack((u,v)),axis=0,weights=weights)
            offsets=np.column_stack((u,v))-center
        offsets -= np.average(offsets, axis=0, weights=weights)
        covariance = (offsets * weights[:, None]).T @ offsets / weights.sum()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        anisotropy = float((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-30))
        axis = eigenvectors[:, 1]
        angle = float(np.rad2deg(np.arctan2(axis[1], axis[0])) % 180)
        components.append({"label": index, "pixels": int(selected.sum()),
                           "area_fraction": float(selected.mean()),
                           "intensity_sum": float(weights.sum()), "peak": float(weights.max()),
                           "centroid_cell_xy": (center - .5).tolist(),
                           "major_axis_deg_mod180": angle if anisotropy > .05 else None,
                           "anisotropy": anisotropy})
        if not periodic_xy:
            components[-1]['centroid_fov_xy']=components[-1].pop('centroid_cell_xy')
    total = float(image.sum())
    stats = {"peak_enhancement": float(image.max()), "mean_enhancement": float(image.mean()),
             "seed_threshold": float(high), "grow_threshold": float(low),
             "hotspot_area_fraction": float(mask.mean()), "component_count": count,
             "hotspot_intensity_fraction": float(image[mask].sum() / total) if total else 0.,
             "largest_component_area_fraction": max((c["area_fraction"] for c in components), default=0.),
             "components": components, "periodic_xy": periodic_xy,
             "angle_interpretation": "morphological axis only; not energy-flow direction"}
    return mask.astype(np.uint8), labels, stats


def poynting(electric, magnetic):
    """Time averaged real power flow, W/m² for CW-normalized SI fields."""
    e, h = np.asarray(electric), np.asarray(magnetic)
    if e.shape != h.shape or e.shape[-1] != 3 or not np.isfinite(e).all() or not np.isfinite(h).all():
        raise ValueError("E and H must be finite co-located complex [...,3] arrays")
    return .5 * np.real(np.cross(e, np.conj(h)))
