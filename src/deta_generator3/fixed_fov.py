"""Zero-Bloch complex-field reconstruction from the project numerical operator."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import json

def periodic_complex_sample(field,x_nm,y_nm,pixel_centers_nm):
    field=np.asarray(field)
    x,y,pixels=[np.asarray(v,dtype=float) for v in (x_nm,y_nm,pixel_centers_nm)]
    if any(v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() or np.any(np.diff(v)<=0) for v in (x,y,pixels)):
        raise ValueError('Need finite ascending physical coordinate vectors')
    if field.shape[:2]!=(len(y),len(x)) or field.shape[-1]!=3 or not np.iscomplexobj(field) or not np.isfinite(field).all():
        raise ValueError('Need native co-registered complex vector fields [y,x,...,3]')
    periods=np.array([x[-1]-x[0],y[-1]-y[0]])
    if not np.isclose(*periods,rtol=1e-8,atol=1e-6):
        raise ValueError('This diagnostic requires square periodic native cells')
    scale=max(float(np.abs(field).max()),1e-12)
    seam=max(float(np.abs(field[:,0]-field[:,-1]).max()),float(np.abs(field[0]-field[-1]).max()))/scale
    if seam>1e-5:
        raise ValueError('Native periodic boundaries disagree; do not create artificial tile seams')
    px=(pixels-x[0])%periods[0]+x[0]
    py=(pixels-y[0])%periods[1]+y[0]
    yy,xx=np.meshgrid(py,px,indexing='ij')
    result=RegularGridInterpolator((y,x),field,bounds_error=True)(np.stack((yy,xx),-1))
    return result,dict(native_period_nm=periods.tolist(),relative_boundary_max_error=seam)

def render_fixed_observations(native,settings):
    """Forward sampling only: no geometry label is accepted by this operator."""
    from .vision import hotspots,poynting
    from .materials import C0
    if settings.field_xy_span_nm is None:
        raise ValueError('Fixed physical observation span must be explicit')
    pixels=((np.arange(settings.image_size)+.5)/settings.image_size-.5)*settings.field_xy_span_nm
    electric=[];magnetic=[];seams=[]
    for plane in range(len(settings.field_z_nm)):
        pairs=[]
        for key in ('E','H'):
            sample,check=periodic_complex_sample(native[f'native_{key}_{plane}'],
                native[f'native_x_nm_{plane}'],native[f'native_y_nm_{plane}'],pixels)
            if sample.shape!=(settings.image_size,settings.image_size,len(settings.field_wavelengths_nm),3):
                raise ValueError('Native observation frequencies or component shapes differ')
            pairs.append(sample.transpose(2,0,1,3).astype(np.complex64));seams.append(check)
        electric.append(pairs[0]);magnetic.append(pairs[1])
    electric,magnetic=np.asarray(electric),np.asarray(magnetic)
    intensity=np.sum(np.abs(electric)**2,-1).astype(np.float32)
    flux=poynting(electric,magnetic).astype(np.float32)
    masks=[];labels=[];features=[]
    for plane,z in enumerate(settings.field_z_nm):
        pm=[];pl=[];pf=[]
        for wave,image in zip(settings.field_wavelengths_nm,intensity[plane]):
            mask,label,stats=hotspots(image,settings.hotspot_seed_fraction,settings.hotspot_grow_fraction,
                settings.hotspot_min_enhancement,periodic_xy=False)
            stats.update(z_nm=z,wavelength_nm=wave)
            pm.append(mask);pl.append(label);pf.append(stats)
        masks.append(pm);labels.append(pl);features.append(pf)
    return dict(electric=electric,magnetic=magnetic,intensity=intensity,
        poynting_w_m2=flux,poynting_incident_normalized=flux/(.5*settings.background_index*8.8541878128e-12*C0),
        hotspot_mask=np.asarray(masks,dtype=np.uint8),hotspot_labels=np.asarray(labels,dtype=np.int32),
        hotspot_features_json=json.dumps(features,allow_nan=False),pixel_x_nm=pixels,pixel_y_nm=pixels),seams
