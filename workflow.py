"""Run the CCD steps in order, without changing any source frame."""
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clip
from scipy.ndimage import rotate as ndi_rotate, shift as ndi_shift

from reduction import (add_history, calibrate, combine, estimate_translation,
                       frame, read_frame, translate_frame, _registration_array)


def subtract_background(image, sigma=3.0):
    result = read_frame(image)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Background sigma must be positive.')
    valid = result.data[~result.mask & np.isfinite(result.data)]
    if not valid.size:
        raise ValueError(f'{result.name}: no valid pixels for background estimation.')
    clipped = sigma_clip(valid, sigma=sigma, maxiters=5)
    level = float(np.ma.median(clipped))
    if not np.isfinite(level):
        raise ValueError(f'{result.name}: background estimation failed.')
    result.data -= level
    result.header['BKGDSUB'] = (True, 'Global sky background subtracted')
    result.header['BKGDLEV'] = (level, 'Subtracted sky level')
    add_history(result.header, f'Global sigma-clipped sky median subtracted: {level:.8g}')
    return result


def rotate_frame(image, angle):
    """Rotate around the image center, masking uncovered or partly valid pixels."""
    image = read_frame(image)
    valid = (~image.mask & np.isfinite(image.data)).astype(float)
    numerator = ndi_rotate(np.where(valid > 0, image.data, 0.), angle, reshape=False,
                           order=1, mode='constant', cval=0., prefilter=False)
    weight = ndi_rotate(valid, angle, reshape=False, order=1, mode='constant',
                        cval=0., prefilter=False)
    values = np.divide(numerator, weight, out=np.full_like(numerator, np.nan), where=weight > 1e-12)
    result = frame(values, image.header, weight < 1 - 1e-7, image.name)
    add_history(result.header, f'Image rotated about center by {angle:.4f} degrees')
    return result


def align_frame(reference, moving, max_shift=100, max_angle=0.0):
    """Find a small rotation, then the X/Y translation to the reference."""
    reference, moving = read_frame(reference), read_frame(moving)
    if not np.isfinite(max_angle) or max_angle < 0 or max_angle > 15:
        raise ValueError('Maximum rotation must be between 0 and 15 degrees.')
    angle = 0.0
    if max_angle:
        step = max(1, int(np.ceil(max(reference.data.shape) / 384)))
        ref_small = frame(reference.data[::step, ::step], reference.header,
                          reference.mask[::step, ::step], reference.name)
        mov_small = frame(moving.data[::step, ::step], moving.header,
                          moving.mask[::step, ::step], moving.name)
        limit = min(ref_small.data.shape) // 2 - 2
        small_shift = min(limit, max(1, int(np.ceil(max_shift / step))))
        if limit < 1:
            raise ValueError('Image is too small for rotational alignment.')
        prepared_reference = _registration_array(ref_small)

        def score(candidate):
            try:
                rotated = rotate_frame(mov_small, candidate)
                dy, dx = estimate_translation(ref_small, rotated, small_shift, False)
                prepared = _registration_array(rotated)
                shifted = ndi_shift(prepared, (dy, dx), order=1, mode='constant', cval=0.)
                return float(np.sum(prepared_reference * shifted))
            except ValueError:
                return -np.inf

        coarse = np.linspace(-max_angle, max_angle, 9)
        angle = float(max(coarse, key=score))
        spacing = coarse[1] - coarse[0]
        fine = np.linspace(max(-max_angle, angle-spacing),
                           min(max_angle, angle+spacing), 7)
        angle = float(max(fine, key=score))
        if not np.isfinite(score(angle)):
            raise ValueError(f'{moving.name}: cannot determine rotational alignment.')
    rotated = rotate_frame(moving, angle) if angle else moving
    dy, dx = estimate_translation(reference, rotated, max_shift, True)
    aligned = translate_frame(rotated, (dy, dx))
    return aligned, (angle, dy, dx)


def run_workflow(current, others=(), *, bias_sources=(), bias_master=None,
                 flat_sources=(), flat_master=None, background=False,
                 background_sigma=3.0, alignment='None', max_shift=100,
                 max_angle=5.0, method='median', rejection_sigma=5.0):
    """Build masters, calibrate each science frame, subtract sky, align, combine."""
    if bias_sources and bias_master is not None or flat_sources and flat_master is not None:
        raise ValueError('Choose either source frames or an existing master for each calibration.')
    if alignment not in ('None', 'Translation', 'Translation + rotation'):
        raise ValueError('Invalid alignment mode.')
    master_bias = combine(bias_sources, kind='bias') if bias_sources else (
        read_frame(bias_master) if bias_master is not None else None)
    master_flat = combine(flat_sources, kind='flat', bias=master_bias) if flat_sources else (
        read_frame(flat_master) if flat_master is not None else None)
    sources = [current, *others]
    calibrated = []
    for source in sources:
        image = calibrate(read_frame(source), bias=master_bias, flat=master_flat,
                          flat_min=0.05)
        if background:
            image = subtract_background(image, background_sigma)
        calibrated.append(image)
    offsets = [(0.0, 0.0, 0.0)]
    if len(calibrated) > 1 and alignment != 'None':
        for index in range(1, len(calibrated)):
            calibrated[index], offset = align_frame(
                calibrated[0], calibrated[index], max_shift,
                max_angle if alignment == 'Translation + rotation' else 0.0)
            offsets.append(offset)
    result = (combine(calibrated, method=method, sigma=rejection_sigma)
              if len(calibrated) > 1 else calibrated[0])
    if len(calibrated) > 1 and alignment != 'None':
        result.header['ALIGN'] = ('ROTATION' if alignment == 'Translation + rotation'
                                  else 'TRANSLATION', 'Registration before combination')
        for index, (angle, dy, dx) in enumerate(offsets[:100]):
            result.header[f'ALN{index:03d}R'] = (angle, 'Applied rotation in degrees')
            result.header[f'ALN{index:03d}Y'] = (dy, 'Applied row shift')
            result.header[f'ALN{index:03d}X'] = (dx, 'Applied column shift')
    result.name = Path(read_frame(current).name).stem + '_workflow.fits'
    add_history(result.header, 'Workflow: bias -> flat -> background -> alignment -> combination')
    return result
