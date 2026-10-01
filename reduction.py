"""CCD reduction on 2-D FITS images, using NumPy and Astropy.

Masks use True for excluded pixels. No input file or input array is modified.
"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import tempfile
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy import units as u
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from scipy.optimize import minimize


@dataclass
class Frame:
    data: np.ndarray
    header: fits.Header
    mask: np.ndarray
    name: str = 'image.fits'

    def display(self):
        return np.where(self.mask, np.nan, self.data)


def clean_header(header):
    header = header.copy()
    for key in ('BSCALE', 'BZERO', 'BLANK', 'DATAMIN', 'DATAMAX', 'CHECKSUM', 'DATASUM'):
        header.remove(key, remove_all=True, ignore_missing=True)
    return header


def frame(data, header=None, mask=None, name='image.fits'):
    values = np.array(data, dtype=float, copy=True)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError('Processing requires a non-empty 2D image.')
    bad = ~np.isfinite(values)
    if mask is not None:
        if np.shape(mask) != values.shape:
            raise ValueError('Mask dimensions do not match the image.')
        bad |= np.asarray(mask, dtype=bool)
    return Frame(values, clean_header(header if header is not None else fits.Header()), bad, name)


def read_frame(source):
    if isinstance(source, Frame):
        return frame(source.data, source.header, source.mask, source.name)
    name = Path(source).name if isinstance(source, (str, Path)) else 'carregado.fits'
    with fits.open(BytesIO(source) if isinstance(source, bytes) else source, memmap=False) as hdus:
        candidates = [h for h in hdus if isinstance(h, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                      and h.header.get('NAXIS') == 2 and h.name.upper() != 'MASK']
        if not candidates:
            raise ValueError(f'{name}: does not contain a 2D image.')
        hdu = candidates[0]
        header = hdu.header.copy()
        mask = hdus['MASK'].data if 'MASK' in hdus else None
        return frame(hdu.data, header, mask, name)


def add_history(header, text):
    # FITS headers are ASCII; names can contain Portuguese characters.
    header.add_history(text.encode('ascii', 'replace').decode('ascii'))


def compatible(reference, other, check_filter=False, check_unit=True):
    if reference.data.shape != other.data.shape:
        raise ValueError(f'{other.name}: dimensions {other.data.shape} differ from {reference.data.shape}.')
    for key in ('XBINNING', 'YBINNING', 'GAIN', 'INSTRUME'):
        a, b = reference.header.get(key), other.header.get(key)
        if a is not None and b is not None and a != b:
            raise ValueError(f'{other.name}: {key} incompatible ({b} / {a}).')
    if check_filter:
        a, b = reference.header.get('FILTER'), other.header.get('FILTER')
        if a is not None and b is not None and str(a).strip().casefold() != str(b).strip().casefold():
            raise ValueError(f'{other.name}: FILTER {b} differs from {a}.')
    if check_unit:
        # Missing BUNIT is conventional ADU for the local camera files.
        def unit(header):
            value = str(header.get('BUNIT', 'adu')).strip()
            return u.Unit('adu' if value.casefold() == 'adu' else value)
        a, b = unit(reference.header), unit(other.header)
        if a != b:
            raise ValueError(f'{other.name}: incompatible units ({b} / {a}).')


def exposure(image):
    value = image.header.get('EXPTIME', image.header.get('EXPOSURE'))
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{image.name}: falta EXPTIME/EXPOSURE em segundos.') from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{image.name}: exposure must be positive.')
    return value


def _registration_array(image):
    """Create a background-suppressed, tapered image for phase correlation."""
    values = np.asarray(image.data, dtype=float)
    valid = ~image.mask & np.isfinite(values)
    if valid.sum() < 16:
        raise ValueError(f'{image.name}: too few valid pixels for alignment.')
    sample = values[valid]
    # Keep sparse star fields useful while limiting isolated detector outliers.
    low, high = np.percentile(sample, [0.01, 99.99])
    filled = np.where(valid, np.clip(values, low, high), np.median(sample))
    sigma = max(2., min(values.shape) / 128.)
    prepared = filled - gaussian_filter(filled, sigma=sigma, mode='nearest')
    prepared[~valid] = 0.
    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    prepared *= window
    norm = np.linalg.norm(prepared)
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError(f'{image.name}: sem estrutura suficiente para estimar o alinhamento.')
    return prepared / norm


def estimate_translation(reference, moving, max_shift=100, subpixel=True):
    """Estimate the (dy, dx) shift to apply to moving via phase correlation."""
    reference = read_frame(reference)
    moving = read_frame(moving)
    compatible(reference, moving, check_filter=True)
    limit = min(reference.data.shape) // 2 - 2
    if not isinstance(max_shift, (int, np.integer)) or not 1 <= max_shift <= limit:
        raise ValueError(f'Maximum shift must be between 1 and {limit} pixels.')
    a, b = _registration_array(reference), _registration_array(moving)
    product = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    magnitude = np.abs(product)
    product /= np.where(magnitude > np.finfo(float).eps, magnitude, 1.)
    correlation = np.fft.fftshift(np.abs(np.fft.ifft2(product)))
    cy, cx = np.array(correlation.shape) // 2
    area = correlation[cy-max_shift:cy+max_shift+1, cx-max_shift:cx+max_shift+1]
    py, px = np.unravel_index(np.argmax(area), area.shape)
    py, px = py + cy - max_shift, px + cx - max_shift
    dy, dx = float(py - cy), float(px - cx)
    if abs(dy) == max_shift or abs(dx) == max_shift:
        raise ValueError(f'{moving.name}: shift reached the {max_shift}-pixel limit; increase it and retry.')
    if subpixel:
        def offset(before, center, after):
            denominator = before - 2*center + after
            value = 0.5 * (before - after) / denominator if abs(denominator) > 1e-15 else 0.
            return float(np.clip(value, -0.5, 0.5))
        dy += offset(correlation[py-1, px], correlation[py, px], correlation[py+1, px])
        dx += offset(correlation[py, px-1], correlation[py, px], correlation[py, px+1])
        # The parabolic peak gives a good initial value. Refine it by maximizing
        # normalized correlation after bilinear translation.
        margin = min(max_shift + 2, min(a.shape) // 4)
        if min(a.shape) - 2 * margin >= 16:
            reference_crop = a[margin:-margin, margin:-margin]
            reference_power = np.sum(reference_crop * reference_crop)

            def objective(candidate):
                shifted = ndi_shift(b, candidate, order=1, mode='constant', cval=0., prefilter=False)
                shifted = shifted[margin:-margin, margin:-margin]
                denominator = np.sqrt(reference_power * np.sum(shifted * shifted))
                return -float(np.sum(reference_crop * shifted) / denominator) if denominator else 0.

            solution = minimize(objective, (dy, dx), method='Powell',
                                bounds=((dy - .75, dy + .75), (dx - .75, dx + .75)),
                                options={'xtol': .01, 'ftol': 1e-7, 'maxiter': 20})
            if solution.success and np.all(np.isfinite(solution.x)):
                dy, dx = (float(value) for value in solution.x)
    else:
        dy, dx = float(round(dy)), float(round(dx))
    return dy, dx


def translate_frame(image, shift, subpixel=True):
    """Resample an image by (dy, dx), masking uncovered and partial pixels."""
    image = read_frame(image)
    dy, dx = (float(value) for value in shift)
    if not subpixel:
        dy, dx = float(round(dy)), float(round(dx))
    order = 1 if subpixel else 0
    valid = (~image.mask & np.isfinite(image.data)).astype(float)
    numerator = ndi_shift(np.where(valid > 0, image.data, 0.), (dy, dx), order=order,
                          mode='constant', cval=0., prefilter=False)
    weight = ndi_shift(valid, (dy, dx), order=order, mode='constant', cval=0., prefilter=False)
    data = np.divide(numerator, weight, out=np.full_like(numerator, np.nan), where=weight > 1e-12)
    mask = weight < 1 - 1e-7
    result = frame(data, image.header, mask, image.name)
    add_history(result.header, f'Image translated dy={dy:.4f}; dx={dx:.4f}; interpolation order={order}')
    return result


def calibrate(image, bias=None, dark=None, flat=None, mask=None,
              scale_dark=False, dark_contains_bias=False, flat_min=0.0):
    result = read_frame(image)
    if dark is not None and dark_contains_bias and (bias is not None or scale_dark):
        raise ValueError('A dark containing bias requires equal exposure and cannot be used with separate bias subtraction.')
    if bias is not None:
        if result.header.get('BIASCOR') is True:
            raise ValueError('This image already has BIASCOR; do not subtract bias again.')
        compatible(result, bias)
        result.data -= bias.data
        result.mask |= bias.mask
        result.header['BIASCOR'] = True
        add_history(result.header, f'Bias subtracted: {bias.name}')
    if dark is not None:
        if result.header.get('DARKCOR') is True:
            raise ValueError('This image already has DARKCOR; do not subtract dark again.')
        if dark_contains_bias and result.header.get('BIASCOR') is True:
            raise ValueError('Bias has already been subtracted; use a bias-free dark.')
        compatible(result, dark)
        science_time, dark_time = exposure(result), exposure(dark)
        if not scale_dark and not np.isclose(science_time, dark_time, rtol=1e-6, atol=1e-6):
            raise ValueError('Dark and image exposures differ. Use a matching dark or enable exposure scaling for a bias-free dark.')
        factor = science_time / dark_time if scale_dark else 1.0
        result.data -= dark.data * factor
        result.mask |= dark.mask
        result.header['DARKCOR'] = True
        if dark_contains_bias:
            result.header['BIASCOR'] = True
        add_history(result.header, f'Dark subtracted: {dark.name}; factor={factor}; includes bias={dark_contains_bias}')
    if flat is not None:
        if result.header.get('FLATCOR') is True:
            raise ValueError('This image already has FLATCOR; do not apply another flat.')
        compatible(result, flat, check_filter=True, check_unit=False)
        if not np.isfinite(flat_min) or flat_min < 0:
            raise ValueError('Flat threshold must be finite and nonnegative.')
        valid = ~flat.mask & np.isfinite(flat.data) & (flat.data > 0)
        if not np.any(valid):
            raise ValueError('The flat has no valid positive values.')
        norm = float(np.median(flat.data[valid]))
        response = flat.data / norm
        invalid = ~valid | (response <= flat_min)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            result.data = np.divide(result.data, response, out=np.full_like(result.data, np.nan), where=~invalid)
        result.mask |= invalid
        result.header['FLATCOR'] = True
        add_history(result.header, f'Flat divided: {flat.name}; normalization median={norm}; min response={flat_min}')
    if mask is not None:
        if mask.shape != result.data.shape:
            raise ValueError('Mask dimensions do not match the image.')
        result.mask |= np.asarray(mask, dtype=bool)
        add_history(result.header, 'External bad-pixel mask applied')
    result.mask |= ~np.isfinite(result.data)
    return result


def combine(sources, method='mean', sigma=5., normalize=False, kind='combined',
            bias=None, dark=None, scale_dark=False, dark_contains_bias=False, progress=None,
            align=False, max_shift=100, subpixel=True):
    if len(sources) < 2:
        raise ValueError('Select at least two images.')
    if method not in ('mean', 'median') or (sigma is not None and (not np.isfinite(sigma) or sigma <= 0)):
        raise ValueError('Invalid method or sigma threshold.')
    if kind not in ('combined', 'bias', 'flat'):
        raise ValueError('Invalid combination type.')
    if align and kind != 'combined':
        raise ValueError('Alignment is available only for Combine images.')
    normalize = normalize or kind == 'flat'
    first = read_frame(sources[0])
    height, width = first.data.shape
    headers = []
    history = []
    alignment_offsets = []
    # Disk-backed stack avoids keeping every full image in RAM. Sigma clipping
    # operates on bounded row blocks across the exposure axis, never spatially.
    with tempfile.TemporaryDirectory(prefix='fits-combine-') as temp:
        stack = np.memmap(Path(temp) / 'stack.dat', mode='w+', dtype='float64',
                          shape=(len(sources), height, width))
        try:
            for i, source in enumerate(sources):
                image = read_frame(source)
                compatible(first, image, check_filter=kind != 'bias')
                image_type = str(image.header.get('IMAGETYP', '')).upper()
                if kind == 'bias' and image_type and 'BIAS' not in image_type and 'ZERO' not in image_type:
                    raise ValueError(f'{image.name}: IMAGETYP does not identify a bias frame.')
                if kind == 'flat' and image_type and 'FLAT' not in image_type:
                    raise ValueError(f'{image.name}: IMAGETYP does not identify a flat frame.')
                headers.append(image.header.copy())
                if kind == 'flat':
                    image = calibrate(image, bias=bias, dark=dark, scale_dark=scale_dark,
                                      dark_contains_bias=dark_contains_bias)
                dy, dx = 0., 0.
                if align and i:
                    dy, dx = estimate_translation(first, image, max_shift=max_shift, subpixel=subpixel)
                    image = translate_frame(image, (dy, dx), subpixel=subpixel)
                values = image.display()
                if normalize:
                    valid = values[np.isfinite(values)]
                    median = float(np.median(valid)) if valid.size else np.nan
                    if not np.isfinite(median) or median <= 0:
                        raise ValueError(f'{image.name}: median is not positive; cannot normalize.')
                    values /= median
                stack[i] = values
                history.append(f'Input {i + 1}: {image.name}; alignment dy={dy:.4f} dx={dx:.4f}')
                alignment_offsets.append((dy, dx))
                if progress:
                    progress((i + 1) / len(sources) * 0.5)
            output = np.empty((height, width), dtype=float)
            rows = max(1, min(height, int(64e6 / (len(sources) * width * 8 * 5))))
            for start in range(0, height, rows):
                block = np.ma.masked_invalid(np.array(stack[:, start:start + rows]))
                if sigma is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        block = sigma_clip(block, sigma=sigma, axis=0, cenfunc='median',
                                           stdfunc='mad_std', maxiters=5, masked=True)
                reduced = np.ma.mean(block, axis=0) if method == 'mean' else np.ma.median(block, axis=0)
                output[start:start + rows] = reduced.filled(np.nan)
                if progress:
                    progress(0.5 + min(start + rows, height) / height * 0.5)
        finally:
            del stack
    header = clean_header(first.header)
    header['COMBINED'] = True
    header['NCOMBINE'] = len(sources)
    header['COMBMETH'] = method
    if align:
        header['ALIGN'] = ('TRANSLATION', 'Registration before combination')
        header['ALNSUBPX'] = (bool(subpixel), 'Subpixel interpolation used')
        header['ALNMAX'] = (int(max_shift), 'Maximum allowed shift in pixels')
        # Copy shift diagnostics collected on the temporary frame headers.
        for i, (dy, dx) in enumerate(alignment_offsets[:1000]):
            header[f'ALN{i:03d}Y'] = (dy, 'Applied row shift')
            header[f'ALN{i:03d}X'] = (dx, 'Applied column shift')
    if normalize:
        header['BUNIT'] = '1'
        header['NORMALIZ'] = True
    if kind == 'bias':
        header['IMAGETYP'] = 'MASTER_BIAS'
        header['EXPTIME'] = 0.
        header['EXPOSURE'] = 0.
        header.remove('FILTER', ignore_missing=True)
    if kind == 'flat':
        valid = np.isfinite(output) & (output > 0)
        if not valid.any():
            raise ValueError('Master flat has no positive response.')
        output /= np.median(output[valid])
        output[~valid] = np.nan
        header['IMAGETYP'] = 'MASTER_FLAT'
        header['BIASCOR'] = bias is not None or (dark is not None and dark_contains_bias) or all(h.get('BIASCOR') is True for h in headers)
        header['DARKCOR'] = dark is not None or all(h.get('DARKCOR') is True for h in headers)
    if kind == 'flat' or any(h.get('EXPTIME', h.get('EXPOSURE')) !=
                             headers[0].get('EXPTIME', headers[0].get('EXPOSURE')) for h in headers):
        for key in ('EXPTIME', 'EXPOSURE'):
            header.remove(key, ignore_missing=True)
    add_history(header, f'Combine method={method}; sigma={sigma}; iterations=5; median normalization={normalize}; align={align}')
    for line in history:
        add_history(header, line)
    if bias is not None:
        add_history(header, f'Flat calibration bias: {bias.name}')
    if dark is not None:
        add_history(header, f'Flat calibration dark: {dark.name}; scale={scale_dark}; includes bias={dark_contains_bias}')
    return frame(output, header, name=f'{"master_" if kind != "combined" else ""}{kind}.fits')


def mask_image(image, lower=None, upper=None, rectangle=None, external=None):
    result = read_frame(image)
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError('Lower limit must be less than upper limit.')
    if lower is not None:
        result.mask |= result.data < lower
    if upper is not None:
        result.mask |= result.data > upper
    if rectangle is not None:
        x0, x1, y0, y1 = rectangle
        if not (0 <= x0 < x1 <= result.data.shape[1] and 0 <= y0 < y1 <= result.data.shape[0]):
            raise ValueError('Invalid mask region.')
        result.mask[y0:y1, x0:x1] = True
    if external is not None:
        if external.shape != result.data.shape:
            raise ValueError('External mask dimensions differ.')
        result.mask |= np.asarray(external, dtype=bool)
    result.header['MASKED'] = True
    add_history(result.header, f'Mask: lower={lower}; upper={upper}; rectangle={rectangle}; external={external is not None}')
    return result


def read_mask(source):
    if isinstance(source, Frame):
        return source.mask.copy()
    with fits.open(BytesIO(source) if isinstance(source, bytes) else source, memmap=False) as hdus:
        if 'MASK' in hdus:
            data = hdus['MASK'].data
        else:
            images = [h.data for h in hdus if h.header.get('NAXIS') == 2 and h.data is not None]
            if not images:
                raise ValueError('File does not contain a 2D mask.')
            data = images[0]
        if np.ndim(data) != 2:
            raise ValueError('Mask must be 2D.')
        return (np.asarray(data) != 0) | ~np.isfinite(data)


def to_fits(image):
    header = clean_header(image.header)
    header['NMASKED'] = int(image.mask.sum())
    hdus = fits.HDUList([fits.PrimaryHDU(image.data, header=header),
                         fits.ImageHDU(image.mask.astype('uint8'), name='MASK')])
    hdus[1].header['MASKTRUE'] = '1 = excluded pixel'
    buffer = BytesIO()
    hdus.writeto(buffer, checksum=True)
    return buffer.getvalue()
