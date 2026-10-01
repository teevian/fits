"""FITS operations independent of the UI; never modify original files."""
from io import BytesIO
import numpy as np
from astropy.io import fits


def statistics(data):
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {}
    return dict(pixeis=int(values.size), minimo=float(values.min()),
                maximo=float(values.max()), media=float(values.mean()),
                mediana=float(np.median(values)), desvio=float(values.std()),
                soma=float(values.sum()))


def display_values(data, low, high, scale):
    normalized = np.clip((np.asarray(data, dtype=float) - low) / max(high-low, 1e-30), 0, 1)
    if scale == 'Logarithmic':
        return np.log1p(1000 * normalized) / np.log1p(1000)
    if scale == 'Asinh':
        return np.arcsinh(10 * normalized) / np.arcsinh(10)
    if scale == 'Square root':
        return np.sqrt(normalized)
    return normalized


def export_crop(data, header, x0, x1, y0, y1, offset=0., gain=1., mask=None):
    """Recorta os dois eixos espaciais, preservando os restantes eixos e WCS."""
    if not (0 <= x0 < x1 <= data.shape[-1] and 0 <= y0 < y1 <= data.shape[-2]):
        raise ValueError('Invalid crop bounds.')
    result = (np.array(data[..., y0:y1, x0:x1], dtype=float) - offset) * gain
    hdr = header.copy()
    for key in list(hdr):
        if key in ('BSCALE', 'BZERO', 'BLANK', 'DATAMIN', 'DATAMAX', 'CHECKSUM', 'DATASUM'):
            del hdr[key]
    # Include alternate WCS versions (CRPIX1A, CRPIX2A, ...).
    for key in list(hdr):
        if key.startswith('CRPIX1'):
            hdr[key] -= x0
        elif key.startswith('CRPIX2'):
            hdr[key] -= y0
    hdr.add_history(f'Crop x=[{x0},{x1}), y=[{y0},{y1}); (data - {offset}) * {gain}')
    buffer = BytesIO()
    output = [fits.PrimaryHDU(result, header=hdr)]
    if mask is not None:
        if np.shape(mask) != np.shape(data):
            raise ValueError('Mask dimensions do not match the image.')
        cropped_mask = np.asarray(mask, dtype=bool)[..., y0:y1, x0:x1] | ~np.isfinite(result)
        output[0].header['NMASKED'] = int(cropped_mask.sum())
        output.append(fits.ImageHDU(cropped_mask.astype('uint8'), name='MASK'))
    fits.HDUList(output).writeto(buffer, checksum=True)
    return buffer.getvalue()
