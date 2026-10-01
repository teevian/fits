import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate, shift
from streamlit.testing.v1 import AppTest

from reduction import frame, to_fits
from workflow import align_frame, run_workflow


def image(values, name='image.fits', **header):
    return frame(values, fits.Header(header), name=name)


def test_workflow_calibrates_flats_before_science_then_subtracts_background():
    y, x = np.indices((40, 48))
    response = 0.7 + x / 80 + y / 120
    signal = 80 * np.exp(-((x-22)**2 + (y-17)**2) / 8)
    raw = image(100 + response * (25 + signal), FILTER='Blue')
    biases = [image(np.full(raw.data.shape, 100), name=f'bias{i}.fits', IMAGETYP='BIAS')
              for i in range(2)]
    flats = [image(100 + level * response, name=f'flat{i}.fits',
                   IMAGETYP='FLAT', FILTER='Blue') for i, level in enumerate((1000, 2000))]
    result = run_workflow(raw, bias_sources=biases, flat_sources=flats, background=True)
    assert result.header['BIASCOR'] and result.header['FLATCOR'] and result.header['BKGDSUB']
    np.testing.assert_allclose(result.data[17, 22], signal[17, 22] * np.median(response), atol=0.5)
    assert abs(np.median(result.data)) < 0.1
    np.testing.assert_array_equal(raw.data, 100 + response * (25 + signal))


def test_workflow_aligns_and_combines_science_frames():
    y, x = np.indices((96, 128))
    stars = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / 5)
                for cy, cx, peak in [(24, 30, 100), (66, 82, 60), (45, 104, 80)])
    first = image(stars, FILTER='Blue')
    second = image(shift(stars, (4, -6), order=1, mode='constant', cval=np.nan),
                   name='second.fits', FILTER='Blue')
    result = run_workflow(first, [second], alignment='Translation', max_shift=20,
                          rejection_sigma=None)
    assert result.header['NCOMBINE'] == 2
    assert result.header['ALIGN'] == 'TRANSLATION'
    np.testing.assert_allclose([result.header['ALN001Y'], result.header['ALN001X']],
                               [-4, 6], atol=0.2)
    assert np.mean((result.data[12:-12, 12:-12] - stars[12:-12, 12:-12])**2) < 0.1


def test_rotation_alignment_handles_small_camera_rotation():
    y, x = np.indices((128, 128))
    stars = sum(peak * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*width**2))
                for cy, cx, peak, width in [(21, 30, 100, 1.5), (96, 45, 80, 2),
                                            (67, 103, 70, 1.3), (31, 91, 50, 1.7)])
    first = image(stars, FILTER='Blue')
    turned = rotate(stars, 2, reshape=False, order=1, mode='constant', cval=np.nan)
    second = image(shift(turned, (3, -4), order=1, mode='constant', cval=np.nan),
                   FILTER='Blue')
    aligned, (angle, dy, dx) = align_frame(first, second, max_shift=20, max_angle=5)
    assert abs(angle + 2) < 0.6
    assert abs(dy + 3) < 1 and abs(dx - 4) < 1
    valid = ~aligned.mask & ~first.mask
    assert np.mean((aligned.data[valid] - first.data[valid])**2) < 2


def test_workflow_ui_runs_selected_steps_once(tmp_path):
    for name, value, kind in [('science.fits', 120, 'OBJECT'),
                              ('bias1.fits', 10, 'BIAS'), ('bias2.fits', 10, 'BIAS')]:
        (tmp_path / name).write_bytes(to_fits(image(np.full((12, 12), value),
                                               name=name, IMAGETYP=kind)))
    source = f'''
from pathlib import Path
import streamlit as st
from reduction import read_frame
from workflow_ui import workflow_panel
folder = Path({str(tmp_path)!r})
result = workflow_panel(read_frame(folder / 'science.fits'), 'source', folder)
st.session_state.preview = result.name if result is not None else None
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not app.exception
    app.selectbox(key='workflow_bias_mode').select('Create master from frames').run()
    app.multiselect(key='workflow_bias_frames').set_value(['bias1.fits', 'bias2.fits']).run()
    app.button(key='workflow_run').click().run()
    assert not app.exception and not app.error
    assert app.session_state['preview'] == 'science_workflow.fits'
    np.testing.assert_allclose(app.session_state['workflow_last'][1].data, 110)
    app.checkbox(key='workflow_background').set_value(True).run()
    assert app.session_state['preview'] is None
