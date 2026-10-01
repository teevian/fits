from pathlib import Path

import numpy as np
from astropy.io import fits
from streamlit.testing.v1 import AppTest


def test_viewer_tools_and_direct_export(tmp_path):
    fits.writeto(tmp_path / 'image.fits', np.arange(48.).reshape(6, 8))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.session_state['folder_path'] = str(tmp_path)
    app.run()
    assert not app.exception
    assert app.selectbox(key='processing_operation').value == 'None'
    assert not app.radio
    labels = [tab.label for tab in app.tabs]
    assert all(label in labels for label in ['File', 'Processing', 'Workflow', 'Statistics', 'Header',
                                               'Crop / adjust', 'Histogram', 'Profiles'])
    assert labels[:3] == ['Crop / adjust', 'Histogram', 'Profiles']
    assert not any(button.label in ('Download statistics CSV', 'Download header')
                   for button in app.get('download_button'))
    assert any(button.label == 'Download FITS' for button in app.get('download_button'))
    assert not any(button.label == 'Prepare FITS' for button in app.button)
