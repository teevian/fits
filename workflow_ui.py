"""One-run sidebar controls for the sequential CCD workflow."""
from pathlib import Path

import streamlit as st

from processing_ui import fingerprint, list_fits, load, select_input
from reduction import to_fits
from workflow import run_workflow


def workflow_panel(current, source_id, folder, on_view_change=None):
    catalog = {}
    try:
        catalog.update({path.name: path for path in list_fits(folder)})
    except OSError:
        pass
    catalog.update({f'✓ {name}': value for name, value in
                    st.session_state.get('reduction_products', {}).items()})

    st.markdown('**1. Bias**')
    bias_mode = st.selectbox('Bias source', ['None', 'Create master from frames',
                                             'Use master bias'], key='workflow_bias_mode',
                             label_visibility='collapsed')
    bias_sources = select_input('Bias frames', 'workflow_bias_frames', catalog, multiple=True) if (
        bias_mode == 'Create master from frames') else []
    bias_master = select_input('Master bias', 'workflow_bias_master', catalog) if (
        bias_mode == 'Use master bias') else None

    st.markdown('**2. Flats**')
    flat_mode = st.selectbox('Flat source', ['None', 'Create master from frames',
                                             'Use master flat'], key='workflow_flat_mode',
                             label_visibility='collapsed')
    flat_sources = select_input('Flat frames', 'workflow_flat_frames', catalog, multiple=True) if (
        flat_mode == 'Create master from frames') else []
    flat_master = select_input('Master flat', 'workflow_flat_master', catalog) if (
        flat_mode == 'Use master flat') else None

    st.markdown('**3. Background**')
    background = st.checkbox('Subtract global sky median', key='workflow_background')
    background_sigma = st.number_input('Clipping sigma', 1.0, 10.0, 3.0, 0.5,
                                        key='workflow_background_sigma') if background else 3.0

    st.markdown('**4. Alignment**')
    alignment = st.selectbox('Align frames', ['None', 'Translation', 'Translation + rotation'],
                              key='workflow_alignment', label_visibility='collapsed')
    max_shift = min(100, max(1, min(current.data.shape) // 2 - 2))
    max_shift = st.number_input('Maximum shift (pixels)', 1,
                                 max(1, min(current.data.shape) // 2 - 2), max_shift,
                                 key='workflow_max_shift') if alignment != 'None' else max_shift
    max_angle = st.number_input('Maximum rotation (degrees)', 0.5, 15.0, 5.0, 0.5,
                                 key='workflow_max_angle') if alignment == 'Translation + rotation' else 0.0

    st.markdown('**5. Combination**')
    others = select_input('Additional science frames', 'workflow_science', catalog, multiple=True)
    method = st.selectbox('Method', ['Median', 'Mean'], key='workflow_method')
    reject = st.checkbox('Reject outliers', value=True, key='workflow_reject')
    rejection_sigma = st.number_input('Rejection sigma', 1.0, 20.0, 5.0, 0.5,
                                       key='workflow_rejection_sigma') if reject else None

    config = (source_id, bias_mode, tuple(map(fingerprint, bias_sources)), fingerprint(bias_master),
              flat_mode, tuple(map(fingerprint, flat_sources)), fingerprint(flat_master),
              background, background_sigma, alignment, max_shift, max_angle,
              tuple(map(fingerprint, others)), method, rejection_sigma)
    signature = repr(config)
    if st.button('Run workflow', key='workflow_run'):
        try:
            source = lambda value: load(value) if isinstance(value, tuple) else value
            with st.spinner('Running workflow…'):
                result = run_workflow(current, [source(value) for value in others],
                                      bias_sources=[source(value) for value in bias_sources],
                                      bias_master=source(bias_master) if bias_master else None,
                                      flat_sources=[source(value) for value in flat_sources],
                                      flat_master=source(flat_master) if flat_master else None,
                                      background=background,
                                      background_sigma=background_sigma,
                                      alignment=alignment, max_shift=max_shift,
                                      max_angle=max_angle, method=method.lower(),
                                      rejection_sigma=rejection_sigma)
            st.session_state.workflow_last = (signature, result)
            st.rerun()
        except (ValueError, OSError, TypeError, KeyError, MemoryError) as exc:
            st.error(str(exc) or 'The workflow could not be completed.')

    last = st.session_state.get('workflow_last')
    if last and last[0] == signature:
        result = last[1]
        st.download_button('Download FITS', to_fits(result), result.name,
                           'application/fits', key='workflow_download')
        view = st.selectbox('View', ['Result', 'Original'], key='workflow_preview',
                            on_change=on_view_change)
        return result if view == 'Result' else None
    if last:
        st.caption('Options changed. Run workflow to update the result.')
    return None
