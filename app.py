"""Direct FITS viewer: file controls at the side, image and tools in one page."""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy.io import fits

from fits_tools import export_crop, statistics
from image_viewer import show as show_image
from processing_ui import identity, list_fits, processing_panel
from reduction import frame
from workflow_ui import workflow_panel

st.set_page_config(page_title='FITS Viewer', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {padding-top: .5rem;}
:root {--image-height: clamp(450px, calc((100vw - 530px) * .667 + 75px), 1250px);}
[data-testid="stVerticalBlock"] {gap: .5rem !important;}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {display: none !important;}
[data-testid="stSidebar"] {width: 380px !important; min-width: 380px; max-width: 380px; transform: translateX(0) !important;}
[data-testid="stSidebarUserContent"] {padding: .5rem .75rem;}
[data-testid="stFileUploaderDropzone"] {height: 42px; min-height: 0; padding: .25rem .5rem;}
[data-testid="stFileUploaderDropzoneInstructions"] > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:last-child {display: none;}
[data-testid="stNumberInput"] button {display: none;}
[data-testid="stMain"] [data-testid="stTabs"] [role="tabpanel"] {height: 350px; overflow-y: auto;}
.st-key-image_viewer {height: var(--image-height); min-height: var(--image-height); overflow: hidden;}
iframe[title="st.iframe"] {height: var(--image-height) !important; border: 0 !important;}
[data-testid="stElementContainer"]:has(iframe[title="st.iframe"]) {
    height: var(--image-height) !important;
    flex: 0 0 var(--image-height) !important;
}
</style>''', unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DEFAULT_FOLDER = ROOT / 'share-20260922-download'
if 'folder_path' not in st.session_state:
    st.session_state.folder_path = str(DEFAULT_FOLDER if DEFAULT_FOLDER.is_dir() else ROOT)


def header_panel(header):
    lines = []
    for card in header.cards:
        suffix = f' {card.comment}' if card.comment else ''
        lines.append(f'{card.keyword:<12}  {card.value}{suffix}')
    st.text('\n'.join(lines))


def statistics_panel(info):
    rows = [('Pixels', info['pixeis']), ('Minimum', info['minimo']),
            ('Maximum', info['maximo']), ('Mean', info['media']),
            ('Median', info['mediana']), ('Std dev', info['desvio']),
            ('Sum', info['soma'])]
    st.text('\n'.join(f'{key:<12}  {value:,}' if isinstance(value, int)
                      else f'{key:<12}  {value:.6g}' for key, value in rows))


def sidebar_details(slot, header, info=None):
    with slot.container():
        if info:
            statistics_tab, header_tab = st.tabs(['Statistics', 'Header'])
            with statistics_tab:
                statistics_panel(info)
        else:
            header_tab, = st.tabs(['Header'])
        with header_tab:
            header_panel(header)


def refresh_image_view():
    st.session_state.refresh_image_view = True


@st.fragment
def processing_controls(original, source_id, folder):
    result = processing_panel(original, source_id, folder, on_view_change=refresh_image_view)
    if st.session_state.pop('refresh_image_view', False):
        st.rerun()
    return result


@st.fragment
def workflow_controls(original, source_id, folder):
    result = workflow_panel(original, source_id, folder, on_view_change=refresh_image_view)
    if st.session_state.pop('refresh_image_view', False):
        st.rerun()
    return result


@st.fragment
def crop_controls(data, header, mask, name):
    height, width = data.shape[-2:]
    x0, x1, y0, y1 = st.columns(4)
    x0 = x0.number_input('X start', 0, width-1, 0)
    x1 = x1.number_input('X end (exclusive)', int(x0)+1, width, width)
    y0 = y0.number_input('Y start', 0, height-1, 0)
    y1 = y1.number_input('Y end (exclusive)', int(y0)+1, height, height)
    offset, gain = st.columns(2)
    offset = offset.number_input('Subtract', value=0.)
    gain = gain.number_input('Multiply by', value=1.)
    output = export_crop(data, header, x0, x1, y0, y1, offset, gain, mask=mask)
    st.download_button('Download FITS', output, Path(name).stem+'_result.fits',
                       'application/fits')


@st.fragment
def profile_controls(plane):
    height, width = plane.shape
    col1, col2 = st.columns(2)
    px = col1.number_input('X pixel', 0, width-1, 0)
    py = col2.number_input('Y pixel', 0, height-1, 0)
    st.write(f'Value: {plane[py, px]:.8g}')
    for axis, values in [(f'Row Y={py}', plane[py, :]), (f'Column X={px}', plane[:, px])]:
        profile = go.Figure(go.Scatter(y=values, mode='lines', line_color='#555555'))
        profile.update_layout(xaxis_title=axis, yaxis_title='Value',
                              template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000',
                              height=220, margin=dict(l=45, r=10, t=10, b=40))
        st.plotly_chart(profile, use_container_width=True,
                        config={'displayModeBar': False})


with st.sidebar:
    file_tab, processing_tab, workflow_tab = st.tabs(['File', 'Processing', 'Workflow'])
    with file_tab:
        folder = Path(st.text_input('Folder', key='folder_path')).expanduser()
        files = []
        if folder.is_dir():
            try:
                files = list_fits(folder)
            except OSError as exc:
                st.error(f'Cannot open folder: {exc}')
        else:
            st.error('Folder not found.')
        local_file = st.selectbox('File in folder', files, format_func=lambda path: path.name,
                                  index=0 if files else None, placeholder='Choose a FITS file')
        upload = st.file_uploader('Upload FITS', type=['fits', 'fit', 'fts', 'gz', 'fz'])
        source = upload if upload is not None else local_file
        details_slot = st.empty()

if source is None:
    st.info('Choose a FITS file in the sidebar.')
    st.stop()

name = source.name
source_id = identity(source) if isinstance(source, Path) else identity(source.getvalue())
st.caption(name)

try:
    hdus = fits.open(source if isinstance(source, Path) else BytesIO(source.getvalue()), memmap=False)
except Exception as exc:
    st.error(f'Cannot read FITS file: {exc}')
    st.stop()

with hdus:
    index = st.selectbox('HDU', range(len(hdus)),
                         format_func=lambda i: f'{i} · {hdus[i].name} ({type(hdus[i]).__name__})') if len(hdus) > 1 else 0
    hdu = hdus[index]
    header = hdu.header.copy()
    try:
        data = hdu.data
    except Exception as exc:
        st.error(f'Cannot read image data: {exc}')
        st.stop()
    if data is None:
        st.info('This HDU has no data. Choose another HDU.')
        st.stop()

    if data.dtype.names:
        table = pd.DataFrame({column: [value.tolist() if isinstance(value, np.ndarray) else value
                                       for value in data[column]] for column in data.dtype.names})
        st.dataframe(table, width='stretch')
        st.download_button('Download CSV', table.to_csv(index=False), 'table.csv', 'text/csv')
        sidebar_details(details_slot, header)
        with workflow_tab:
            st.caption('Workflow requires a 2D image.')
    elif data.ndim == 1:
        figure = go.Figure(go.Scatter(y=data, mode='lines', line_color='#555555'))
        figure.update_layout(xaxis_title='Pixel', yaxis_title=header.get('BUNIT', 'Value'),
                             template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000',
                             height=320, margin=dict(l=45, r=10, t=10, b=40))
        st.plotly_chart(figure, use_container_width=True,
                        config={'scrollZoom': False, 'displayModeBar': False})
        st.download_button('Download CSV', pd.DataFrame({'pixel': np.arange(data.size),
                           'value': data}).to_csv(index=False), 'series.csv', 'text/csv')
        sidebar_details(details_slot, header, statistics(data))
        with workflow_tab:
            st.caption('Workflow requires a 2D image.')
    elif data.ndim >= 2 and all(data.shape):
        active_mask = None
        if data.ndim == 2:
            source_mask = hdus['MASK'].data if 'MASK' in hdus and hdu.name != 'MASK' else None
            original = frame(data, header, source_mask, name)
            with processing_tab:
                processed = processing_controls(original, (source_id, index), folder)
            with workflow_tab:
                workflow_result = workflow_controls(original, (source_id, index), folder)
            active = workflow_result or processed or original
            data, header, active_mask = active.data, active.header, active.mask
        slices = []
        if data.ndim > 2:
            with st.sidebar:
                for axis, size in enumerate(data.shape[:-2]):
                    slices.append(st.number_input(f'Plane · axis {data.ndim-axis}', 0, size-1, 0,
                                                   key=f'axis{axis}'))
        plane = np.asarray(data[tuple(slices)] if slices else data, dtype=float)
        if active_mask is not None:
            plane = np.where(active_mask, np.nan, plane)
        height, width = plane.shape
        finite = plane[np.isfinite(plane)]
        if not finite.size:
            st.warning('This image has no valid pixels to display.')
            st.stop()
        step = max(1, int(np.ceil(max(plane.shape) / 1600)))
        preview = plane[::step, ::step]
        with st.container(key='image_viewer'):
            show_image(preview, step)
        info = statistics(plane)
        sidebar_details(details_slot, header, info)
        tabs = st.tabs(['Crop / adjust', 'Histogram', 'Profiles'])
        with tabs[0]:
            crop_controls(data, header, active_mask, name)
        with tabs[1]:
            counts, bins = np.histogram(finite, bins=100)
            histogram = go.Figure(go.Bar(x=(bins[1:]+bins[:-1])/2, y=counts,
                                         marker_color='#555555'))
            histogram.update_layout(xaxis_title='Value', yaxis_title='Pixels',
                                    template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000',
                                    height=320, margin=dict(l=45, r=10, t=10, b=40))
            st.plotly_chart(histogram, use_container_width=True,
                            config={'displayModeBar': False})
        with tabs[2]:
            profile_controls(plane)
    else:
        st.info('This HDU does not contain a supported image or series.')
