# FITS Viewer

A local FITS viewer for astronomical images. The interface is in English and runs in a browser; files and processing stay on your computer.

## Start

On macOS, double-click `iniciar.command`. Or run:

```sh
source .venv/bin/activate
python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://localhost:8501. Press Ctrl+C in the terminal to stop it.

## Use

In the fixed-width sidebar, use **File** to choose a FITS file from a folder or upload one directly. Statistics and the complete FITS header are separate tabs below the file selector. Header values and units appear on the same line. Use **Processing** for individual CCD operations or **Workflow** to configure bias, flats, background, alignment, and combination, then run them in order. Stretch, grayscale inversion, contrast percentiles, a light grid, and PNG export sit below the image. X and Y axes show pixel positions. Sidebar option edits leave the image in place until an operation is run. Scrolling over the fixed image scrolls the page.

The tabs below the image provide crop and numeric adjustment, histogram, and pixel profiles. In **Crop / adjust**, choose bounds and click **Download FITS**. Coordinates begin at zero; end bounds are exclusive. Contrast and stretch affect only the view, not FITS exports.

In **Processing**, choose an operation and its inputs, then click **Run**. Available operations are master bias, flat creation/correction, image combination, masking, and science image reduction. Processing results can be previewed and downloaded. A created master can be reused within the session. Combining images optionally aligns by X/Y translation; it does not correct rotation or scale.

In **Workflow**, choose existing masters or source bias/flat frames, optionally subtract the global sigma-clipped sky median, align additional science frames by translation or translation plus small rotation, and combine by median or mean. **Run workflow** applies the selected steps in order to the open image and any additional science frames; the result can be previewed or downloaded. Rotation support in Workflow is limited to 15 degrees, and scaling is not corrected.

Original files are never overwritten. The viewer reduces previews above 1600 pixels per axis; analysis and exports use full resolution. Browser uploads use Streamlit's default 200 MB limit. Large FITS files and batches can require substantial memory.

## Install and test

Python 3.9 or later:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

`requirements-lock.txt` records the versions used to build the included environment.
