"""Browser-side FITS preview with instant, non-destructive display adjustments."""
from base64 import b64encode
from json import dumps

import numpy as np
import streamlit.components.v1 as components


def show(preview, step):
    values = np.asarray(preview, dtype='<f8', order='C')
    encoded = b64encode(values.tobytes()).decode('ascii')
    finite = values[np.isfinite(values)]
    percentiles = np.percentile(finite, np.arange(101)).tolist()
    rows, columns = values.shape
    html = f'''<!doctype html><html><head><style>
        * {{box-sizing:border-box;}}
        html, body {{margin:0; font:13px sans-serif; color:#ddd; background:#000; overflow:hidden;}}
        .image {{width:100%; height:calc(100vh - 60px); display:flex; align-items:center; justify-content:center; overflow:hidden;}}
        .plot {{position:relative; flex:none;}}
        .plot canvas {{position:absolute; display:block; image-rendering:auto;}}
        #axes {{pointer-events:none;}}
        .readout {{height:20px; color:#aaa; font-size:12px;}}
        .controls {{display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding-top:4px;}}
        label {{display:flex; align-items:center; gap:4px; font-size:12px;}}
        select, button {{border:1px solid #555; border-radius:0; background:#000; color:#ddd; height:26px; padding:0 5px;}}
        input[type=range], input[type=checkbox] {{accent-color:#bbb;}}
        input[type=range] {{width:95px;}}
    </style></head><body>
    <div class="image"><div class="plot"><canvas id="image" width="{columns}" height="{rows}"></canvas><canvas id="axes"></canvas></div></div>
    <div class="readout" id="readout">X (px) · Y (px) · Value</div>
    <div class="controls">
      <label>Stretch<select id="stretch"><option>Linear</option><option>Asinh</option><option>Logarithmic</option><option>Square root</option></select></label>
      <label>Low % <span id="low-label">1</span><input id="low" type="range" min="0" max="99" value="1"></label>
      <label>High % <span id="high-label">99</span><input id="high" type="range" min="1" max="100" value="99"></label>
      <label>Invert<input id="invert" type="checkbox"></label>
      <label><span>Grid</span><input id="grid" type="checkbox"></label>
      <button id="save" type="button">Save PNG</button>
    </div>
    <script>
    const raw = atob({dumps(encoded)});
    const bytes = Uint8Array.from(raw, c => c.charCodeAt(0));
    const pixels = new Float64Array(bytes.buffer);
    const percentiles = {dumps(percentiles)};
    const rows = {rows}, columns = {columns}, step = {step};
    const canvas = document.getElementById('image');
    const holder = document.querySelector('.image');
    const plot = document.querySelector('.plot');
    const axes = document.getElementById('axes');
    const axisContext = axes.getContext('2d');
    const grid = document.getElementById('grid');
    const left = 44, bottom = 24;
    const ctx = canvas.getContext('2d', {{alpha:false}});
    const buffer = document.createElement('canvas');
    buffer.width = columns; buffer.height = rows;
    const offscreen = buffer.getContext('2d', {{alpha:false}});
    const controls = ['stretch','low','high','invert'].map(id => document.getElementById(id));
    let scheduled = false;
    let displayWidth = 0, displayHeight = 0, displayScale = 1;
    function tickInterval(total) {{
      const rough = total / 6;
      const magnitude = 10 ** Math.floor(Math.log10(Math.max(rough, 1)));
      return [1, 2, 5, 10].map(n => n*magnitude).find(n => n >= rough);
    }}
    function drawAxes() {{
      const c = axisContext;
      const width = left + displayWidth, height = displayHeight;
      c.clearRect(0, 0, width, height + bottom);
      c.lineWidth = 1;
      c.font = '11px sans-serif';
      c.fillStyle = '#aaa';
      c.strokeStyle = '#777';
      c.beginPath(); c.moveTo(left, 0); c.lineTo(left, height); c.lineTo(width, height); c.stroke();
      const xInterval = tickInterval(columns*step);
      const yInterval = tickInterval(rows*step);
      for (let x=0; x<columns*step; x+=xInterval) {{
        const px = left + x/step*displayScale;
        if (px > width) break;
        if (grid.checked && x) {{
          c.strokeStyle='rgba(255,255,255,0.22)'; c.beginPath();
          c.moveTo(px,0); c.lineTo(px,height); c.stroke();
        }}
        c.strokeStyle='#777'; c.beginPath(); c.moveTo(px,height); c.lineTo(px,height+4); c.stroke();
        c.textAlign='center'; c.fillText(String(x), Math.min(width-16, Math.max(left+10,px)), height+17);
      }}
      for (let y=0; y<rows*step; y+=yInterval) {{
        const py = height-y/step*displayScale;
        if (py < 0) break;
        if (grid.checked && y) {{
          c.strokeStyle='rgba(255,255,255,0.22)'; c.beginPath();
          c.moveTo(left,py); c.lineTo(width,py); c.stroke();
        }}
        c.strokeStyle='#777'; c.beginPath(); c.moveTo(left-4,py); c.lineTo(left,py); c.stroke();
        c.textAlign='right'; c.fillText(String(y), left-7, Math.max(11,py+3));
      }}
      c.save(); c.translate(10, 45); c.rotate(-Math.PI/2);
      c.textAlign='center'; c.fillText('Y (px)', 0, 0); c.restore();
    }}
    function fitCanvas() {{
      displayScale = Math.max(0.01, Math.min((holder.clientWidth-left)/columns,
                                            (holder.clientHeight-bottom)/rows));
      displayWidth = columns*displayScale;
      displayHeight = rows*displayScale;
      plot.style.width = `${{left+displayWidth}}px`;
      plot.style.height = `${{displayHeight+bottom}}px`;
      canvas.style.left = `${{left}}px`;
      canvas.style.width = `${{displayWidth}}px`;
      canvas.style.height = `${{displayHeight}}px`;
      const ratio = window.devicePixelRatio || 1;
      axes.width = Math.ceil((left+displayWidth)*ratio);
      axes.height = Math.ceil((displayHeight+bottom)*ratio);
      axes.style.width = `${{left+displayWidth}}px`;
      axes.style.height = `${{displayHeight+bottom}}px`;
      axisContext.setTransform(ratio,0,0,ratio,0,0);
      drawAxes();
    }}
    window.addEventListener('resize', fitCanvas);
    new ResizeObserver(fitCanvas).observe(holder);
    grid.addEventListener('change', drawAxes);
    function render() {{
      scheduled = false;
      const [stretch, lowControl, highControl, invert] = controls;
      const lo = +lowControl.value, hi = +highControl.value;
      document.getElementById('low-label').textContent = lo;
      document.getElementById('high-label').textContent = hi;
      const low = percentiles[lo], high = percentiles[hi];
      const span = high > low ? high-low : 1;
      const image = offscreen.createImageData(columns, rows);
      const out = image.data;
      for (let y=0; y<rows; y++) for (let x=0; x<columns; x++) {{
        const source = (rows-1-y)*columns+x;
        let v = pixels[source];
        const index = 4*(y*columns+x);
        if (!Number.isFinite(v)) {{out[index]=out[index+1]=out[index+2]=0; out[index+3]=255; continue;}}
        v = Math.max(0, Math.min(1, (v-low)/span));
        if (stretch.value === 'Asinh') v = Math.asinh(10*v)/Math.asinh(10);
        else if (stretch.value === 'Logarithmic') v = Math.log1p(1000*v)/Math.log1p(1000);
        else if (stretch.value === 'Square root') v = Math.sqrt(v);
        const gray = Math.round(255*(invert.checked ? 1-v : v));
        out[index]=out[index+1]=out[index+2]=gray; out[index+3]=255;
      }}
      offscreen.putImageData(image, 0, 0);
      ctx.drawImage(buffer, 0, 0);
    }}
    function queueRender() {{if (!scheduled) {{scheduled=true; requestAnimationFrame(render);}}}}
    controls.forEach(control => control.addEventListener('input', queueRender));
    canvas.addEventListener('mousemove', event => {{
      const rect=canvas.getBoundingClientRect();
      const x=Math.min(columns-1,Math.max(0,Math.floor((event.clientX-rect.left)*columns/rect.width)));
      const y=Math.min(rows-1,Math.max(0,Math.floor((event.clientY-rect.top)*rows/rect.height)));
      document.getElementById('readout').textContent=`X=${{x*step}} px · Y=${{(rows-1-y)*step}} px · Value=${{pixels[(rows-1-y)*columns+x].toPrecision(6)}}`;
    }});
    document.getElementById('save').addEventListener('click', () => {{
      const link=document.createElement('a'); link.download='fits_view.png'; link.href=canvas.toDataURL('image/png'); link.click();
    }});
    fitCanvas();
    render();
    </script></body></html>'''
    components.html(html, height=1250, scrolling=False)
