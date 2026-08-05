/* Hand-rolled SVG plotting.
 *
 * Deviation from docs/demo-plan.md 6.7, which specified vendoring Plotly: the
 * only plots this dashboard needs are a magnitude-vs-time scatter with error
 * bars and a lag sparkline. Rendering those as SVG directly is ~200 lines and
 * removes a 3.5 MB vendored blob from the repository, while keeping the
 * offline guarantee absolute — there is no library to fail to load.
 *
 * Astronomical conventions honoured here:
 *   - the magnitude axis is inverted (brighter is up);
 *   - non-detections are 5-sigma upper limits, drawn as hollow downward
 *     triangles at diffmaglim, never as data points;
 *   - filters keep their conventional colours (ZTF-g green, ZTF-r red).
 */
const PLOT = (() => {
  const BAND_COLOUR = { g: '#2ca02c', r: '#d62728', i: '#8c564b', '?': '#7b8794' };
  const NS = 'http://www.w3.org/2000/svg';

  function el(name, attrs, text) {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function lightcurve(container, payload, options) {
    const opts = Object.assign(
      { width: 720, height: 300, margin: { top: 12, right: 14, bottom: 34, left: 46 } },
      options || {}
    );
    container.innerHTML = '';

    const bands = payload && payload.bands ? payload.bands : {};
    const points = [];
    const limits = [];
    Object.entries(bands).forEach(([band, series]) => {
      (series.detections || []).forEach((d) => {
        if (d.magpsf !== null && d.magpsf !== undefined) {
          points.push({ band, mjd: d.mjd, mag: d.magpsf, err: d.sigmapsf || 0 });
        }
      });
      (series.nondetections || []).forEach((d) => {
        if (d.diffmaglim !== null && d.diffmaglim !== undefined) {
          limits.push({ band, mjd: d.mjd, mag: d.diffmaglim });
        }
      });
    });

    if (!points.length && !limits.length) {
      container.innerHTML = '<div class="empty">No photometry stored for this object.</div>';
      return;
    }

    const all = points.concat(limits);
    const xs = all.map((p) => p.mjd);
    const ysLow = points.map((p) => p.mag - (p.err || 0)).concat(limits.map((p) => p.mag));
    const ysHigh = points.map((p) => p.mag + (p.err || 0)).concat(limits.map((p) => p.mag));

    let x0 = Math.min(...xs);
    let x1 = Math.max(...xs);
    if (x1 - x0 < 1e-6) { x0 -= 1; x1 += 1; }
    const xPad = (x1 - x0) * 0.04;
    x0 -= xPad; x1 += xPad;

    let y0 = Math.min(...ysLow);
    let y1 = Math.max(...ysHigh);
    if (y1 - y0 < 1e-6) { y0 -= 0.5; y1 += 0.5; }
    const yPad = (y1 - y0) * 0.08;
    y0 -= yPad; y1 += yPad;

    const { width: W, height: H, margin: M } = opts;
    const plotW = W - M.left - M.right;
    const plotH = H - M.top - M.bottom;
    const sx = (v) => M.left + ((v - x0) / (x1 - x0)) * plotW;
    // Inverted: brighter (smaller magnitude) is higher on the page.
    const sy = (v) => M.top + ((v - y0) / (y1 - y0)) * plotH;

    const svg = el('svg', {
      class: 'lightcurve', width: W, height: H, viewBox: `0 0 ${W} ${H}`,
      role: 'img', 'aria-label': 'Light curve',
    });

    const grid = el('g', { class: 'axis' });
    for (let i = 0; i <= 4; i += 1) {
      const y = M.top + (plotH * i) / 4;
      grid.appendChild(el('line', {
        x1: M.left, x2: M.left + plotW, y1: y, y2: y,
        stroke: '#eef1f5', 'stroke-width': 1,
      }));
      const value = y0 + ((y1 - y0) * i) / 4;
      grid.appendChild(el('text', {
        x: M.left - 6, y: y + 3, 'text-anchor': 'end',
      }, value.toFixed(1)));
    }
    for (let i = 0; i <= 4; i += 1) {
      const x = M.left + (plotW * i) / 4;
      grid.appendChild(el('line', {
        x1: x, x2: x, y1: M.top, y2: M.top + plotH,
        stroke: '#f4f6f9', 'stroke-width': 1,
      }));
      const value = x0 + ((x1 - x0) * i) / 4;
      grid.appendChild(el('text', {
        x, y: M.top + plotH + 15, 'text-anchor': 'middle',
      }, value.toFixed(1)));
    }
    svg.appendChild(grid);

    svg.appendChild(el('line', {
      x1: M.left, x2: M.left + plotW, y1: M.top + plotH, y2: M.top + plotH,
      stroke: '#dfe5ec',
    }));
    svg.appendChild(el('line', {
      x1: M.left, x2: M.left, y1: M.top, y2: M.top + plotH, stroke: '#dfe5ec',
    }));
    svg.appendChild(el('text', {
      x: M.left + plotW / 2, y: H - 3, 'text-anchor': 'middle',
    }, 'MJD'));
    svg.appendChild(el('text', {
      x: 11, y: M.top + plotH / 2, 'text-anchor': 'middle',
      transform: `rotate(-90 11 ${M.top + plotH / 2})`,
    }, 'magnitude (difference PSF)'));

    // Upper limits first, so detections draw over them.
    limits.forEach((p) => {
      const x = sx(p.mjd);
      const y = sy(p.mag);
      const colour = BAND_COLOUR[p.band] || BAND_COLOUR['?'];
      const tri = el('path', {
        d: `M ${x - 4} ${y - 4} L ${x + 4} ${y - 4} L ${x} ${y + 4} Z`,
        fill: 'none', stroke: colour, 'stroke-width': 1.2, opacity: 0.75,
      });
      tri.appendChild(el('title', {},
        `${p.band}: upper limit ${p.mag.toFixed(2)} mag at MJD ${p.mjd.toFixed(3)}`));
      svg.appendChild(tri);
    });

    points.forEach((p) => {
      const x = sx(p.mjd);
      const y = sy(p.mag);
      const colour = BAND_COLOUR[p.band] || BAND_COLOUR['?'];
      if (p.err) {
        svg.appendChild(el('line', {
          x1: x, x2: x, y1: sy(p.mag - p.err), y2: sy(p.mag + p.err),
          stroke: colour, 'stroke-width': 1, opacity: 0.6,
        }));
      }
      const dot = el('circle', { cx: x, cy: y, r: 3.2, fill: colour });
      dot.appendChild(el('title', {},
        `${p.band} = ${p.mag.toFixed(3)} ± ${(p.err || 0).toFixed(3)} at MJD ${p.mjd.toFixed(3)}`));
      svg.appendChild(dot);
    });

    const wrap = document.createElement('div');
    wrap.className = 'lc-wrap';
    wrap.appendChild(svg);
    container.appendChild(wrap);

    const usedBands = Array.from(new Set(all.map((p) => p.band)));
    const legend = document.createElement('div');
    legend.className = 'lc-legend';
    legend.innerHTML = usedBands.map((band) =>
      `<span class="key"><span class="swatch" style="background:${BAND_COLOUR[band] || '#7b8794'}"></span>
       ZTF-${FMT.esc(band)}</span>`
    ).join('') +
      '<span class="key muted">▽ 5σ upper limit (observed, not detected)</span>';
    container.appendChild(legend);
  }

  function sparkline(values, opts) {
    const o = Object.assign({ width: 90, height: 18, colour: '#4ade80' }, opts || {});
    if (!values || values.length < 2) return '';
    const max = Math.max(...values, 1);
    const step = o.width / (values.length - 1);
    const points = values
      .map((v, i) => `${(i * step).toFixed(1)},${(o.height - (v / max) * o.height).toFixed(1)}`)
      .join(' ');
    return `<svg class="sparkline" width="${o.width}" height="${o.height}"
             viewBox="0 0 ${o.width} ${o.height}">
      <polyline points="${points}" fill="none" stroke="${o.colour}" stroke-width="1.4"/>
    </svg>`;
  }

  function bars(entries, opts) {
    const o = Object.assign({ height: 90, colour: () => '#1f77b4' }, opts || {});
    const max = Math.max(...entries.map((e) => e[1]), 1);
    return `<div style="display:flex;align-items:flex-end;gap:4px;height:${o.height}px">` +
      entries.map(([label, value]) => {
        const h = Math.max(2, (value / max) * (o.height - 18));
        return `<div style="flex:1;text-align:center" title="${FMT.esc(label)}: ${value}">
          <div style="height:${h}px;background:${o.colour(label)};border-radius:2px 2px 0 0"></div>
          <div class="small muted" style="font-size:.62rem">${FMT.esc(label)}</div>
        </div>`;
      }).join('') + '</div>';
  }

  return { lightcurve, sparkline, bars };
})();
