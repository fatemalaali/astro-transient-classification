/* Fetch wrappers and shared formatting helpers.
 *
 * No framework, no build step: the dashboard is six views, and a bundler would
 * add a demo-day failure mode without buying anything.
 */
const API = (() => {
  async function get(path, params) {
    const url = new URL(path, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        if (v !== null && v !== undefined && v !== '') url.searchParams.set(k, v);
      });
    }
    const response = await fetch(url, { headers: { Accept: 'application/json' } });
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail || detail; } catch (e) { /* keep */ }
      const error = new Error(detail);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  return {
    config:        ()               => get('/api/config'),
    alerts:        (params)         => get('/api/alerts', params),
    alert:         (candid)         => get(`/api/alerts/${candid}`),
    trace:         (candid)         => get(`/api/alerts/${candid}/trace`),
    packet:        (candid)         => get(`/api/alerts/${candid}/packet`),
    object:        (oid)            => get(`/api/objects/${encodeURIComponent(oid)}`),
    lightcurve:    (oid)            => get(`/api/objects/${encodeURIComponent(oid)}/lightcurve`),
    health:        ()               => get('/api/stream/health'),
    stats:         ()               => get('/api/stats'),
    topics:        ()               => get('/api/topics'),
    disagreements: (params)         => get('/api/disagreements', params),
    alerceStatus:  ()               => get('/api/alerce/status'),
    alerceProbe:   async ()         => {
      const r = await fetch('/api/alerce/probe', { method: 'POST' });
      if (!r.ok) throw new Error('probe failed');
      return r.json();
    },
    provenance:    ()               => get('/api/provenance'),
    evaluation:    ()               => get('/api/evaluation'),
    models:        ()               => get('/api/models'),
    stampUrl:      (candid, kind, stretch) =>
      `/api/alerts/${candid}/stamp/${kind}.png?stretch=${stretch || 'sigmoid'}`,
  };
})();

/* ----------------------------------------------------------------- helpers */
const FMT = {
  CLASSES: ['SN', 'AGN', 'VS'],

  esc(value) {
    if (value === null || value === undefined) return '';
    return String(value).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  },

  num(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return Number(value).toFixed(digits);
  },

  int(value) {
    if (value === null || value === undefined) return '—';
    return Number(value).toLocaleString();
  },

  /* Sexagesimal is what an astronomer reads coordinates in. */
  ra(deg) {
    if (deg === null || deg === undefined || Number.isNaN(deg)) return '—';
    const hours = deg / 15;
    const h = Math.floor(hours);
    const m = Math.floor((hours - h) * 60);
    const s = (((hours - h) * 60) - m) * 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
  },

  dec(deg) {
    if (deg === null || deg === undefined || Number.isNaN(deg)) return '—';
    const sign = deg < 0 ? '-' : '+';
    const a = Math.abs(deg);
    const d = Math.floor(a);
    const m = Math.floor((a - d) * 60);
    const s = (((a - d) * 60) - m) * 60;
    return `${sign}${String(d).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
  },

  time(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '—';
    return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
  },

  ago(iso) {
    if (!iso) return '—';
    const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
    if (!Number.isFinite(seconds)) return '—';
    if (seconds < 60) return `${Math.round(seconds)}s ago`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
    if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h ago`;
    return `${(seconds / 86400).toFixed(1)}d ago`;
  },

  duration(seconds) {
    if (seconds === null || seconds === undefined) return '—';
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  },

  classChip(name) {
    const cls = name && FMT.CLASSES.includes(name) ? name : 'none';
    return `<span class="chip chip-${cls}">${FMT.esc(name || 'n/a')}</span>`;
  },

  classColour(name) {
    return { SN: '#d62728', AGN: '#1f77b4', VS: '#2ca02c' }[name] || '#97a4b2';
  },

  confBar(value, className) {
    if (value === null || value === undefined) return '—';
    const pct = Math.max(0, Math.min(1, value)) * 100;
    return `<span class="confbar"><span style="width:${pct.toFixed(1)}%;
            background:${FMT.classColour(className)}"></span></span>
            <span class="confval">${value.toFixed(3)}</span>`;
  },

  probRows(proba, argmax) {
    if (!proba) return '<div class="muted small">not available</div>';
    return FMT.CLASSES.map((cls) => {
      const value = proba[cls] ?? 0;
      const isMax = cls === argmax;
      return `<div class="probrow ${isMax ? 'argmax' : ''}">
        <span class="name">${cls}</span>
        <span class="bar"><span style="width:${(value * 100).toFixed(1)}%;
              background:${FMT.classColour(cls)}"></span></span>
        <span class="val">${value.toFixed(3)}</span>
      </div>`;
    }).join('');
  },

  /* An alert for an object the model was fitted on must be visibly flagged:
     quoting accuracy over such objects is the mistake examiners look for.
     Held-out (test) objects are marked differently — the model never saw them,
     so their predictions ARE evidence, and conflating the two would be as
     misleading as hiding the flag entirely. */
  trainingBadge(known) {
    if (!known || !known.in_training_set) return '';
    const split = known.training_split;
    if (split === 'train' || split === 'val') {
      return `<span class="tag tag-danger" title="This object was fitted on (split: ${split}).
Its prediction is not evidence of generalisation.">fitted on (${FMT.esc(split)})</span>`;
    }
    if (split === 'test') {
      return `<span class="tag tag-info" title="Gold-set object from the held-out test fold.
The deployed models never trained on it, so this prediction is out-of-sample.">held-out test</span>`;
    }
    return `<span class="tag" title="In the gold set, split unknown.">gold set</span>`;
  },
};
