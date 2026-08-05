/* Alert-stream view: filter panel + auto-refreshing results table.
 *
 * Modelled on the ALeRCE Explorer's home page (verified at alerce.online):
 * a left "Search Filters" panel with General Filters, a discovery-date section
 * and SEARCH / CLEAR buttons, beside a sortable results table.
 *
 * Two deliberate departures. ALeRCE's "Classifier" dropdown becomes a *branch*
 * selector, because which branch you are looking at is this thesis' comparison
 * axis. And the table is alert-centric rather than object-centric — this is a
 * stream, not an archive — so Received and Topic replace FirstMJD/DeltaMJD.
 *
 * Refresh is polling, not SSE: the alert rate is minutes-per-alert, polling
 * survives a consumer restart with no reconnect logic, and it is debuggable
 * from the browser network tab in front of an examiner.
 */
const StreamView = (() => {
  const POLL_MS = 3000;

  const state = {
    filters: {
      object_id: '',
      class: '',
      min_confidence: 0,
      since: '',
      topic: '',
      fusion_mode: '',
      disagree_only: false,
    },
    branch: 'fused',
    sort: 'received',
    order: 'desc',
    items: [],
    total: 0,
    timer: null,
    topics: [],
  };

  function render(root) {
    root.innerHTML = `
      <div class="layout">
        <aside>
          <div class="panel filters">
            <h2>Search filters</h2>
            <h4>General</h4>

            <label for="f-oid">Object ID</label>
            <input type="text" id="f-oid" placeholder="ZTF26abc…" value="${FMT.esc(state.filters.object_id)}">

            <label for="f-branch">Branch shown</label>
            <select id="f-branch">
              <option value="fused">Late fusion (deployed)</option>
              <option value="tabular">Tabular only</option>
              <option value="image">Image only</option>
            </select>

            <label>Predicted class</label>
            <div class="checks" id="f-classes">
              ${FMT.CLASSES.map((c) => `
                <label class="check"><input type="checkbox" value="${c}"> ${FMT.classChip(c)}</label>
              `).join('')}
            </div>

            <label for="f-conf">Confidence &ge; <span id="f-conf-val" class="mono">0.00</span></label>
            <input type="range" id="f-conf" min="0" max="1" step="0.01" value="0">

            <label for="f-since">Time window</label>
            <select id="f-since">
              <option value="">all time</option>
              <option value="15m">last 15 minutes</option>
              <option value="1h">last hour</option>
              <option value="6h">last 6 hours</option>
              <option value="24h">last 24 hours</option>
              <option value="7d">last 7 days</option>
            </select>

            <label for="f-topic">Kafka topic</label>
            <select id="f-topic"><option value="">all topics</option></select>

            <label for="f-mode">Fusion mode</label>
            <select id="f-mode">
              <option value="">any</option>
              <option value="both">both branches</option>
              <option value="tabular_only">tabular only (no stamp)</option>
              <option value="image_only">image only (no features)</option>
            </select>

            <div class="checks">
              <label class="check">
                <input type="checkbox" id="f-disagree"> disagreements only
              </label>
            </div>

            <div class="actions">
              <button class="primary" id="f-search">SEARCH</button>
              <button class="ghost" id="f-clear">CLEAR</button>
            </div>
          </div>

          <div class="panel" id="stats-panel"><h2>Statistics</h2>
            <div class="loading">…</div></div>
        </aside>

        <section>
          <div class="panel">
            <div class="panel-head">
              <h2>Alert stream</h2>
              <span class="count" id="result-count"></span>
              <span class="count muted" style="margin-left:auto" id="refresh-note">
                auto-refreshing every 3&nbsp;s</span>
            </div>
            <div id="table-wrap"><div class="loading">Loading alerts…</div></div>
          </div>
        </section>
      </div>`;

    wire(root);
    loadTopics();
    refresh();
    loadStats();
    start();
  }

  function wire(root) {
    const conf = root.querySelector('#f-conf');
    conf.addEventListener('input', () => {
      root.querySelector('#f-conf-val').textContent = Number(conf.value).toFixed(2);
    });
    root.querySelector('#f-search').addEventListener('click', () => {
      collect(root);
      refresh();
    });
    root.querySelector('#f-clear').addEventListener('click', () => {
      state.filters = {
        object_id: '', class: '', min_confidence: 0, since: '',
        topic: '', fusion_mode: '', disagree_only: false,
      };
      App.navigate('#/');
    });
    root.querySelector('#f-oid').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { collect(root); refresh(); }
    });
    root.querySelector('#f-branch').addEventListener('change', (e) => {
      state.branch = e.target.value;
      draw();
    });
  }

  function collect(root) {
    state.filters.object_id = root.querySelector('#f-oid').value.trim();
    state.filters.class = Array.from(
      root.querySelectorAll('#f-classes input:checked')
    ).map((i) => i.value).join(',');
    state.filters.min_confidence = Number(root.querySelector('#f-conf').value) || 0;
    state.filters.since = root.querySelector('#f-since').value;
    state.filters.topic = root.querySelector('#f-topic').value;
    state.filters.fusion_mode = root.querySelector('#f-mode').value;
    state.filters.disagree_only = root.querySelector('#f-disagree').checked;
  }

  async function loadTopics() {
    try {
      const data = await API.topics();
      state.topics = data.observed || [];
      const select = document.querySelector('#f-topic');
      if (!select) return;
      state.topics.forEach((t) => {
        const option = document.createElement('option');
        option.value = t.topic;
        option.textContent = `${t.topic} (${t.count})`;
        select.appendChild(option);
      });
    } catch (e) { /* the filter degrades to "all topics" */ }
  }

  async function refresh() {
    try {
      const params = Object.assign({}, state.filters, {
        limit: 60, sort: state.sort, order: state.order,
      });
      params.min_confidence = params.min_confidence || undefined;
      params.disagree_only = params.disagree_only || undefined;
      const data = await API.alerts(params);
      state.items = data.items;
      state.total = data.total_matching;
      draw();
    } catch (error) {
      const wrap = document.querySelector('#table-wrap');
      if (wrap) {
        wrap.innerHTML = `<div class="empty">
          <p><strong>Could not load alerts.</strong></p>
          <p class="small">${FMT.esc(error.message)}</p>
          ${error.status === 503
            ? '<p class="small">Start the consumer first:<br><span class="mono">python -m demo.run_consumer --mode offline</span><br>or seed the database:<br><span class="mono">python scripts/seed_demo_db.py</span></p>'
            : ''}
        </div>`;
      }
    }
  }

  function probaFor(item) {
    if (state.branch === 'tabular') return item.p_tab;
    if (state.branch === 'image') return item.p_img;
    return item.p_fused;
  }

  function classFor(item) {
    const proba = probaFor(item);
    if (!proba) return null;
    return Object.entries(proba).sort((a, b) => b[1] - a[1])[0][0];
  }

  function draw() {
    const wrap = document.querySelector('#table-wrap');
    const count = document.querySelector('#result-count');
    if (!wrap) return;
    if (count) {
      count.textContent = `${FMT.int(state.total)} matching · showing ${state.items.length}`;
    }

    if (!state.items.length) {
      wrap.innerHTML = `<div class="empty">
        No alerts match these filters.
        <p class="small">If the stream is idle, that may simply be daylight at
        Palomar — ZTF observes at night.</p></div>`;
      return;
    }

    const header = (key, label, sortable = true) => {
      const active = state.sort === key;
      const arrow = active ? (state.order === 'desc' ? ' ▾' : ' ▴') : '';
      return `<th class="${sortable ? '' : 'no-sort'}" ${sortable ? `data-sort="${key}"` : ''}>
        ${label}<span class="arrow">${arrow}</span></th>`;
    };

    const branchLabel = { fused: 'Fused', tabular: 'Tabular', image: 'Image' }[state.branch];

    wrap.innerHTML = `
      <table class="grid">
        <thead><tr>
          <th class="no-sort">Object ID</th>
          <th class="no-sort">Topic</th>
          ${header('received', 'Received (UTC)')}
          <th class="no-sort">RA / Dec</th>
          <th class="no-sort">Filter</th>
          ${header('magpsf', 'Mag')}
          <th class="no-sort">${branchLabel} class</th>
          ${header('confidence', 'Confidence')}
          <th class="no-sort">Agreement</th>
          <th class="no-sort">Latency</th>
        </tr></thead>
        <tbody>
          ${state.items.map(rowHtml).join('')}
        </tbody>
      </table>`;

    wrap.querySelectorAll('th[data-sort]').forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        if (state.sort === key) {
          state.order = state.order === 'desc' ? 'asc' : 'desc';
        } else {
          state.sort = key; state.order = 'desc';
        }
        refresh();
      });
    });
    wrap.querySelectorAll('tbody tr').forEach((tr) => {
      tr.addEventListener('click', () => App.navigate(`#/object/${tr.dataset.oid}?candid=${tr.dataset.candid}`));
    });
  }

  function rowHtml(item) {
    const cls = classFor(item);
    const proba = probaFor(item);
    const confidence = proba && cls ? proba[cls] : null;
    let agreement = '<span class="muted">—</span>';
    if (item.fusion_mode === 'both') {
      agreement = item.branch_disagree
        ? '<span class="tag tag-warn" title="The two modalities picked different classes">disagree</span>'
        : '<span class="tag" title="Both modalities picked the same class">agree</span>';
    } else if (item.fusion_mode) {
      agreement = `<span class="tag tag-info" title="Only one modality was available">${FMT.esc(item.fusion_mode.replace('_', ' '))}</span>`;
    }

    // broker_to_classified is real latency for a live alert, but for one pulled
    // off a days-old backlog it is queue age. Showing "787085.9s" under a
    // "Latency" heading would be true and misleading at the same time, so
    // anything over an hour is relabelled as what it actually is.
    const brokerMs = item.latency && item.latency.broker_to_classified_ms;
    const pipelineMs = item.latency && item.latency.pipeline_ms;
    let latency;
    if (brokerMs && brokerMs > 3.6e6) {
      const days = brokerMs / 86.4e6;
      latency = `<span class="muted" title="Time between the broker timestamp and classification.
This alert came off the backlog, so this is queue age, not system latency.
In-pipeline processing took ${pipelineMs ? pipelineMs.toFixed(0) : '?'} ms.">queued ${days.toFixed(1)}d</span>`;
    } else if (brokerMs) {
      latency = `${(brokerMs / 1000).toFixed(1)}s`;
    } else if (pipelineMs) {
      latency = `${pipelineMs.toFixed(0)}ms`;
    } else {
      latency = '—';
    }

    return `<tr data-oid="${FMT.esc(item.object_id)}" data-candid="${item.candid}">
      <td class="oid">${FMT.esc(item.object_id)} ${FMT.trainingBadge(item.known_label)}</td>
      <td class="small muted">${FMT.esc(item.topic || item.source)}</td>
      <td class="small nowrap" title="${FMT.esc(FMT.time(item.received_utc))}">${FMT.ago(item.received_utc)}</td>
      <td class="mono small">${FMT.ra(item.ra)} ${FMT.dec(item.dec)}</td>
      <td>ZTF-${FMT.esc(item.band)}</td>
      <td class="num">${FMT.num(item.magpsf, 2)}</td>
      <td>${FMT.classChip(cls)}</td>
      <td class="nowrap">${FMT.confBar(confidence, cls)}</td>
      <td>${agreement}</td>
      <td class="num small" title="broker timestamp to classified">${latency}</td>
    </tr>`;
  }

  async function loadStats() {
    const panel = document.querySelector('#stats-panel');
    if (!panel) return;
    try {
      const s = await API.stats();
      const classEntries = FMT.CLASSES.map((c) => [c, s.by_class[c] || 0]);
      const p = s.latency.percentiles || {};
      panel.innerHTML = `
        <h2>Statistics</h2>
        <div class="small muted">${FMT.int(s.totals.alerts)} alerts ·
          ${FMT.int(s.totals.objects)} objects</div>
        <h4 style="margin-top:.8rem">Predicted class</h4>
        ${PLOT.bars(classEntries, { colour: (l) => FMT.classColour(l), height: 76 })}
        <h4 style="margin-top:.8rem">Confidence</h4>
        ${PLOT.bars(
          s.confidence_histogram.counts.map((v, i) => [`${(i / 10).toFixed(1)}`, v]),
          { height: 60 }
        )}
        <h4 style="margin-top:.8rem">Latency</h4>
        <table class="kv">
          <tr><th>pipeline p50</th><td>${FMT.num((p.t_pipeline_ms || {}).p50, 1)} ms</td></tr>
          <tr><th>pipeline p95</th><td>${FMT.num((p.t_pipeline_ms || {}).p95, 1)} ms</td></tr>
          <tr><th>broker→class p95</th><td>${FMT.num(((p.t_broker_to_classified_ms || {}).p95 || 0) / 1000, 1)} s</td></tr>
        </table>
        <div class="callout small">${FMT.esc(s.class_prior_caveat)}</div>`;
    } catch (e) {
      panel.innerHTML = '<h2>Statistics</h2><div class="muted small">unavailable</div>';
    }
  }

  function start() {
    stop();
    state.timer = setInterval(() => {
      if (document.visibilityState === 'visible') refresh();
    }, POLL_MS);
  }

  function stop() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }

  return { render, stop };
})();
