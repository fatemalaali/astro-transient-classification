/* Object-detail view.
 *
 * Mirrors the ALeRCE Explorer object page (components verified in
 * github.com/alercebroker/ztf_explorer): a stamps card bound to a selected
 * detection, a light-curve card, a basic-information card, a classifier card
 * and a cross-match card, with a modal showing the raw alert packet.
 *
 * Two changes carry the thesis argument. The classifier card becomes a
 * three-way tabular/image/fused comparison, and the packet modal is split into
 * instrument / broker-derived / system sections instead of one flat table.
 *
 * The Aladin sky view is deliberately not reproduced: Aladin Lite is a CDN
 * script that fetches remote tiles, and both would break the offline guarantee.
 * Coordinates and outbound links stand in for it.
 */
const ObjectView = (() => {
  const state = { oid: null, candid: null, data: null };

  async function render(root, oid, candid) {
    state.oid = oid;
    // Candids stay STRINGS end to end. A ZTF candid is ~3.5e18, past
    // Number.MAX_SAFE_INTEGER, so any numeric conversion here would round it
    // and every lookup would target a neighbouring, non-existent alert.
    state.candid = candid || null;
    root.innerHTML = `<div class="loading">Loading ${FMT.esc(oid)}…</div>`;

    let data;
    try {
      data = await API.object(oid);
    } catch (error) {
      root.innerHTML = `<div class="panel"><div class="empty">
        <p><strong>${FMT.esc(oid)} not found.</strong></p>
        <p class="small">${FMT.esc(error.message)}</p>
        <p><a href="#/">Back to the alert stream</a></p></div></div>`;
      return;
    }
    state.data = data;

    const alert = state.candid
      ? (data.alerts.find((a) => String(a.candid) === String(state.candid)) || data.latest)
      : data.latest;
    state.candid = String(alert.candid);

    root.innerHTML = `
      <div class="panel">
        <div class="panel-head">
          <h2 class="mono">${FMT.esc(oid)}</h2>
          ${FMT.classChip(alert.predicted_class)}
          <span class="count">${FMT.num(alert.confidence, 3)} confidence ·
            ${FMT.esc(alert.fusion_mode || 'n/a')}</span>
          ${FMT.trainingBadge(alert.known_label)}
          <span style="margin-left:auto">
            <a href="#/">&larr; back to stream</a>
          </span>
        </div>
        ${knownLabelCallout(alert)}
      </div>

      <div class="layout" style="grid-template-columns: 1fr 380px">
        <section>
          <div class="panel">
            <div class="panel-head"><h3>Per-branch comparison</h3>
              <span class="count">this is where fusion does or does not do work</span>
            </div>
            <div id="branches"></div>
          </div>

          <div class="panel">
            <div class="panel-head"><h3>Light curve</h3>
              <span class="count" id="lc-note"></span></div>
            <div id="lightcurve"><div class="loading">…</div></div>
          </div>

          <div class="panel">
            <div class="panel-head"><h3>Pipeline trace</h3>
              <span class="count">alert ${state.candid} · click a stage for its numbers</span>
            </div>
            <div id="trace"><div class="loading">…</div></div>
          </div>
        </section>

        <aside>
          <div class="panel">
            <div class="panel-head"><h3>Cutouts</h3>
              <span class="count">${FMT.esc(alert.cutout_status)}</span></div>
            <div id="stamps"></div>
            <div class="small muted" style="margin-top:.5rem">
              63&times;63 pixels, sigmoid stretch. The model consumes the raw
              arrays, not these renderings.
              <a href="/api/alerts/${state.candid}/stamps.npy">download .npy</a>
            </div>
          </div>

          <div class="panel">
            <h3>Basic information</h3>
            <table class="kv">
              <tr><th>RA / Dec</th><td>${FMT.ra(alert.ra)} ${FMT.dec(alert.dec)}</td></tr>
              <tr><th>degrees</th><td>${FMT.num(alert.ra, 6)}, ${FMT.num(alert.dec, 6)}</td></tr>
              <tr><th>filter</th><td>ZTF-${FMT.esc(alert.band)}</td></tr>
              <tr><th>magpsf</th><td>${FMT.num(alert.magpsf, 3)} ± ${FMT.num(alert.sigmapsf, 3)}</td></tr>
              <tr><th>diffmaglim</th><td>${FMT.num(alert.diffmaglim, 2)}</td></tr>
              <tr><th>detections</th><td>${FMT.int(alert.n_det)} (+${FMT.int(alert.n_nondet)} limits)</td></tr>
              <tr><th>emitted</th><td>${FMT.time(alert.emitted_utc)}</td></tr>
              <tr><th>received</th><td>${FMT.time(alert.received_utc)}</td></tr>
              <tr><th>topic</th><td>${FMT.esc(alert.topic || alert.source)}</td></tr>
              <tr><th>offset</th><td>${alert.partition ?? '—'} / ${alert.offset ?? '—'}</td></tr>
              <tr><th>features</th><td>${FMT.esc(alert.feature_provenance || '—')}
                  ${alert.n_features_present != null ? `(${alert.n_features_present})` : ''}</td></tr>
              <tr><th>split_id</th><td>${FMT.esc(alert.split_id || '—')}</td></tr>
            </table>
            <div style="margin-top:.6rem">
              <button class="ghost" id="btn-packet">View alert packet</button>
            </div>
          </div>

          ${brokerCard(alert)}

          ${alert.n_alerts > 1 ? '' : ''}
          <div class="panel">
            <h3>Alerts for this object</h3>
            <div class="small muted">${FMT.int(data.n_alerts)} stored</div>
            <table class="grid" style="margin-top:.4rem">
              <tbody>
                ${data.alerts.slice(0, 12).map((a) => `
                  <tr data-candid="${FMT.esc(a.candid)}" class="${String(a.candid) === state.candid ? 'selected' : ''}">
                    <td class="small">${FMT.ago(a.received_utc)}</td>
                    <td>ZTF-${FMT.esc(a.band)}</td>
                    <td class="num">${FMT.num(a.magpsf, 2)}</td>
                    <td>${FMT.classChip(a.predicted_class)}</td>
                  </tr>`).join('')}
              </tbody>
            </table>
          </div>
        </aside>
      </div>`;

    drawBranches(data.branch_comparison);
    drawStamps(alert);
    loadLightcurve(oid);
    loadTrace(state.candid);

    document.querySelector('#btn-packet').addEventListener('click', () => showPacket(state.candid));
    root.querySelectorAll('tr[data-candid]').forEach((tr) => {
      tr.addEventListener('click', () =>
        App.navigate(`#/object/${oid}?candid=${tr.dataset.candid}`));
    });
  }

  function knownLabelCallout(alert) {
    const known = alert.known_label;
    if (!known || !known.coarse) return '';
    const correct = known.correct;
    const verdict = correct === null ? '' :
      (correct
        ? '<span class="tag" style="color:#1a8754;border-color:#8fd3ae">prediction matches</span>'
        : '<span class="tag tag-danger">prediction differs</span>');
    return `<div class="callout ${correct ? 'info' : ''}">
      <strong>Known classification:</strong> ${FMT.esc(known.coarse)}
      ${known.fine ? `/ ${FMT.esc(known.fine)}` : ''}
      from <strong>${FMT.esc(known.source || 'catalogue')}</strong> ${verdict}
      ${known.in_training_set ? `<br><span class="small">${
        known.training_split === 'test'
          ? `Gold-set object from the <strong>held-out test</strong> fold — the
             deployed models never trained on it, so this prediction is
             out-of-sample.`
          : `This object was <strong>fitted on</strong> (split:
             <strong>${FMT.esc(known.training_split || 'unknown')}</strong>).
             Its prediction is not evidence of generalisation.`
      }</span>` : ''}
      <br><span class="small muted">Label source is spectroscopic or catalogue —
      never a broker classifier.</span>
    </div>`;
  }

  function brokerCard(alert) {
    const meta = alert.broker_meta;
    if (!meta) return '';
    const rows = (obj) => Object.entries(obj || {})
      .filter(([k]) => !k.startsWith('_'))
      .map(([k, v]) => `<tr><th>${FMT.esc(k)}</th><td>${FMT.esc(
        typeof v === 'object' ? JSON.stringify(v) : v)}</td></tr>`).join('');
    const classifications = rows(meta.classifications);
    const crossmatch = rows(meta.crossmatch);
    if (!classifications && !crossmatch) return '';
    return `<div class="panel packet-section broker">
      <h3>Broker-derived <span class="broker-badge">display only</span></h3>
      <div class="note">${FMT.esc(meta.note)}</div>
      ${classifications ? `<h4>Broker classifications</h4>
        <table class="kv">${classifications}</table>` : ''}
      ${crossmatch ? `<h4 style="margin-top:.6rem">Cross-match</h4>
        <table class="kv">${crossmatch}</table>` : ''}
    </div>`;
  }

  function drawBranches(comparison) {
    const host = document.querySelector('#branches');
    if (!host || !comparison) return;
    const sub = {
      tabular: 'LightGBM · 242 ALeRCE features',
      image: 'EfficientNet-B0 · 3×63×63 cutouts',
      fused: 'multinomial logistic stack on log-probabilities',
    };
    host.innerHTML = `
      <div class="branches">
        ${comparison.branches.map((b) => `
          <div class="branch ${b.id === 'fused' ? 'fused' : ''}">
            <h4>${FMT.esc(b.label)}</h4>
            <div class="sub">${sub[b.id] || ''}</div>
            ${b.available
              ? FMT.probRows(b.proba, b.argmax)
              : '<div class="muted small">branch did not run for this alert</div>'}
            <div class="small" style="margin-top:.4rem">
              ${b.argmax ? `argmax <strong>${FMT.esc(b.argmax)}</strong>` : '&nbsp;'}
            </div>
          </div>`).join('')}
      </div>
      ${comparison.branch_disagree ? `
        <div class="callout"><strong>The branches disagree.</strong>
        Tabular and image picked different classes; the fused decision is doing
        real work here. ${comparison.fusion_flips
          ? 'Fusion also chose a class <em>neither</em> branch picked on its own.' : ''}
        </div>` : ''}
      ${comparison.fusion_mode && comparison.fusion_mode !== 'both' ? `
        <div class="callout"><strong>Missing modality:</strong>
        <span class="mono">${FMT.esc(comparison.fusion_mode)}</span>. The surviving
        branch's calibrated output is used unchanged — no imputation, because the
        stack was fitted only on rows where both branches were present.</div>` : ''}`;
  }

  function drawStamps(alert) {
    const host = document.querySelector('#stamps');
    if (!host) return;
    if (!alert.has_stamps) {
      host.innerHTML = `<div class="empty small">No cutouts stored
        (${FMT.esc(alert.cutout_status)}). The image branch could not run.</div>`;
      return;
    }
    host.innerHTML = `<div class="stamps">
      ${['science', 'reference', 'difference'].map((kind) => `
        <div class="stamp"><figure>
          <!-- Not lazy-loaded: the triplet is the point of this card, it is
               always above the fold, and the three PNGs are a few KB each. -->
          <img src="${API.stampUrl(alert.candid, kind)}" alt="${kind} cutout"
               decoding="async">
          <figcaption>${kind}</figcaption>
        </figure></div>`).join('')}
    </div>`;
  }

  async function loadLightcurve(oid) {
    const host = document.querySelector('#lightcurve');
    if (!host) return;
    try {
      const data = await API.lightcurve(oid);
      PLOT.lightcurve(host, data, { width: Math.min(760, host.clientWidth || 720) });
      const note = document.querySelector('#lc-note');
      if (note) note.textContent = data.note;
    } catch (error) {
      host.innerHTML = `<div class="empty small">${FMT.esc(error.message)}</div>`;
    }
  }

  async function loadTrace(candid) {
    const host = document.querySelector('#trace');
    if (!host) return;
    try {
      const data = await API.trace(candid);
      Method.renderTrace(host, data);
    } catch (error) {
      host.innerHTML = `<div class="empty small">${FMT.esc(error.message)}</div>`;
    }
  }

  async function showPacket(candid) {
    App.openModal('Alert packet', '<div class="loading">…</div>');
    try {
      const data = await API.packet(candid);
      const section = (key, extraClass) => {
        const s = data.sections[key];
        const rows = Object.entries(s.fields || {})
          .filter(([k]) => !k.startsWith('_'))
          .map(([k, v]) => `<tr><th>${FMT.esc(k)}</th><td>${FMT.esc(
            typeof v === 'object' && v !== null ? JSON.stringify(v) : v)}</td></tr>`)
          .join('') || '<tr><td class="muted">none</td></tr>';
        return `<div class="packet-section ${extraClass || ''}">
          <h4>${FMT.esc(s.label)}
            ${s.model_eligible
              ? '<span class="tag" style="color:#1a8754;border-color:#8fd3ae">model input</span>'
              : '<span class="broker-badge">not model input</span>'}</h4>
          <div class="note">${FMT.esc(s.note)}</div>
          <table class="kv">${rows}</table>
        </div>`;
      };
      App.openModal(`Alert packet · ${candid}`, `
        <div class="section-grid">
          ${section('instrument')}
          ${section('broker_derived', 'broker')}
          ${section('system')}
        </div>`);
    } catch (error) {
      App.openModal('Alert packet', `<div class="empty">${FMT.esc(error.message)}</div>`);
    }
  }

  return { render };
})();
