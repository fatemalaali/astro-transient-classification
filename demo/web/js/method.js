/* Methodology showcase — the view the viva runs on.
 *
 * Four things, in the order they get presented:
 *   1. the architecture trace of one alert, with the intermediate numbers;
 *   2. the disagreement gallery, surfaced automatically;
 *   3. the provenance statement, served by the API rather than typed in HTML;
 *   4. the held-out evaluation table, with its significance caveat.
 */
const Method = (() => {
  const state = { traceCandid: null, disagreements: [] };

  async function render(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="panel-head">
          <h2>Architecture — one alert, end to end</h2>
          <span class="count" id="trace-subject"></span>
          <span style="margin-left:auto" class="count muted">
            click a stage to see its numbers</span>
        </div>
        <div id="m-trace"><div class="loading">…</div></div>
      </div>

      <div class="layout" style="grid-template-columns: 1fr 1fr">
        <div>
          <div class="panel">
            <div class="panel-head"><h2>Branch disagreements</h2>
              <span class="count" id="disagree-count"></span></div>
            <div class="small muted" style="margin-bottom:.5rem">
              Cases where the light-curve and image branches picked different
              classes, sharpest contradiction first. These are where late fusion
              has something to resolve.
            </div>
            <div id="m-disagree"><div class="loading">…</div></div>
          </div>
        </div>
        <div>
          <div class="panel">
            <h2>Held-out evaluation</h2>
            <div id="m-eval"><div class="loading">…</div></div>
          </div>
          <div class="panel">
            <h2>Models</h2>
            <div id="m-models"><div class="loading">…</div></div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>Provenance</h2>
        <div id="m-provenance"><div class="loading">…</div></div>
      </div>`;

    loadTrace();
    loadDisagreements();
    loadEvaluation();
    loadModels();
    loadProvenance();
  }

  /* ---------------------------------------------------------------- trace */
  async function loadTrace(candid) {
    const host = document.querySelector('#m-trace');
    if (!host) return;
    try {
      let target = candid;
      if (!target) {
        const recent = await API.alerts({ limit: 1 });
        if (!recent.items.length) {
          host.innerHTML = '<div class="empty">No alerts yet.</div>';
          return;
        }
        target = recent.items[0].candid;
      }
      state.traceCandid = target;
      const data = await API.trace(target);
      renderTrace(host, data);
      const subject = document.querySelector('#trace-subject');
      if (subject) {
        subject.innerHTML = `<a href="#/object/${FMT.esc(data.object_id)}?candid=${target}"
          class="mono">${FMT.esc(data.object_id)}</a> ·
          ${FMT.esc(data.topic || '')} · split_id ${FMT.esc(data.split_id || '—')}`;
      }
    } catch (error) {
      host.innerHTML = `<div class="empty small">${FMT.esc(error.message)}</div>`;
    }
  }

  /* Shared with the object view, so the trace looks the same in both places. */
  function renderTrace(host, data) {
    const stages = data.stages || [];
    if (!stages.length) {
      host.innerHTML = '<div class="empty small">No trace recorded for this alert.</div>';
      return;
    }

    const hint = (stage) => {
      const d = stage.detail || {};
      if (stage.id === 'kafka' || stage.id === 'ingest') {
        return d.offset != null ? `p${d.partition}@${d.offset}` : (d.source || '');
      }
      if (stage.id === 'avro') return `${d.cutouts_present ?? 0}/3 cutouts`;
      if (stage.id === 'normalise') return `${d.n_detections ?? 0} det`;
      if (stage.id === 'bogus') return 'skipped';
      if (stage.id === 'features') return `${d.n_present ?? 0}/${d.n_expected ?? 0}`;
      if (stage.id === 'tabular' || stage.id === 'image') {
        return d.calibrated ? argmaxLabel(d.calibrated) : (stage.ok ? '' : 'n/a');
      }
      if (stage.id === 'fusion') return d.mode || '';
      if (stage.id === 'result') return d.class ? `${d.class} ${Number(d.confidence || 0).toFixed(2)}` : '—';
      return '';
    };

    host.innerHTML = `
      <div class="trace">
        ${stages.map((stage, i) => `
          ${i ? '<span class="stage-arrow">→</span>' : ''}
          <div class="stage ${stage.skipped ? 'skipped' : ''} ${stage.ok === false ? 'failed' : ''}"
               data-index="${i}">
            <div class="n">stage ${i + 1}</div>
            <div class="label">${FMT.esc(stage.label)}</div>
            <div class="hint">${FMT.esc(hint(stage))}</div>
          </div>`).join('')}
      </div>
      <div class="stage-detail" id="stage-detail"></div>`;

    const cards = host.querySelectorAll('.stage');
    const show = (index) => {
      cards.forEach((c) => c.classList.toggle('selected', Number(c.dataset.index) === index));
      const detail = host.querySelector('#stage-detail');
      const stage = stages[index];
      const rows = Object.entries(stage.detail || {})
        .filter(([, v]) => v !== null && v !== undefined && v !== '')
        .map(([k, v]) => `<tr><th>${FMT.esc(k)}</th><td>${FMT.esc(
          Array.isArray(v) ? `[${v.map((x) => (typeof x === 'number' ? x.toFixed(4) : x)).join(', ')}]`
            : (typeof v === 'object' ? JSON.stringify(v) : v))}</td></tr>`)
        .join('') || '<tr><td class="muted">no detail recorded</td></tr>';
      detail.innerHTML = `
        <h4>${FMT.esc(stage.label)}${stage.skipped ? ' — skipped' : ''}</h4>
        <table class="kv">${rows}</table>`;
    };
    cards.forEach((card) => card.addEventListener('click', () => show(Number(card.dataset.index))));
    // Open on fusion: the stage examiners ask about.
    const fusionIndex = stages.findIndex((s) => s.id === 'fusion');
    show(fusionIndex >= 0 ? fusionIndex : stages.length - 1);
  }

  function argmaxLabel(vector) {
    if (!Array.isArray(vector) || !vector.length) return '';
    let best = 0;
    vector.forEach((v, i) => { if (v > vector[best]) best = i; });
    return `${FMT.CLASSES[best]} ${Number(vector[best]).toFixed(2)}`;
  }

  /* -------------------------------------------------------- disagreements */
  async function loadDisagreements() {
    const host = document.querySelector('#m-disagree');
    if (!host) return;
    try {
      const data = await API.disagreements({ limit: 40, sort: 'margin' });
      state.disagreements = data.items;
      const count = document.querySelector('#disagree-count');
      if (count) {
        const rate = data.disagreement_rate != null
          ? ` (${(data.disagreement_rate * 100).toFixed(1)}% of two-branch alerts)` : '';
        count.textContent = `${FMT.int(data.total_disagreements)}${rate}`;
      }
      if (!data.items.length) {
        host.innerHTML = `<div class="empty small">
          No disagreements yet. Both branches have agreed on every alert
          classified with both modalities available.</div>`;
        return;
      }
      host.innerHTML = data.items.map((item) => `
        <div class="disagree-row" data-oid="${FMT.esc(item.object_id)}"
             data-candid="${item.candid}">
          <span class="oid">${FMT.esc(item.object_id)}</span>
          <span class="summary">
            ${FMT.classChip(item.tabular_class)} vs ${FMT.classChip(item.image_class)}
            → ${FMT.classChip(item.predicted_class)}
            ${item.fusion_flips ? '<span class="tag tag-warn">fusion flipped</span>' : ''}
            ${item.known_coarse ? `<span class="tag tag-info">truth ${FMT.esc(item.known_coarse)}</span>` : ''}
          </span>
          <span class="small muted" style="margin-left:auto">${FMT.ago(item.received_utc)}</span>
        </div>`).join('');
      host.querySelectorAll('.disagree-row').forEach((row) => {
        row.addEventListener('click', () =>
          App.navigate(`#/object/${row.dataset.oid}?candid=${row.dataset.candid}`));
      });
    } catch (error) {
      host.innerHTML = `<div class="empty small">${FMT.esc(error.message)}</div>`;
    }
  }

  /* ----------------------------------------------------------- evaluation */
  async function loadEvaluation() {
    const host = document.querySelector('#m-eval');
    if (!host) return;
    try {
      const data = await API.evaluation();
      if (!data.rows.length) {
        host.innerHTML = '<div class="empty small">No evaluation summary stored.</div>';
        return;
      }
      const best = Math.max(...data.rows.map((r) => r.macro_f1 || 0));
      host.innerHTML = `
        <table class="grid metrics">
          <thead><tr>
            <th class="no-sort">Scope</th><th class="no-sort">macro-F1</th>
            <th class="no-sort">bal. acc.</th><th class="no-sort">accuracy</th>
            <th class="no-sort">log-loss</th>
          </tr></thead>
          <tbody>
            ${data.rows.map((r) => `
              <tr class="${r.macro_f1 === best ? 'best' : ''}" style="cursor:default">
                <td>${FMT.esc(r.label || r.scope)}</td>
                <td class="num">${FMT.num(r.macro_f1, 4)}</td>
                <td class="num">${FMT.num(r.balanced_accuracy, 4)}</td>
                <td class="num">${FMT.num(r.accuracy, 4)}</td>
                <td class="num">${FMT.num(r.log_loss, 4)}</td>
              </tr>`).join('')}
          </tbody>
        </table>
        <div class="callout small"><strong>Caveat.</strong> ${FMT.esc(data.caveat)}</div>
        ${data.significance ? `
          <h4 style="margin-top:.6rem">Significance</h4>
          <table class="kv">
            ${data.significance.map((s) => `
              <tr><th>${FMT.esc(s.comparison)}</th>
                  <td>Δ ${FMT.num(s.delta_macro_f1, 4)}
                      CI [${FMT.num(s.ci_lo, 4)}, ${FMT.num(s.ci_hi, 4)}]
                      ${s.excludes_zero ? '<span class="tag">excludes 0</span>'
                        : '<span class="tag tag-warn">includes 0</span>'}
                      p=${FMT.num(s.mcnemar_p, 3)}</td></tr>`).join('')}
          </table>` : ''}
        <div class="small muted" style="margin-top:.5rem">${FMT.esc(data.note)}</div>`;
    } catch (error) {
      host.innerHTML = `<div class="empty small">${FMT.esc(error.message)}</div>`;
    }
  }

  /* --------------------------------------------------------------- models */
  async function loadModels() {
    const host = document.querySelector('#m-models');
    if (!host) return;
    try {
      const data = await API.models();
      const card = (m, title) => `
        <div class="packet-section" style="margin-bottom:.6rem">
          <h4>${FMT.esc(title)}
            ${m.available ? '' : '<span class="tag tag-danger">not loaded</span>'}
            ${m.stub ? '<span class="tag tag-warn">STUB</span>' : ''}</h4>
          <table class="kv">
            ${Object.entries(m)
              .filter(([k, v]) => !['branch', 'component', 'W', 'b', 'baselines',
                'significance', 'test_metrics', 'blend', 'penalty', 'input_columns',
                'class_names'].includes(k) && v !== null && v !== undefined)
              .map(([k, v]) => `<tr><th>${FMT.esc(k)}</th><td>${FMT.esc(
                typeof v === 'object' ? JSON.stringify(v) : v)}</td></tr>`).join('')}
          </table>
        </div>`;
      host.innerHTML = `
        ${data.using_stubs ? `<div class="callout"><strong>Stub mode.</strong>
          Predictions are seeded placeholders, not model output
          (<span class="mono">DEMO_USE_STUBS=1</span>).</div>` : ''}
        ${card(data.tabular, 'Tabular branch')}
        ${card(data.image, 'Image branch')}
        ${card(data.fusion, 'Fusion head')}
        <div class="small muted">Shared split identity:
          <span class="mono">${FMT.esc(data.split_id || 'n/a')}</span> — every
          component was trained on the same partition.</div>`;
    } catch (error) {
      host.innerHTML = `<div class="empty small">${FMT.esc(error.message)}</div>`;
    }
  }

  /* ----------------------------------------------------------- provenance */
  async function loadProvenance(host) {
    const target = host || document.querySelector('#m-provenance');
    if (!target) return;
    try {
      const data = await API.provenance();
      const list = (items) => `<ul style="margin:.2rem 0 .6rem 1.1rem;padding:0">
        ${items.map((i) => `<li class="small">${FMT.esc(
          typeof i === 'object' ? `${i.name} — ${i.detail}` : i)}</li>`).join('')}</ul>`;
      target.innerHTML = `
        <div class="callout info">${FMT.esc(data.statement)}</div>
        <div class="section-grid">
          <div class="packet-section">
            <h4>Label sources</h4>${list(data.label_sources)}
          </div>
          <div class="packet-section">
            <h4>Brokers supply</h4>${list(data.brokers_supply)}
            <h4>Brokers never supply</h4>${list(data.brokers_never_supply)}
          </div>
          <div class="packet-section">
            <h4>How this is enforced</h4>${list(data.enforcement)}
            <h4>Out of scope</h4>${list(data.out_of_scope)}
          </div>
        </div>`;
    } catch (error) {
      target.innerHTML = `<div class="empty small">${FMT.esc(error.message)}</div>`;
    }
  }

  return { render, renderTrace, loadProvenance };
})();
