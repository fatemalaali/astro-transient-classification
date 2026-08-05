/* Bootstrap, hash routing, the live indicator and the modal.
 *
 * Hash routing rather than a router library: three routes do not justify a
 * dependency, and `#/object/ZTF…` survives a page reload, which matters when a
 * demo needs to jump straight back to a specific object.
 */
const App = (() => {
  const HEALTH_MS = 2000;
  let healthTimer = null;

  function parseRoute() {
    const hash = window.location.hash || '#/';
    const [path, query] = hash.slice(1).split('?');
    const params = new URLSearchParams(query || '');
    const parts = path.split('/').filter(Boolean);
    if (!parts.length) return { name: 'stream', params };
    if (parts[0] === 'methodology') return { name: 'methodology', params };
    if (parts[0] === 'object' && parts[1]) {
      return { name: 'object', oid: decodeURIComponent(parts[1]), params };
    }
    return { name: 'stream', params };
  }

  function navigate(hash) {
    if (window.location.hash === hash) route();
    else window.location.hash = hash;
  }

  function route() {
    const view = document.querySelector('#view');
    const current = parseRoute();

    StreamView.stop();
    document.querySelectorAll('.tab').forEach((tab) => {
      const active = (current.name === 'methodology' && tab.dataset.route === 'methodology')
        || (current.name !== 'methodology' && tab.dataset.route === 'stream');
      tab.classList.toggle('active', active);
    });

    if (current.name === 'methodology') Method.render(view);
    else if (current.name === 'object') {
      ObjectView.render(view, current.oid, current.params.get('candid'));
    } else StreamView.render(view);
  }

  /* ------------------------------------------------------- live indicator */
  async function pollHealth() {
    const badge = document.querySelector('#live-badge');
    const detail = document.querySelector('#live-detail');
    if (!badge) return;
    try {
      const h = await API.health();
      // Green is reserved for a genuine push stream. A polled or replayed
      // source never earns it, so nobody can mistake a recording for live data.
      let cls = 'badge-unknown';
      if (h.is_live_stream) cls = 'badge-live';
      else if (h.badge === 'CONSUMER DOWN' || h.badge === 'DISCONNECTED' || h.badge === 'NO CONSUMER') cls = 'badge-down';
      else if (h.connected) cls = 'badge-degraded';
      badge.className = `badge ${cls}`;
      badge.textContent = h.badge;

      const totalLag = h.total_lag;
      const bits = [];
      if (h.topics && h.topics.length) {
        bits.push(`${h.topics.length} topic${h.topics.length > 1 ? 's' : ''}`);
      }
      if (totalLag !== null && totalLag !== undefined) {
        bits.push(`<span class="lag">lag ${FMT.int(totalLag)}</span>
                   ${PLOT.sparkline(h.lag_sparkline, { colour: '#9fb0c4' })}`);
      }
      if (h.seconds_since_last_alert !== null && h.seconds_since_last_alert !== undefined) {
        bits.push(`last alert ${FMT.duration(h.seconds_since_last_alert)} ago`);
      }
      if (h.queue_depth) bits.push(`queue ${h.queue_depth}`);
      if (h.dropped_total) bits.push(`<span style="color:#fbbf24">dropped ${h.dropped_total}</span>`);
      if (h.decode_failures) bits.push(`<span style="color:#fbbf24">decode fails ${h.decode_failures}</span>`);

      let note = '';
      if (!h.palomar_is_night) {
        note = `<div title="${FMT.esc(h.palomar_note || '')}">
          Palomar ${FMT.esc(h.palomar_local_time)} — daytime, gaps expected</div>`;
      } else {
        note = `<div>Palomar ${FMT.esc(h.palomar_local_time)} — observing hours</div>`;
      }
      detail.innerHTML = bits.join(' · ') + note +
        (h.error ? `<div style="color:#f87171">${FMT.esc(h.error)}</div>` : '');
    } catch (error) {
      badge.className = 'badge badge-down';
      badge.textContent = 'API UNREACHABLE';
      detail.textContent = error.message;
    }
  }

  /* ------------------------------------------------------------------ modal */
  function openModal(title, html) {
    document.querySelector('#modal-title').textContent = title;
    document.querySelector('#modal-body').innerHTML = html;
    document.querySelector('#modal').classList.remove('hidden');
  }

  function closeModal() {
    document.querySelector('#modal').classList.add('hidden');
  }

  /* ------------------------------------------------------------------ init */
  async function init() {
    window.addEventListener('hashchange', route);
    document.querySelector('#modal-close').addEventListener('click', closeModal);
    document.querySelector('#modal').addEventListener('click', (e) => {
      if (e.target.id === 'modal') closeModal();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
    });
    document.querySelector('#provenance-more').addEventListener('click', () => {
      openModal('Provenance', '<div id="modal-provenance"><div class="loading">…</div></div>');
      Method.loadProvenance(document.querySelector('#modal-provenance'));
    });

    try {
      const config = await API.config();
      const footer = document.querySelector('#footer-config');
      footer.innerHTML = `mode <span class="mono">${FMT.esc(config.mode)}</span> ·
        ${config.topics.length} configured topic(s) ·
        ALeRCE features ${config.alerce_enabled ? 'enabled' : 'disabled'} ·
        ${config.using_stubs ? '<strong style="color:#b8860b">STUB MODELS</strong>' : 'trained models'} ·
        <span class="mono">${FMT.esc(config.db_path)}</span>`;
      if (config.using_stubs) {
        document.querySelector('#provenance-strip').insertAdjacentHTML('beforeend',
          ' <span class="tag tag-warn">stub models — predictions are placeholders</span>');
      }
    } catch (e) { /* the footer is cosmetic */ }

    route();
    pollHealth();
    healthTimer = setInterval(() => {
      if (document.visibilityState === 'visible') pollHealth();
    }, HEALTH_MS);
  }

  document.addEventListener('DOMContentLoaded', init);
  return { navigate, openModal, closeModal, route };
})();
