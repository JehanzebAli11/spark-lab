// Spark Guard page logic. No chrome.* APIs here, so tests can load it into any page.
// Privacy: never collects input values; text is cut to 300 characters; the extension only talks to
// an endpoint on this machine (see background.js).
(function (root) {
  const AD_HINT = /(^|[\s_-])(ad|ads|advert|advertisement|sponsor|sponsored|promo|promoted|banner-ad|adslot|dfp)([\s_-]|$)/i;
  const AD_TEXT = /^\s*(advertisement|sponsored|ad|promoted)\s*$/i;
  const COOKIE_TEXT = /\b(cookie|cookies|consent|gdpr|privacy choices|tracking)\b/i;
  const MAX_TEXT = 300;
  let counter = 0;

  function keyOf(el) {
    if (!el.dataset.sgKey) el.dataset.sgKey = 'sg' + (++counter);
    return el.dataset.sgKey;
  }

  function rectOf(el) {
    const r = el.getBoundingClientRect();
    return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)];
  }

  function visible(el) {
    const st = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return st.display !== 'none' && st.visibility !== 'hidden' && r.width > 4 && r.height > 4;
  }

  function textOf(el) {
    return (el.innerText || '').replace(/\s+/g, ' ').trim().slice(0, MAX_TEXT);
  }

  function hintFor(el) {
    const attrs = [el.id, typeof el.className === 'string' ? el.className : '', el.getAttribute('aria-label') || '',
      el.getAttribute('data-ad') !== null ? 'ad' : ''].join(' ');
    if (el.tagName === 'IFRAME') return 'iframe';
    if (AD_HINT.test(attrs) || AD_TEXT.test(textOf(el).slice(0, 20))) return 'ad_hint';
    const st = getComputedStyle(el);
    if ((st.position === 'fixed' || st.position === 'sticky') && COOKIE_TEXT.test(textOf(el))) return 'cookie_hint';
    if (st.position === 'fixed' || st.position === 'sticky') {
      const r = el.getBoundingClientRect();
      if (r.width * r.height > 0.15 * innerWidth * innerHeight) return 'overlay';
    }
    return null;
  }

  function findCandidates(doc) {
    doc = doc || document;
    const out = [];
    const seen = new Set();
    for (const el of doc.querySelectorAll('body *')) {
      if (out.length >= 60) break;
      if (el.dataset.sgHidden || !visible(el)) continue;
      const hint = hintFor(el);
      if (!hint) continue;
      if ([...seen].some((p) => p.contains(el))) continue; // keep the outermost candidate only
      seen.add(el);
      const buttons = [...el.querySelectorAll('button, a, [role=button], input[type=button], input[type=submit]')]
        .slice(0, 8).map((b) => ({key: keyOf(b), text: (b.innerText || b.value || '').trim().slice(0, 60)}));
      out.push({
        key: keyOf(el), tag: el.tagName.toLowerCase(), hint, text: textOf(el), rect: rectOf(el), buttons,
        attrs: {id: el.id || '', class: (typeof el.className === 'string' ? el.className : '').slice(0, 120),
                src: el.getAttribute('src') || '', href: el.getAttribute('href') || ''},
      });
    }
    return out;
  }

  function pageInfo(doc) {
    doc = doc || document;
    return {
      url: location.href, title: doc.title.slice(0, 200),
      viewport: {w: innerWidth, h: innerHeight},
      has_password_field: !!doc.querySelector('input[type=password]'),
      form_actions: [...doc.querySelectorAll('form')].map((f) => f.getAttribute('action') || '').slice(0, 5),
      headings: [...doc.querySelectorAll('h1, h2')].map((h) => textOf(h).slice(0, 80)).slice(0, 5),
    };
  }

  function buildRequest(doc) {
    return Object.assign(pageInfo(doc), {candidates: findCandidates(doc)});
  }

  function showBanner(doc, page) {
    if (doc.getElementById('spark-guard-warning')) return;
    const b = doc.createElement('div');
    b.id = 'spark-guard-warning';
    b.setAttribute('role', 'alert');
    Object.assign(b.style, {position: 'fixed', top: '0', left: '0', right: '0', zIndex: 2147483647,
      background: '#b91c1c', color: '#fff', font: '600 15px system-ui, sans-serif', padding: '12px 16px'});
    b.textContent = 'Spark Guard: this page may be a phishing page' + (page.brand ? ' imitating ' + page.brand : '') +
      '. Do not enter passwords, card numbers or seed phrases. (' + Math.round(100 * page.phishing_p) + '%)';
    doc.body.prepend(b);
  }

  // Apply the model's verdicts. Returns a log of what was done, for the popup and for tests.
  function applyVerdicts(doc, response, threshold) {
    doc = doc || document;
    threshold = threshold == null ? 0.5 : threshold;
    const done = [];
    for (const v of (response.verdicts || [])) {
      if (v.p < threshold) continue;
      const el = doc.querySelector('[data-sg-key="' + v.key + '"]');
      if (!el) continue;
      if (v.action === 'hide') {
        el.style.setProperty('display', 'none', 'important');
        el.dataset.sgHidden = v.kind;
        done.push({key: v.key, did: 'hid ' + v.kind});
      } else if (v.action === 'click' && v.click_key) {
        const btn = doc.querySelector('[data-sg-key="' + v.click_key + '"]');
        if (btn) { btn.click(); done.push({key: v.key, did: 'clicked ' + v.click_key}); }
      }
    }
    const page = response.page || {};
    if ((page.phishing_p || 0) >= threshold) { showBanner(doc, page); done.push({key: 'page', did: 'warned'}); }
    return done;
  }

  root.SparkGuard = {findCandidates, pageInfo, buildRequest, applyVerdicts, AD_HINT, COOKIE_TEXT};
})(typeof window !== 'undefined' ? window : globalThis);
