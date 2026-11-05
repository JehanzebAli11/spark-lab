// Scans the page, asks the background worker (which calls the local model) and applies the answer.
// Rescans when the page changes, at most every 2 seconds.
(function () {
  let busy = false;
  let pending = false;

  async function check() {
    if (busy) { pending = true; return; }
    busy = true;
    try {
      const req = window.SparkGuard.buildRequest(document);
      if (!req.candidates.length && !req.has_password_field) return;
      const res = await chrome.runtime.sendMessage({type: 'page-check', request: req});
      if (res && !res.error) {
        const done = window.SparkGuard.applyVerdicts(document, res, res.threshold);
        if (done.length) chrome.runtime.sendMessage({type: 'log', url: location.href, done});
      }
    } catch (e) {
      // The extension keeps working without the model; it just does nothing.
    } finally {
      busy = false;
      if (pending) { pending = false; setTimeout(check, 2000); }
    }
  }

  check();
  let timer = null;
  new MutationObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(check, 2000);
  }).observe(document.documentElement, {childList: true, subtree: true});
})();
