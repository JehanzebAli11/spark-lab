const KEYS = {endpoint: 'http://127.0.0.1:8765/v1/page-check', screenshots: false, threshold: 0.5};
chrome.storage.local.get(Object.keys(KEYS)).then((s) => {
  const v = Object.assign({}, KEYS, s);
  document.getElementById('endpoint').value = v.endpoint;
  document.getElementById('screenshots').checked = v.screenshots;
  document.getElementById('threshold').value = v.threshold;
});
document.getElementById('save').addEventListener('click', async () => {
  const endpoint = document.getElementById('endpoint').value.trim();
  const host = (() => { try { return new URL(endpoint).hostname; } catch (e) { return ''; } })();
  const status = document.getElementById('status');
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(host)) {
    status.textContent = 'The endpoint must be on localhost.';
    return;
  }
  await chrome.storage.local.set({
    endpoint,
    screenshots: document.getElementById('screenshots').checked,
    threshold: Math.min(1, Math.max(0, parseFloat(document.getElementById('threshold').value) || 0.5)),
  });
  status.textContent = 'Saved.';
});
