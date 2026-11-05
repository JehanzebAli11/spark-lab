// Calls the local model endpoint. Only localhost endpoints are allowed, so page data never leaves
// this machine. Screenshots are off by default.
const DEFAULTS = {endpoint: 'http://127.0.0.1:8765/v1/page-check', screenshots: false, threshold: 0.5};

function isLocal(url) {
  try {
    const h = new URL(url).hostname;
    return h === '127.0.0.1' || h === 'localhost' || h === '[::1]';
  } catch (e) {
    return false;
  }
}

async function settings() {
  return Object.assign({}, DEFAULTS, await chrome.storage.local.get(Object.keys(DEFAULTS)));
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.type === 'log') {
    console.log('Spark Guard', msg.url, msg.done);
    return false;
  }
  if (msg.type !== 'page-check') return false;
  (async () => {
    const s = await settings();
    if (!isLocal(s.endpoint)) return reply({error: 'endpoint must be on localhost'});
    const body = Object.assign({}, msg.request);
    if (s.screenshots && sender.tab) {
      try {
        const png = await chrome.tabs.captureVisibleTab(sender.tab.windowId, {format: 'png'});
        body.screenshot = png.split(',')[1];
      } catch (e) { /* no screenshot permission for this tab */ }
    }
    try {
      const r = await fetch(s.endpoint, {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)});
      const out = await r.json();
      out.threshold = s.threshold;
      reply(out);
    } catch (e) {
      reply({error: String(e)});
    }
  })();
  return true; // keep the channel open for the async reply
});
