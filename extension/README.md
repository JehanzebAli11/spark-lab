# Spark Guard Chrome extension (dev shell)

A Manifest V3 extension that hides ads and clutter, rejects cookie banners and warns on phishing
login pages. The decisions come from a model running on your own machine: today a rule-based
stub (`guard/extension_stub`), later the guard model served by the Spark.

## Try it

1. Start the local endpoint: `python -m guard.extension_stub` (listens on `127.0.0.1:8765`).
2. Open `chrome://extensions`, turn on Developer mode, click **Load unpacked** and pick this folder.
3. Browse. The service worker console (on the extension's details page) logs what was done.

The options page sets the endpoint, the confidence threshold and whether screenshots are sent
(off by default).

## How it works

- **`core.js`** finds candidates on the page: ad-like elements, iframes, fixed cookie banners
  and large overlays, each with its buttons, text (cut to 300 characters) and box. It also
  collects page facts: URL, title, whether there's a password field. It then applies the
  verdicts: hide, click, or show a warning banner. It has no `chrome.*` calls, so tests load it
  into any page.
- **`content.js`** runs the scan, sends it to the background worker, applies the answer, and
  rescans when the page changes (at most every 2 seconds).
- **`background.js`** POSTs the scan to the endpoint and refuses any endpoint that isn't
  localhost. It can attach a screenshot.

The request and response shapes are documented in `guard/extension_stub/server.py`; a real model
server must keep them.

## Privacy

- Form and password values are never read.
- Page data only goes to an endpoint on this machine.
- Screenshots stay off unless you turn them on.
