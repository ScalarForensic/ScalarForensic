// ── The JS harness: execute the frontend, do not read it ─────────────────────
//
// WHY THIS EXISTS.  `player.js` once shipped a `?? … ||` precedence
// SyntaxError — the file did not parse at all — and all fourteen text-level
// wiring tests in `tests/test_video_playback.py` passed against it.  A wiring
// test opens a file and greps it; it cannot tell you a browser can run it.
// Everything here goes through `vm.Script`, which *compiles* and then *runs*
// the source, so a parse error is a test failure and not a green line.
//
// WHY `node:test` AND NOT VITEST/JEST.  Zero dependencies.  This project ships
// to an isolated LAN (`docs/deployment.md`); a test dependency with a
// several-hundred-package tree is a real airgap cost, and none of it buys
// anything here.  The frontend is plain browser scripts that register object
// fragments on `window.__sfnParts` — no bundler, no JSX, no module graph — so
// a `vm` context with a `window` object is a faithful enough browser, and the
// parts under test touch no DOM.  `package.json` therefore has no
// `dependencies` at all: see it, and this paragraph, as one statement.
//
// WHAT IT DOES NOT COVER.  There is no DOM.  Markup — the `.vc-block`
// template, Alpine directives, CSS — is still wiring-pinned only, and
// rendering it needs a full analysis session with Qdrant and an indexed
// corpus.  This harness pins the *script* half: that it parses, and that its
// methods and getters compute what they claim to.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

export const REPO_ROOT = path.resolve(fileURLToPath(new URL('.', import.meta.url)), '..', '..');
const STATIC_DIR = path.join(REPO_ROOT, 'src', 'scalar_forensic', 'web', 'static');
const INDEX_HTML = path.join(STATIC_DIR, 'index.html');

// The load list comes out of `index.html` rather than being restated here, so
// the harness runs exactly what the browser runs, in the browser's order, and
// a new part file is covered the moment its <script> tag lands.  A part list
// maintained in two places would drift, and the copy in the test would be the
// one that stays green.
export function scriptPaths() {
  const html = fs.readFileSync(INDEX_HTML, 'utf8');
  const srcs = [...html.matchAll(/<script\s+src="\/static\/([^"]+)"/g)].map((m) => m[1]);
  assert.ok(srcs.length > 0, `no local <script src> found in ${INDEX_HTML}`);
  return srcs.map((src) => path.join(STATIC_DIR, src));
}

// A browser-ish global set.  Deliberately small: anything a part file needs at
// load time that is missing here throws, which is information — it means the
// part reached for a browser API outside this list and the harness should be
// extended on purpose rather than by accident.
export function browserContext() {
  // `listeners` records every registration so a test can assert that a handler
  // was added while a job ran and removed when it ended, and can invoke it.
  // Recording rather than ignoring: a stub that swallowed addEventListener would
  // have handed us a green `beforeunload` test asserting nothing, which is the
  // same defect class as the wiring tests that passed on an unparseable file.
  const listeners = new Map();
  const window = {
    __sfnParts: [],
    listeners,
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    removeEventListener(type, fn) {
      const fns = listeners.get(type);
      if (!fns) return;
      const i = fns.indexOf(fn);
      if (i !== -1) fns.splice(i, 1);
    },
  };
  const sandbox = {
    window,
    console,
    setInterval,
    clearInterval,
    setTimeout,
    clearTimeout,
    queueMicrotask,
    URL,
    URLSearchParams,
    encodeURIComponent,
    decodeURIComponent,
    fetch: async () => {
      throw new Error('fetch() was called without a stub; set component._fetch in the test');
    },
  };
  window.window = window;
  return vm.createContext(sandbox);
}

// Compile and run one script in `context`.  `new vm.Script(...)` is the whole
// point of the harness: it throws SyntaxError on malformed source, with the
// real filename and line, before a single assertion runs.
export function runScript(file, context) {
  const source = fs.readFileSync(file, 'utf8');
  new vm.Script(source, { filename: path.relative(REPO_ROOT, file) }).runInContext(context);
}

// Assemble the real `sfn()` component: every part file the page loads, merged
// by the real assembler in `static/app.js` (which is the last <script> in the
// list).  Using the shipped assembler rather than a copy means the merge rule
// itself — property descriptors, never Object.assign — is under test.
// `context` comes back too: it is the global object the parts closed over, so
// a test that needs to stub `fetch` assigns `context.fetch` and the component's
// methods see it.
export function loadFrontend() {
  const context = browserContext();
  for (const file of scriptPaths()) runScript(file, context);
  assert.equal(typeof context.sfn, 'function', 'static/app.js did not define sfn()');
  return { context, component: context.sfn() };
}

export function loadComponent() {
  return loadFrontend().component;
}
