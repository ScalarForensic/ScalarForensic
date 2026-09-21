import { test, expect } from '@playwright/test';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';

const staticRoot = resolve('src/scalar_forensic/web/static');
const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=', 'base64');
const hit = { is_video_frame: true, path: '/images/frame.jpg', image_hash: 'a'.repeat(64), scores: { exact: 1 }, video_path: '/images/source.mov', video_hash: 'b'.repeat(64), frame_timecode_ms: 0 };

test('synthetic upload renders hits, face selection/compare and playback failure download', async ({ page }) => {
  const errors = [];
  const calls = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('http://sfn.test/**', async route => {
    const path = new URL(route.request().url()).pathname;
    calls.push(path);
    if (!path.startsWith('/api/')) {
      const file = path === '/' ? 'index.html' : path.replace(/^\/static\//, '');
      const contentType = file.endsWith('.js') ? 'application/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.html') ? 'text/html' : 'image/svg+xml';
      return route.fulfill({ body: await readFile(resolve(staticRoot, file)), contentType });
    }
    if (/image|thumbnail|chip|thumb/.test(path) && !path.includes('by-image') && !path.includes('meta')) {
      return route.fulfill({ body: pixel, contentType: 'image/png' });
    }
    if (path === '/api/analyze') return route.fulfill({ contentType: 'text/event-stream', body: 'data: {"type":"done","session_id":"synthetic"}\n\n' });
    const responses = {
      '/api/collections': { modes: ['exact'], has_reference: false },
      '/api/faces/availability': { faces_available: true },
      '/api/query': { results: [{ file_id: 'probe', filename: 'probe.png', hits: [hit], errors: [] }] },
      '/api/faces/query-faces': { detection_token: 'synthetic-detection', embedder: { embedder_model_name: 'query-test-model' }, faces: [{ index: 0, searchable: true, det_conf: 0.99, chip_url: '/api/chip/probe', bbox: [0, 0, 1, 1] }] },
      [`/api/faces/by-image/${hit.image_hash}`]: { faces: [{ id: 'face1', observation_key: 'synthetic', review_chip_hash: 'c'.repeat(64), searchable: true, det_conf: 0.99 }] },
      '/api/faces/search': { hits: [{ image_hash: hit.image_hash, image_path: hit.path, score: 0.9, face: { point_id: 'face1' }, n_collapsed: 3 }], embedder: { embedder_model_name: 'search-test-model' }, compat: { ok: true, warnings: ['Search library version differs'] } },
      '/api/faces/compare': { pairs: [{ query_face_index: 0, point_id: 'face1', score: 0.9 }], n_query_comparable: 1, n_query_review_only: 2, n_match_comparable: 1, n_match_review_only: 3, embedder: { embedder_model_name: 'compare-test-model' }, compat: { ok: true, warnings: ['Comparison library version differs'] } },
      '/api/faces/audit': { n_observations: 0, detector: {}, embedder: {},
        gates_in_force_at_index_time: {}, library_versions: {}, pipeline_config_hash: null,
        file_totals: { n_detected: 0, n_kept: 0, n_review_only: 0, review_only_reasons: {}, rejected: {} },
        enablement: null, compat: { ok: null, warnings: ['No surviving observation provenance'] }, caveat: '' },
      '/api/video-playback-info': { source_path: '/images/source.mov', filename: 'source.mov', mode: 'original', player_state: 'playable', stale_evidence: false },
    };
    return route.fulfill({ json: responses[path] ?? {} });
  });
  await page.goto('http://sfn.test/');
  await page.locator('input[x-ref="fileInput"]').setInputFiles({ name: 'probe.png', mimeType: 'image/png', buffer: pixel });
  await page.getByRole('button', { name: '▶ Analyze', exact: true }).click();
  await expect(page.getByAltText('Best match', { exact: true })).toBeVisible();
  await expect(page.locator('.query-face-chip')).toBeVisible();
  await expect.poll(() => calls.includes('/api/faces/compare')).toBe(true);
  await expect(page.locator('.query-face-chip')).toHaveClass(/query-face-chip-selected/);
  await expect(page.locator('.query-face-chip')).toHaveClass(/face-cross-matched/);
  await expect(page.getByText('3 face matches represented', { exact: true })).toBeVisible();
  await expect(page.getByText('Search library version differs', { exact: true })).toBeVisible();
  await expect(page.getByText('Comparison library version differs', { exact: true })).toBeVisible();
  await expect(page.locator('[x-show="faceCompareCounts"]')).toContainText('2 review only');
  for (const [label, model] of [['Query face model provenance', 'query-test-model'],
    ['Face search model provenance', 'search-test-model'], ['Face comparison model provenance', 'compare-test-model']]) {
    await page.getByText(label, { exact: true }).click();
    await expect(page.getByText(model, { exact: true })).toBeVisible();
  }
  await page.getByTitle('Exclude from search', { exact: true }).click();
  await expect(page.locator('.query-face-chip')).not.toHaveClass(/query-face-chip-selected/);
  await page.getByRole('button', { name: 'FACE Audit', exact: true }).click();
  const audit = page.locator('.face-audit-modal-backdrop');
  await expect(audit.getByText('unknown', { exact: true })).toBeVisible();
  await expect(audit.getByText('No surviving observation provenance', { exact: true })).toBeVisible();
  await audit.getByRole('button', { name: '✕', exact: true }).click();
  await page.getByRole('button', { name: 'Play source video at this frame' }).click();
  await expect(page.locator('[x-text="videoPlaybackError"]')).toContainText('The browser could not decode this stream');
  await expect(page.getByRole('link', { name: /Download original/ }).first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('source seek moves a real chunk video and reset releases its lease', async ({ page }) => {
  const fixtureDir = await mkdtemp(resolve(tmpdir(), 'sfn-browser-chunk-'));
  let video;
  try {
    const fixture = resolve(fixtureDir, 'chunk.mp4');
    // Match production's independent faststart MP4, not a fragmented stream
    // whose loadedmetadata event only knows the first fragment's duration.
    execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error',
      '-f', 'lavfi', '-i', 'color=c=black:s=16x16:r=1', '-t', '30',
      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', fixture]);
    video = await readFile(fixture);
  } finally {
    await rm(fixtureDir, { recursive: true, force: true });
  }
  const positions = [];
  const leases = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('http://sfn.test/**', async route => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (!path.startsWith('/api/')) {
      const file = path === '/' ? 'index.html' : path.replace(/^\/static\//, '');
      const contentType = file.endsWith('.js') ? 'application/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.html') ? 'text/html' : 'image/svg+xml';
      return route.fulfill({ body: await readFile(resolve(staticRoot, file)), contentType });
    }
    if (path === '/api/test-chunk.mp4') {
      const range = /^bytes=(\d+)-(\d*)$/.exec(route.request().headers().range ?? '');
      const start = range ? Number(range[1]) : 0;
      const end = range?.[2] ? Math.min(Number(range[2]), video.length - 1) : video.length - 1;
      return route.fulfill({ status: range ? 206 : 200, body: video.subarray(start, end + 1),
        contentType: 'video/mp4', headers: { 'Accept-Ranges': 'bytes',
          ...(range ? { 'Content-Range': `bytes ${start}-${end}/${video.length}` } : {}) } });
    }
    if (path === '/api/video-chunk') {
      positions.push(Number(url.searchParams.get('t')));
      return route.fulfill({ json: { player_state: 'chunk-ready', chunk_start: 60,
        next_chunk_start: null, chunk_url: '/api/test-chunk.mp4' } });
    }
    if (path === '/api/video-lease') leases.push(url.searchParams.get('release'));
    if (/image|thumbnail|thumb/.test(path) && !path.includes('meta')) return route.fulfill({ body: pixel, contentType: 'image/png' });
    if (path === '/api/analyze') return route.fulfill({ contentType: 'text/event-stream', body: 'data: {"type":"done","session_id":"seek-case"}\n\n' });
    const responses = {
      '/api/collections': { modes: ['exact'], has_reference: false },
      '/api/faces/availability': { faces_available: false },
      '/api/query': { results: [{ file_id: 'probe', filename: 'probe.png', hits: [hit], errors: [] }] },
      '/api/video-playback-info': { source_path: hit.video_path, filename: 'source.mov',
        player_state: 'needs-transcode', duration_ms: 120000, chunk_seconds: 30, lease_seconds: 120 },
    };
    return route.fulfill({ json: responses[path] ?? {} });
  });
  await page.goto('http://sfn.test/');
  await page.locator('input[x-ref="fileInput"]').setInputFiles({ name: 'probe.png', mimeType: 'image/png', buffer: pixel });
  await page.getByRole('button', { name: '▶ Analyze', exact: true }).click();
  await page.getByRole('button', { name: 'Play source video at this frame' }).click();
  const position = page.getByRole('spinbutton', { name: 'Source position (seconds)' });
  const activeVideo = page.locator('video[x-ref="chunkA"]');
  for (const seconds of [65.5, 72]) {
    await position.fill(String(seconds));
    await page.getByRole('button', { name: 'Go to source position', exact: true }).click();
    await expect.poll(() => positions.at(-1)).toBe(seconds);
    await expect(activeVideo).toBeVisible();
    await expect.poll(() => activeVideo.evaluate(element => element.readyState)).toBeGreaterThanOrEqual(2);
    await expect.poll(() => activeVideo.evaluate(element => element.currentTime)).toBe(seconds - 60);
  }
  await page.getByRole('button', { name: '← New search', exact: true }).click();
  await expect.poll(() => leases.includes('true')).toBe(true);
  await expect(activeVideo).not.toBeVisible();
  expect(errors).toEqual([]);
});
