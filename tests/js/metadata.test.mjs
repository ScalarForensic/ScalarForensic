import assert from 'node:assert/strict';
import test from 'node:test';
import { loadComponent } from './harness.mjs';

test('incomplete or nonnumeric GPS coordinates do not break the metadata panel', () => {
  const app = loadComponent();
  for (const missing of [undefined, null, '', '12.5', NaN, Infinity]) {
    for (const coordinates of [
      { gps_lat: 52.5, gps_lon: missing },
      { gps_lat: missing, gps_lon: 13.4 },
    ]) {
      const items = app._metaItems({ ...coordinates, format: 'JPEG', software: 'Camera' });
      assert.equal(items.some(item => item.label === 'GPS'), false);
      assert.equal(items.find(item => item.label === 'Software').value, 'Camera');
    }
  }
});

test('complete GPS coordinates retain precision, matching and zero coordinates', () => {
  const app = loadComponent();
  const meta = { gps_lat: 0, gps_lon: -13.123456 };
  const gps = app._metaItems(meta, meta).find(item => item.label === 'GPS');
  assert.equal(gps.value, '0.00000, -13.12346');
  assert.equal(gps.match, true);
});
