import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: '.',
  testMatch: '*.spec.mjs',
  workers: 1,
  use: { headless: true, viewport: { width: 1600, height: 1100 } },
});
