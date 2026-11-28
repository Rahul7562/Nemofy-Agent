import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import test from 'node:test';
import {
  buildFirecrawlChildEnv,
  FirecrawlLocalProcessManager,
  isLoopbackOrSelfUrl,
} from './firecrawlProcessManager.ts';

test('isLoopbackOrSelfUrl identifies loopback hosts, IPv6, wildcard, and matching host:port', () => {
  assert.equal(isLoopbackOrSelfUrl('http://127.0.0.1:3002'), true);
  assert.equal(isLoopbackOrSelfUrl('http://localhost:3002'), true);
  assert.equal(isLoopbackOrSelfUrl('http://[::1]:3002'), true);
  assert.equal(isLoopbackOrSelfUrl('http://0.0.0.0:3002'), true);
  assert.equal(isLoopbackOrSelfUrl('http://sub.localhost:3002'), true);
  assert.equal(isLoopbackOrSelfUrl('http://10.0.0.5:3002', '10.0.0.5', 3002), true);
  assert.equal(isLoopbackOrSelfUrl('http://10.0.0.5:9999', '10.0.0.5', 3002), false);
  assert.equal(isLoopbackOrSelfUrl('https://api.firecrawl.dev'), false);
  assert.equal(isLoopbackOrSelfUrl('https://api.firecrawl.dev/v2'), false);
  assert.equal(isLoopbackOrSelfUrl(''), false);
  assert.equal(isLoopbackOrSelfUrl('not-a-url'), false);
});

test('buildFirecrawlChildEnv strips loopback FIRECRAWL_API_URL to prevent self-forwarding recursion', () => {
  const childEnv = buildFirecrawlChildEnv(
    {
      PATH: '/usr/bin',
      FIRECRAWL_API_URL: 'http://127.0.0.1:3002',
      FIRECRAWL_CLOUD: 'auto',
    },
    {
      FIRECRAWL_API_URL: 'http://localhost:3002',
    },
    {
      host: '127.0.0.1',
      port: 3002,
      dataHome: '/tmp/fc-data',
      chromePath: '/usr/bin/chromium',
    },
  );

  assert.equal('FIRECRAWL_API_URL' in childEnv, false);
  assert.equal(childEnv.FIRECRAWL_LOCAL_HOME, '/tmp/fc-data');
  assert.equal(childEnv.FIRECRAWL_CLOUD, 'auto');
  assert.equal(childEnv.FIRECRAWL_CHROME_PATH, '/usr/bin/chromium');
});

test('buildFirecrawlChildEnv preserves non-loopback cloud URL and honors FIRECRAWL_CLOUD_API_URL', () => {
  const withCloudBase = buildFirecrawlChildEnv(
    { FIRECRAWL_API_URL: 'https://api.firecrawl.dev' },
    {},
    { host: '127.0.0.1', port: 3002, dataHome: '/tmp/fc-data' },
  );
  assert.equal(withCloudBase.FIRECRAWL_API_URL, 'https://api.firecrawl.dev');

  const withExplicitCloudOverride = buildFirecrawlChildEnv(
    {
      FIRECRAWL_API_URL: 'http://127.0.0.1:3002',
      FIRECRAWL_CLOUD_API_URL: 'https://api.firecrawl.dev',
    },
    {},
    { host: '127.0.0.1', port: 3002, dataHome: '/tmp/fc-data' },
  );
  assert.equal(withExplicitCloudOverride.FIRECRAWL_API_URL, 'https://api.firecrawl.dev');

  // Loopback FIRECRAWL_CLOUD_API_URL must NOT be mapped back to FIRECRAWL_API_URL
  const withLoopbackCloudOverride = buildFirecrawlChildEnv(
    {
      FIRECRAWL_API_URL: 'http://127.0.0.1:3002',
      FIRECRAWL_CLOUD_API_URL: 'http://127.0.0.1:3002',
    },
    {},
    { host: '127.0.0.1', port: 3002, dataHome: '/tmp/fc-data' },
  );
  assert.equal('FIRECRAWL_API_URL' in withLoopbackCloudOverride, false);
});

test('FirecrawlLocalProcessManager.isHealthy sends Authorization header from extraEnv FIRECRAWL_LOCAL_TOKEN', async () => {
  let receivedAuth: string | undefined;
  const server = http.createServer((req, res) => {
    receivedAuth = req.headers.authorization;
    if (req.url === '/health' && receivedAuth === 'Bearer secret-local-token') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok' }));
      return;
    }
    res.writeHead(401);
    res.end();
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert(address && typeof address !== 'string');

  try {
    const manager = new FirecrawlLocalProcessManager(process.cwd(), address.port, () => ({
      FIRECRAWL_LOCAL_TOKEN: 'secret-local-token',
    }));
    const healthy = await manager.isHealthy();
    assert.equal(healthy, true);
    assert.equal(receivedAuth, 'Bearer secret-local-token');
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test('scripts/run-firecrawl-local.sh strips loopback FIRECRAWL_API_URL before launching firecrawl-local', () => {
  const scriptPath = path.resolve(process.cwd(), 'scripts', 'run-firecrawl-local.sh');
  const script = fs.readFileSync(scriptPath, 'utf-8');
  assert.match(script, /unset FIRECRAWL_API_URL/);
  assert.match(script, /FIRECRAWL_CLOUD_API_URL/);
});
