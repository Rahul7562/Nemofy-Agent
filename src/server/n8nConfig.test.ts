import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildN8nChildEnv,
  isForwardedHttps,
  isHttpsProxyRequest,
  isInternalControlPath,
  normalizePublicBaseUrl,
  resolveN8nRuntimeConfig,
  rewriteSetCookieValue,
  sanitizeProxyResponseHeaders,
  shouldProxyRequestToN8n,
  stripFrameAncestorsFromCsp,
} from './n8nConfig.ts';

test('normalizePublicBaseUrl falls back to localhost proxy port for placeholders', () => {
  assert.equal(normalizePublicBaseUrl(undefined, 3000), 'http://localhost:3000/');
  assert.equal(normalizePublicBaseUrl('MY_APP_URL', 3000), 'http://localhost:3000/');
  assert.equal(
    normalizePublicBaseUrl('https://demo.run.app', 3000),
    'https://demo.run.app/',
  );
  assert.equal(
    normalizePublicBaseUrl('https://demo.run.app/', 3000),
    'https://demo.run.app/',
  );
});

test('resolveN8nRuntimeConfig resolves defaults and relative user folder', () => {
  const cfg = resolveN8nRuntimeConfig({}, '/workspace');
  assert.equal(cfg.proxyPort, 3000);
  assert.equal(cfg.n8nPort, 5678);
  assert.equal(cfg.n8nHost, '127.0.0.1');
  assert.equal(cfg.userFolder, '/workspace/.n8n-data');
  assert.equal(cfg.targetOrigin, 'http://127.0.0.1:5678');
  assert.equal(cfg.publicBaseUrl, 'http://localhost:3000/');
});

test('buildN8nChildEnv sets proxy-safe n8n environment variables', () => {
  const cfg = resolveN8nRuntimeConfig(
    {
      APP_URL: 'https://ais-dev.run.app',
      N8N_PORT: '5678',
    },
    '/workspace',
  );
  const env = buildN8nChildEnv({ PATH: '/usr/bin' }, cfg);

  assert.equal(env.N8N_HOST, '127.0.0.1');
  assert.equal(env.N8N_PORT, '5678');
  assert.equal(env.N8N_PROXY_HOPS, '1');
  assert.equal(env.N8N_SECURE_COOKIE, 'false');
  assert.equal(env.N8N_SAMESITE_COOKIE, 'none');
  assert.equal(env.N8N_EDITOR_BASE_URL, 'https://ais-dev.run.app/');
  assert.equal(env.WEBHOOK_URL, 'https://ais-dev.run.app/');
  assert.equal(env.N8N_PUSH_BACKEND, 'websocket');
});

test('rewriteSetCookieValue adapts SameSite, Secure, and Partitioned to HTTP vs HTTPS', () => {
  const raw = 'n8n-auth=abc123; Path=/; HttpOnly; SameSite=Lax; Secure';
  assert.equal(
    rewriteSetCookieValue(raw, false),
    'n8n-auth=abc123; Path=/; HttpOnly; SameSite=Lax',
  );
  assert.equal(
    rewriteSetCookieValue(raw, true),
    'n8n-auth=abc123; Path=/; HttpOnly; SameSite=None; Secure; Partitioned',
  );
});

test('stripFrameAncestorsFromCsp removes only frame-ancestors directive', () => {
  const csp = "default-src 'self'; frame-ancestors 'none'; img-src * data:";
  assert.equal(
    stripFrameAncestorsFromCsp(csp),
    "default-src 'self'; img-src * data:",
  );
});

test('sanitizeProxyResponseHeaders removes X-Frame-Options/CORP, rewrites Location and cookies', () => {
  const headers = sanitizeProxyResponseHeaders(
    {
      'x-frame-options': 'SAMEORIGIN',
      'cross-origin-resource-policy': 'same-origin',
      'content-security-policy': "default-src 'self'; frame-ancestors 'self'",
      location: 'http://127.0.0.1:5678/setup',
      'set-cookie': ['n8n-auth=tok; Path=/; HttpOnly; SameSite=Lax'],
    },
    {
      httpsRequest: true,
      targetOrigin: 'http://127.0.0.1:5678',
    },
  );

  assert.equal(headers['x-frame-options'], undefined);
  assert.equal(headers['cross-origin-resource-policy'], undefined);
  assert.equal(headers['content-security-policy'], "default-src 'self'");
  assert.equal(headers.location, '/setup');
  assert.deepEqual(headers['set-cookie'], [
    'n8n-auth=tok; Path=/; HttpOnly; SameSite=None; Secure; Partitioned',
  ]);
});

test('isForwardedHttps and isHttpsProxyRequest detect proxy HTTPS reliably', () => {
  assert.equal(isForwardedHttps({}), false);
  assert.equal(isForwardedHttps({ 'x-forwarded-proto': 'http' }), false);
  assert.equal(isForwardedHttps({ 'x-forwarded-proto': 'https' }), true);
  assert.equal(isForwardedHttps({ 'x-forwarded-proto': 'https, http' }), true);
  assert.equal(isForwardedHttps({ forwarded: 'for=1.2.3.4;proto=https' }), true);

  assert.equal(
    isHttpsProxyRequest({ host: 'localhost:3000' }, 'https://ais-dev.run.app/'),
    false,
  );
  assert.equal(
    isHttpsProxyRequest({ host: 'ais-dev.run.app' }, 'https://ais-dev.run.app/'),
    true,
  );
  assert.equal(
    isHttpsProxyRequest({ origin: 'https://ais-dev.run.app' }, 'http://localhost:3000/'),
    true,
  );
});

test('shouldProxyRequestToN8n isolates control routes and waits for ready phase', () => {
  assert.equal(isInternalControlPath('/_n8n_proxy/status'), true);
  assert.equal(isInternalControlPath('/_n8n_proxy/boot?preview=1'), true);
  assert.equal(isInternalControlPath('/_n8n_proxy/boot/src/main.tsx'), true);
  assert.equal(isInternalControlPath('/_n8n_proxy/boot/@vite/client'), true);
  assert.equal(isInternalControlPath('/rest/settings'), false);
  assert.equal(isInternalControlPath('/assets/index.js'), false);

  assert.equal(shouldProxyRequestToN8n('/', 'starting'), false);
  assert.equal(shouldProxyRequestToN8n('/', 'error'), false);
  assert.equal(shouldProxyRequestToN8n('/', 'ready'), true);
  assert.equal(shouldProxyRequestToN8n('/assets/index.js', 'ready'), true);
  assert.equal(shouldProxyRequestToN8n('/_n8n_proxy/status', 'ready'), false);
  assert.equal(shouldProxyRequestToN8n('/_n8n_proxy/boot', 'ready'), false);
  assert.equal(
    shouldProxyRequestToN8n('/_n8n_proxy/boot/@vite/client', 'ready'),
    false,
  );
});
