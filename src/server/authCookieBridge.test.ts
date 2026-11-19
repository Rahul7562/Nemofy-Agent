import test from 'node:test';
import assert from 'node:assert/strict';
import {
  appendAuthCookieHeader,
  AuthCookieBridge,
  extractAuthCookieUpdate,
  hasAuthCookie,
} from './authCookieBridge.ts';

test('hasAuthCookie detects non-empty n8n-auth cookie values', () => {
  assert.equal(hasAuthCookie(undefined), false);
  assert.equal(hasAuthCookie('other=1; foo=bar'), false);
  assert.equal(hasAuthCookie('n8n-auth=; other=1'), false);
  assert.equal(hasAuthCookie('other=1; n8n-auth=jwt.token.here'), true);
});

test('appendAuthCookieHeader preserves unrelated cookies and replaces empty n8n-auth', () => {
  assert.equal(
    appendAuthCookieHeader(undefined, 'tok123'),
    'n8n-auth=tok123',
  );
  assert.equal(
    appendAuthCookieHeader('theme=dark; n8n-auth=', 'tok123'),
    'theme=dark; n8n-auth=tok123',
  );
});

test('extractAuthCookieUpdate distinguishes set vs logout clear cookies', () => {
  const setUpdate = extractAuthCookieUpdate(
    ['n8n-auth=jwt.abc.def; Max-Age=3600; Path=/; HttpOnly'],
    1000,
  );
  assert.deepEqual(setUpdate, {
    kind: 'set',
    token: 'jwt.abc.def',
    maxAgeMs: 3600 * 1000,
  });

  const clearEmpty = extractAuthCookieUpdate(
    ['n8n-auth=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT'],
    1000,
  );
  assert.deepEqual(clearEmpty, { kind: 'clear' });

  const clearMaxAgeZero = extractAuthCookieUpdate(
    ['n8n-auth=old; Max-Age=0; Path=/'],
    1000,
  );
  assert.deepEqual(clearMaxAgeZero, { kind: 'clear' });
});

test('AuthCookieBridge supplies fallback token by browser-id and clears on logout', () => {
  const bridge = new AuthCookieBridge();
  const reqHeaders = {
    'browser-id': 'browser-uuid-1',
    'user-agent': 'TestBrowser/1.0',
    'x-forwarded-for': '203.0.113.10',
  };

  bridge.recordResponseCookies(
    ['n8n-auth=jwt.token.1; Max-Age=3600; Path=/; HttpOnly'],
    reqHeaders,
    undefined,
    1000,
  );

  // Missing cookie in cross-origin iframe -> fallback resolves token
  assert.equal(
    bridge.resolveFallbackTokenForHttp(reqHeaders, 2000),
    'jwt.token.1',
  );

  // Different browser-id cannot borrow the session token on HTTP
  assert.equal(
    bridge.resolveFallbackTokenForHttp(
      { ...reqHeaders, 'browser-id': 'other-browser' },
      2000,
    ),
    null,
  );

  // When native browser cookie IS present, bridge steps aside
  assert.equal(
    bridge.resolveFallbackTokenForHttp(
      { ...reqHeaders, cookie: 'n8n-auth=native.jwt' },
      2000,
    ),
    null,
  );

  // WebSocket upgrade (no custom browser-id header) matches fingerprint
  assert.equal(
    bridge.resolveFallbackTokenForWebSocket(
      {
        'user-agent': 'TestBrowser/1.0',
        'x-forwarded-for': '203.0.113.10',
      },
      undefined,
      2000,
    ),
    'jwt.token.1',
  );

  // Logout clears both maps
  bridge.recordResponseCookies(
    ['n8n-auth=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT'],
    reqHeaders,
    undefined,
    3000,
  );
  assert.equal(bridge.resolveFallbackTokenForHttp(reqHeaders, 3500), null);
  assert.equal(
    bridge.resolveFallbackTokenForWebSocket(
      {
        'user-agent': 'TestBrowser/1.0',
        'x-forwarded-for': '203.0.113.10',
      },
      undefined,
      3500,
    ),
    null,
  );
});
