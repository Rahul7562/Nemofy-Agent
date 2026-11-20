import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { AuthStore, parseCookieHeader } from './authStore';

test('AuthStore seeds admin / Rahul1234$ and verifies sessions', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-auth-'));
  const authPath = path.join(tmpDir, 'auth.json');

  const store = new AuthStore(authPath);
  assert.equal(store.getUsername(), 'admin');
  assert.equal(store.verifyCredentials('admin', 'Rahul1234$'), true);
  assert.equal(store.verifyCredentials('admin', 'wrongpass'), false);
  assert.equal(store.verifyCredentials('other', 'Rahul1234$'), false);

  const token = store.createSessionToken();
  const payload = store.verifySessionToken(token);
  assert.ok(payload);
  assert.equal(payload?.username, 'admin');
  assert.equal(store.verifySessionToken(token + 'tampered'), null);

  fs.rmSync(tmpDir, { recursive: true, force: true });
});

test('AuthStore changePassword updates hash and invalidates prior sessions', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-auth-chg-'));
  const authPath = path.join(tmpDir, 'auth.json');

  const store = new AuthStore(authPath);
  const oldToken = store.createSessionToken();

  const bad = store.changePassword('wrong', 'NewPass123!');
  assert.equal(bad.ok, false);

  const short = store.changePassword('Rahul1234$', 'short');
  assert.equal(short.ok, false);

  const ok = store.changePassword('Rahul1234$', 'NewSecure99$');
  assert.equal(ok.ok, true);
  assert.ok(ok.token);

  assert.equal(store.verifySessionToken(oldToken), null);
  assert.ok(store.verifySessionToken(ok.token));
  assert.equal(store.verifyCredentials('admin', 'Rahul1234$'), false);
  assert.equal(store.verifyCredentials('admin', 'NewSecure99$'), true);

  // Verify persistence across reload
  const reloaded = new AuthStore(authPath);
  assert.equal(reloaded.verifyCredentials('admin', 'NewSecure99$'), true);

  fs.rmSync(tmpDir, { recursive: true, force: true });
});

test('parseCookieHeader extracts cookies cleanly', () => {
  const cookies = parseCookieHeader('a=1; nemofy_session=abc.def%3D; theme=dark');
  assert.equal(cookies.a, '1');
  assert.equal(cookies.nemofy_session, 'abc.def=');
  assert.equal(cookies.theme, 'dark');
});
