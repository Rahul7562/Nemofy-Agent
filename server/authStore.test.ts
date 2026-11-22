import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { AuthStore, parseCookieHeader } from './authStore';

test('AuthStore requires first-run setup when NEMOFY_ADMIN_PASSWORD is unset, then verifies sessions', () => {
  const prevEnv = process.env.NEMOFY_ADMIN_PASSWORD;
  delete process.env.NEMOFY_ADMIN_PASSWORD;
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-auth-'));
  const authPath = path.join(tmpDir, 'auth.json');

  try {
    const store = new AuthStore(authPath);
    assert.equal(store.getUsername(), 'admin');
    assert.equal(store.needsSetup(), true);
    assert.equal(store.verifyCredentials('admin', 'anything'), false);

    const shortSetup = store.setupInitialPassword('short');
    assert.equal(shortSetup.ok, false);

    const setup = store.setupInitialPassword('InitialPass123$', 'admin');
    assert.equal(setup.ok, true);
    assert.ok(setup.token);
    assert.equal(store.needsSetup(), false);

    // One-time setup cannot be re-run once configured
    const secondSetup = store.setupInitialPassword('AnotherPass123$', 'admin');
    assert.equal(secondSetup.ok, false);

    assert.equal(store.verifyCredentials('admin', 'InitialPass123$'), true);
    assert.equal(store.verifyCredentials('admin', 'wrongpass'), false);
    assert.equal(store.verifyCredentials('other', 'InitialPass123$'), false);

    const payload = store.verifySessionToken(setup.token);
    assert.ok(payload);
    assert.equal(payload?.username, 'admin');
    assert.equal(store.verifySessionToken(setup.token + 'tampered'), null);
  } finally {
    if (prevEnv === undefined) delete process.env.NEMOFY_ADMIN_PASSWORD;
    else process.env.NEMOFY_ADMIN_PASSWORD = prevEnv;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('AuthStore supports NEMOFY_ADMIN_PASSWORD env override without hardcoded defaults', () => {
  const prevEnv = process.env.NEMOFY_ADMIN_PASSWORD;
  process.env.NEMOFY_ADMIN_PASSWORD = 'EnvManagedPass99$';
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-auth-env-'));
  const authPath = path.join(tmpDir, 'auth.json');

  try {
    const store = new AuthStore(authPath);
    assert.equal(store.needsSetup(), false);
    assert.equal(store.verifyCredentials('admin', 'EnvManagedPass99$'), true);
    assert.equal(store.verifyCredentials('admin', 'wrong'), false);
    const chg = store.changePassword('EnvManagedPass99$', 'AnotherPass99$');
    assert.equal(chg.ok, false);
  } finally {
    if (prevEnv === undefined) delete process.env.NEMOFY_ADMIN_PASSWORD;
    else process.env.NEMOFY_ADMIN_PASSWORD = prevEnv;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('AuthStore changePassword updates hash and invalidates prior sessions, and revokeSessionToken revokes individual tokens', () => {
  const prevEnv = process.env.NEMOFY_ADMIN_PASSWORD;
  delete process.env.NEMOFY_ADMIN_PASSWORD;
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-auth-chg-'));
  const authPath = path.join(tmpDir, 'auth.json');

  try {
    const store = new AuthStore(authPath);
    const setup = store.setupInitialPassword('StartPass123$');
    assert.equal(setup.ok, true);
    const tokA = store.createSessionToken();
    const tokB = store.createSessionToken();

    // Revoking tokA invalidates only tokA and persists across reloads
    store.revokeSessionToken(tokA);
    assert.equal(store.verifySessionToken(tokA), null);
    assert.ok(store.verifySessionToken(tokB));

    const reloadedRevoked = new AuthStore(authPath);
    assert.equal(reloadedRevoked.verifySessionToken(tokA), null);
    assert.ok(reloadedRevoked.verifySessionToken(tokB));

    const bad = store.changePassword('wrong', 'NewPass123!');
    assert.equal(bad.ok, false);

    const short = store.changePassword('StartPass123$', 'short');
    assert.equal(short.ok, false);

    const ok = store.changePassword('StartPass123$', 'NewSecure99$');
    assert.equal(ok.ok, true);
    assert.ok(ok.token);

    assert.equal(store.verifySessionToken(tokB), null);
    assert.ok(store.verifySessionToken(ok.token));
    assert.equal(store.verifyCredentials('admin', 'StartPass123$'), false);
    assert.equal(store.verifyCredentials('admin', 'NewSecure99$'), true);

    // Verify persistence across reload
    const reloaded = new AuthStore(authPath);
    assert.equal(reloaded.verifyCredentials('admin', 'NewSecure99$'), true);
  } finally {
    if (prevEnv === undefined) delete process.env.NEMOFY_ADMIN_PASSWORD;
    else process.env.NEMOFY_ADMIN_PASSWORD = prevEnv;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('parseCookieHeader extracts cookies cleanly and ignores malformed percent-encoding', () => {
  const cookies = parseCookieHeader('a=1; nemofy_session=abc.def%3D; broken=%E0%A4%A; theme=dark');
  assert.equal(cookies.a, '1');
  assert.equal(cookies.nemofy_session, 'abc.def=');
  assert.equal(cookies.broken, '%E0%A4%A');
  assert.equal(cookies.theme, 'dark');
});

test('AuthStore persists an auto-generated internal agentApiToken and honors AGENT_API_TOKEN env override', () => {
  const prevTok = process.env.AGENT_API_TOKEN;
  delete process.env.AGENT_API_TOKEN;
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-auth-tok-'));
  const authPath = path.join(tmpDir, 'auth.json');

  try {
    const storeA = new AuthStore(authPath);
    const tokenA = storeA.getAgentApiToken();
    assert.ok(tokenA.length >= 32);

    const storeB = new AuthStore(authPath);
    assert.equal(storeB.getAgentApiToken(), tokenA);

    process.env.AGENT_API_TOKEN = 'explicit-env-agent-token';
    assert.equal(storeB.getAgentApiToken(), 'explicit-env-agent-token');
  } finally {
    if (prevTok === undefined) delete process.env.AGENT_API_TOKEN;
    else process.env.AGENT_API_TOKEN = prevTok;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

