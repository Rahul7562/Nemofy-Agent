import assert from 'node:assert/strict';
import test from 'node:test';
import {
  hashStandaloneCredential,
  validateInitialSetup,
  validateLoginInput,
  validatePasswordChange,
} from './authValidation.ts';

test('validateLoginInput enforces non-empty username and password', () => {
  assert.equal(validateLoginInput('', 'abc').valid, false);
  assert.equal(validateLoginInput('admin', '').valid, false);
  assert.equal(validateLoginInput('admin', 'secret').valid, true);
});

test('validateInitialSetup enforces username, minimum password length, and confirmation match', () => {
  assert.equal(validateInitialSetup('', 'StrongPass1$', 'StrongPass1$').valid, false);
  assert.equal(validateInitialSetup('admin', 'short', 'short').valid, false);
  assert.equal(validateInitialSetup('admin', 'StrongPass1$', 'Different1$').valid, false);
  assert.equal(validateInitialSetup('admin', 'StrongPass1$', 'StrongPass1$').valid, true);
});

test('validatePasswordChange enforces length, difference, and match', () => {
  assert.equal(validatePasswordChange('', 'NewPass123$', 'NewPass123$').valid, false);
  assert.equal(validatePasswordChange('OldPass1$', 'short', 'short').valid, false);
  assert.equal(validatePasswordChange('SamePass1$', 'SamePass1$', 'SamePass1$').valid, false);
  assert.equal(validatePasswordChange('OldPass1$', 'NewPass123$', 'Mismatch123$').valid, false);
  assert.equal(validatePasswordChange('OldPass1$', 'NewPass123$', 'NewPass123$').valid, true);
});

test('hashStandaloneCredential produces deterministic SHA-256 hex digest and distinguishes credentials', async () => {
  const hashA = await hashStandaloneCredential('admin', 'CustomSecret99$');
  const hashA2 = await hashStandaloneCredential('admin', 'CustomSecret99$');
  const hashB = await hashStandaloneCredential('admin', 'wrongpass');
  assert.equal(hashA, hashA2);
  assert.equal(hashA.length, 64);
  assert.notEqual(hashA, hashB);
});
