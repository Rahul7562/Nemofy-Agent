import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEFAULT_STANDALONE_CREDENTIAL_DIGEST,
  hashStandaloneCredential,
  validateLoginInput,
  validatePasswordChange,
} from './authValidation.ts';

test('validateLoginInput enforces non-empty username and password', () => {
  assert.equal(validateLoginInput('', 'abc').valid, false);
  assert.equal(validateLoginInput('admin', '').valid, false);
  assert.equal(validateLoginInput('admin', 'secret').valid, true);
});

test('validatePasswordChange enforces length, difference, and match', () => {
  assert.equal(validatePasswordChange('', 'NewPass123$', 'NewPass123$').valid, false);
  assert.equal(validatePasswordChange('OldPass1$', 'short', 'short').valid, false);
  assert.equal(validatePasswordChange('SamePass1$', 'SamePass1$', 'SamePass1$').valid, false);
  assert.equal(validatePasswordChange('OldPass1$', 'NewPass123$', 'Mismatch123$').valid, false);
  assert.equal(validatePasswordChange('OldPass1$', 'NewPass123$', 'NewPass123$').valid, true);
});

test('hashStandaloneCredential matches default seeded digest and rejects wrong inputs', async () => {
  const validHash = await hashStandaloneCredential('admin', 'Rahul1234$');
  assert.equal(validHash, DEFAULT_STANDALONE_CREDENTIAL_DIGEST);

  const invalidHash = await hashStandaloneCredential('admin', 'wrongpass');
  assert.notEqual(invalidHash, DEFAULT_STANDALONE_CREDENTIAL_DIGEST);
});
