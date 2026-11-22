import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ProfileSecretVaultStore } from './profileVault.ts';

function createTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-vault-test-'));
}

test('ProfileSecretVaultStore encrypts API keys at rest with AES-256-GCM and never writes plaintext to disk', () => {
  const dir = createTempDir();
  const masterKey = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
  const vault = new ProfileSecretVaultStore(dir, () => masterKey);

  const saved = vault.setSecrets('admin', {
    NVIDIA_API_KEY: 'nvapi-super-secret-value-998877',
    RIVER_AI_API: 'rv_u_secret_river_key_1234',
  });
  assert.deepEqual(saved.sort(), ['NVIDIA_API_KEY', 'RIVER_AI_API']);

  const diskPath = path.join(dir, 'admin.vault.json');
  const rawDisk = fs.readFileSync(diskPath, 'utf-8');
  assert.equal(rawDisk.includes('nvapi-super-secret-value-998877'), false);
  assert.equal(rawDisk.includes('rv_u_secret_river_key_1234'), false);
  assert.equal(rawDisk.includes('aes-256-gcm'), true);

  const decrypted = vault.getDecryptedSecrets('admin');
  assert.equal(decrypted.NVIDIA_API_KEY, 'nvapi-super-secret-value-998877');
  assert.equal(decrypted.RIVER_AI_API, 'rv_u_secret_river_key_1234');

  const meta = vault.listSecretMetadata('admin');
  assert.equal(meta.length, 2);
  assert.equal(meta.some((m) => m.env_var === 'NVIDIA_API_KEY' && m.masked_key === 'nvap...8877'), true);
});

test('ProfileSecretVaultStore isolates user profiles, blocks traversal usernames, and detects tampering', () => {
  const dir = createTempDir();
  const masterKey = 'abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789';
  const vault = new ProfileSecretVaultStore(dir, () => masterKey);

  vault.setSecrets('admin', { OPENAI_API_KEY: 'sk-admin-key-9999' });
  assert.deepEqual(vault.getDecryptedSecrets('otheruser'), {});

  assert.throws(() => vault.getDecryptedSecrets('../etc/passwd'), /Invalid profile username/);
  assert.throws(() => vault.setSecrets('admin', { PATH: '/tmp' }), /Unsupported API key variable/);

  // Tamper with ciphertext on disk -> AES-GCM auth tag verification must reject
  const diskPath = path.join(dir, 'admin.vault.json');
  const parsed = JSON.parse(fs.readFileSync(diskPath, 'utf-8')) as Record<string, string>;
  const buf = Buffer.from(parsed.ciphertext, 'base64');
  buf[0] ^= 0xff;
  parsed.ciphertext = buf.toString('base64');
  fs.writeFileSync(diskPath, JSON.stringify(parsed));

  assert.throws(() => vault.getDecryptedSecrets('admin'));
});

test('ProfileSecretVaultStore deletes secrets cleanly and migrates + scrubs legacy .env provider keys', () => {
  const dir = createTempDir();
  const envFile = path.join(dir, '.env');
  fs.writeFileSync(
    envFile,
    [
      '# Runtime settings',
      'AGENT_API_TOKEN=keep-this-token-123',
      'RIVER_AI_API=rv_legacy_secret_5678',
      'NVIDIA_API_KEY=nvapi_legacy_secret_4321',
    ].join('\n') + '\n',
  );

  const vault = new ProfileSecretVaultStore(path.join(dir, 'vaults'), () => 'master-test-key-material-xyz');
  const migrated = vault.migrateLegacyEnvSecrets('admin', envFile, {});
  assert.deepEqual(migrated.sort(), ['NVIDIA_API_KEY', 'RIVER_AI_API']);

  // Keys are now securely in admin's vault
  const loaded = vault.getDecryptedSecrets('admin');
  assert.equal(loaded.RIVER_AI_API, 'rv_legacy_secret_5678');
  assert.equal(loaded.NVIDIA_API_KEY, 'nvapi_legacy_secret_4321');

  // Plaintext provider keys were scrubbed from .env, while AGENT_API_TOKEN stayed untouched
  const cleanedEnv = fs.readFileSync(envFile, 'utf-8');
  assert.equal(cleanedEnv.includes('AGENT_API_TOKEN=keep-this-token-123'), true);
  assert.equal(cleanedEnv.includes('rv_legacy_secret_5678'), false);
  assert.equal(cleanedEnv.includes('nvapi_legacy_secret_4321'), false);

  // Deleting a single secret keeps other keys intact
  assert.equal(vault.deleteSecret('admin', 'NVIDIA_API_KEY'), true);
  assert.equal(vault.deleteSecret('admin', 'NVIDIA_API_KEY'), false);
  assert.deepEqual(Object.keys(vault.getDecryptedSecrets('admin')), ['RIVER_AI_API']);
});

test('ProfileSecretVaultStore recovers cleanly on write when resetIfUnreadable is enabled after master key rotation', () => {
  const dir = createTempDir();
  const oldVault = new ProfileSecretVaultStore(dir, () => 'old-master-key-material-0000000000');
  oldVault.setSecrets('admin', { OPENROUTER_API_KEY: 'sk-or-old-key-1111' });

  const newVault = new ProfileSecretVaultStore(dir, () => 'new-master-key-material-9999999999');
  // Strict reads and strict writes still fail on key mismatch
  assert.throws(() => newVault.getDecryptedSecrets('admin'));
  assert.throws(() => newVault.setSecrets('admin', { OPENROUTER_API_KEY: 'sk-or-new-key-2222' }));

  // Explicit recovery re-initializes the encrypted vault cleanly under the new master key
  const saved = newVault.setSecrets(
    'admin',
    { OPENROUTER_API_KEY: 'sk-or-new-key-2222' },
    { resetIfUnreadable: true },
  );
  assert.deepEqual(saved, ['OPENROUTER_API_KEY']);
  assert.equal(newVault.getDecryptedSecrets('admin').OPENROUTER_API_KEY, 'sk-or-new-key-2222');
});

