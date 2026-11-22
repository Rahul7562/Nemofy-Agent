import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { AuthStore } from './authStore.ts';
import { WorkspaceAndConfigStore } from './configStore.ts';
import { ChatJobOrchestrator } from './jobOrchestrator.ts';
import { ProfileSecretVaultStore } from './profileVault.ts';

function createTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

test('BUG-1: ProfileSecretVaultStore and AuthStore consistently support dotted usernames and reject traversal', () => {
  const tmpDir = createTempDir('nemofy-bug1-');
  try {
    const authStore = new AuthStore(path.join(tmpDir, 'auth.json'));
    const badDots = authStore.setupInitialPassword('strongpass123', '..');
    assert.equal(badDots.ok, false);

    const badLeading = authStore.setupInitialPassword('strongpass123', '.admin');
    assert.equal(badLeading.ok, false);

    const okDotted = authStore.setupInitialPassword('strongpass123', 'john.doe');
    assert.equal(okDotted.ok, true);
    assert.equal(okDotted.username, 'john.doe');

    const masterKey = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
    const vault = new ProfileSecretVaultStore(path.join(tmpDir, 'vaults'), () => masterKey);
    vault.setSecrets('john.doe', { OPENAI_API_KEY: 'sk-test-1234567890' });
    const loaded = vault.getDecryptedSecrets('john.doe');
    assert.equal(loaded.OPENAI_API_KEY, 'sk-test-1234567890');

    // Path traversal usernames must still be rejected
    assert.throws(() => vault.setSecrets('..', { OPENAI_API_KEY: 'sk-bad' }), /Invalid profile username/);
    assert.throws(() => vault.setSecrets('john..doe', { OPENAI_API_KEY: 'sk-bad' }), /Invalid profile username/);
    assert.throws(() => vault.setSecrets('.hidden', { OPENAI_API_KEY: 'sk-bad' }), /Invalid profile username/);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('BUG-2 & BUG-3: ChatJobOrchestrator launch reservation blocks races and delete tracking prevents resurrection', () => {
  const tmpDir = createTempDir('nemofy-bug23-');
  try {
    const store = new WorkspaceAndConfigStore(tmpDir);

    const now = new Date().toISOString();
    store.saveThreads([
      {
        id: 'thread-race-1',
        title: 'Race test',
        createdAt: now,
        updatedAt: now,
        messages: [],
        pendingApproval: null,
      },
    ]);

    const orchestrator = new ChatJobOrchestrator({
      store,
      callAgentCore: async () => ({ ok: true, status: 200, data: { status: 'running' } }),
    });

    // First synchronous acquire succeeds; second concurrent acquire for same thread fails
    assert.equal(orchestrator.tryAcquireThreadLaunch('thread-race-1'), true);
    assert.equal(orchestrator.tryAcquireThreadLaunch('thread-race-1'), false);
    assert.equal(orchestrator.isThreadLaunching('thread-race-1'), true);

    // Different thread can acquire simultaneously
    assert.equal(orchestrator.tryAcquireThreadLaunch('thread-race-2'), true);

    // Mark deleted mid-flight
    orchestrator.markThreadDeleted('thread-race-1');
    assert.equal(orchestrator.isThreadDeleted('thread-race-1'), true);

    orchestrator.releaseThreadLaunch('thread-race-1');
    orchestrator.releaseThreadLaunch('thread-race-2');
    assert.equal(orchestrator.isThreadLaunching('thread-race-1'), false);

    // Delete thread from store and verify finalizeJob never resurrects it
    store.saveThreads([]);
    const res = orchestrator.finalizeJob('job-deleted-1', {
      thread_id: 'thread-race-1',
      status: 'completed',
      result: { final_response: 'Should not resurrect' },
    });
    assert.equal(res.thread, null);
    assert.equal(store.listThreads().length, 0);

    orchestrator.shutdown();
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('BUG-4: saveLLMConfig writes llm_config.json atomically without leftover temp files', () => {
  const tmpDir = createTempDir('nemofy-bug4-');
  try {
    const store = new WorkspaceAndConfigStore(tmpDir);

    const saved = store.saveLLMConfig({
      max_turns: 120,
      access_mode: 'full_access',
    });
    assert.equal(saved.max_turns, 120);
    assert.equal(saved.access_mode, 'full_access');

    const dataDir = path.join(tmpDir, 'data');
    const files = fs.readdirSync(dataDir);
    assert.equal(files.some((f) => f.endsWith('.tmp')), false);
    const parsed = JSON.parse(fs.readFileSync(path.join(dataDir, 'llm_config.json'), 'utf-8'));
    assert.equal(parsed.max_turns, 120);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
