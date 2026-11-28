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

test('BUG-10: readWorkspaceFile and createWorkspaceFolder return clean relative paths when workspaceDir is a symlink', () => {
  const tmpDir = createTempDir('nemofy-bug10-');
  try {
    const realBase = path.join(tmpDir, 'real-root');
    const linkBase = path.join(tmpDir, 'link-root');
    fs.mkdirSync(realBase, { recursive: true });
    fs.symlinkSync(realBase, linkBase);

    const store = new WorkspaceAndConfigStore(linkBase);
    const created = store.createWorkspaceFolder('nested/sub');
    assert.equal(created.path, 'nested/sub');
    assert.equal(created.path.includes('..'), false);

    store.writeWorkspaceFile('nested/sub/note.txt', 'hello symlink');
    const read = store.readWorkspaceFile('nested/sub/note.txt');
    assert.equal(read.path, 'nested/sub/note.txt');
    assert.equal(read.content, 'hello symlink');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('BUG-11: saveLLMConfig resolves custom_* provider base_url inside accessories and deleteSkill ignores empty sanitized names', () => {
  const tmpDir = createTempDir('nemofy-bug11-');
  try {
    const store = new WorkspaceAndConfigStore(tmpDir);
    store.upsertCustomProvider({
      id: 'custom_local_vlm',
      name: 'Local VLM',
      base_url: 'http://127.0.0.1:11434/v1',
      protocol: 'openai_compatible',
    });

    const saved = store.saveLLMConfig({
      accessories: {
        image: {
          provider: 'custom_local_vlm',
          model: 'llava:latest',
        },
      },
    });
    assert.equal(saved.accessories?.image.provider, 'custom_local_vlm');
    assert.equal(saved.accessories?.image.base_url, 'http://127.0.0.1:11434/v1');

    // Creating a decoy `.md` file in Skills/ must NOT be deleted when deleteSkill receives punctuation-only input
    const skillsDir = path.join(tmpDir, 'Skills');
    fs.mkdirSync(skillsDir, { recursive: true });
    const decoyDotMd = path.join(skillsDir, '.md');
    fs.writeFileSync(decoyDotMd, 'keep me', 'utf-8');

    store.deleteSkill('!!!___...');
    store.deleteSkill('   ');
    assert.equal(fs.existsSync(decoyDotMd), true);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('BUG-12: providerCatalog allows loopback for persisted custom_* provider while blocking cloud openai loopback', async () => {
  const http = await import('node:http');
  const { listProviderModelsDirect, probeProviderModelDirect } = await import('./providerCatalog.ts');

  const server = http.createServer((req, res) => {
    if (req.url === '/v1/models') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ data: [{ id: 'local-custom-model' }] }));
      return;
    }
    if (req.url === '/v1/chat/completions') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ choices: [{ message: { content: 'OK' } }] }));
      return;
    }
    res.writeHead(404);
    res.end();
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert(address && typeof address !== 'string');
  const localBaseUrl = `http://127.0.0.1:${address.port}/v1`;

  try {
    const customProvider = {
      id: 'custom_my_vllm',
      name: 'My vLLM',
      base_url: localBaseUrl,
      protocol: 'openai_compatible' as const,
      enabled: true,
      models: [{ id: 'local-custom-model' }],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    const modelsRes = await listProviderModelsDirect({
      provider: 'custom_my_vllm',
      base_url: localBaseUrl,
      customProvider,
    });
    assert.equal(modelsRes.ok, true);
    assert.deepEqual(
      modelsRes.models.map((m) => m.id),
      ['local-custom-model'],
    );

    const probeRes = await probeProviderModelDirect({
      role: 'primary',
      persistedRoleConfig: {
        provider: 'custom_my_vllm',
        model: 'local-custom-model',
        base_url: localBaseUrl,
        temperature: 0.2,
        max_tokens: 4096,
        reasoning_effort: 'medium',
      },
      customProvider,
    });
    assert.equal(probeRes.ok, true);

    // Built-in cloud provider targeting loopback must remain blocked
    const blockedRes = await listProviderModelsDirect({
      provider: 'openai',
      base_url: localBaseUrl,
      api_key: 'sk-test-key',
    });
    assert.equal(blockedRes.ok, false);
    assert.match(blockedRes.error || '', /Blocked/i);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test('BUG-13: normalizeAccessoriesConfig preserves empty string values during editing', async () => {
  const { normalizeAccessoriesConfig } = await import('../Nemofy Agent/src/lib/llmPresets.ts');
  const normalized = normalizeAccessoriesConfig({
    image: {
      provider: 'openrouter',
      model: '',
      base_url: '',
    },
    video: {
      provider: 'openrouter',
      model: '',
    },
  });

  assert.equal(normalized.image.model, '');
  assert.equal(normalized.image.base_url, '');
  assert.equal(normalized.video.model, '');
  // Omitted field still falls back to default
  assert.equal(normalized.video.base_url, 'https://openrouter.ai/api/v1');
});

