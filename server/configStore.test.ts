import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  WorkspaceAndConfigStore,
  getDefaultAccessoriesConfig,
  maskEnvSecret,
  upsertEnvContent,
} from './configStore.ts';

test('upsertEnvContent preserves comments, updates existing keys, and appends new keys', () => {
  const initial = [
    '# Nemofy Agent env',
    'AGENT_API_TOKEN=secret-tok-123',
    'RIVER_AI_API=old-river-key',
    '',
  ].join('\n');

  const updated = upsertEnvContent(initial, {
    RIVER_AI_API: 'new-river-key-999',
    OPENAI_API_KEY: 'sk-proj-test-1234',
  });

  assert.match(updated, /# Nemofy Agent env/);
  assert.match(updated, /AGENT_API_TOKEN=secret-tok-123/);
  assert.match(updated, /RIVER_AI_API=new-river-key-999/);
  assert.match(updated, /OPENAI_API_KEY=sk-proj-test-1234/);
  assert.equal(updated.includes('old-river-key'), false);
});

test('maskEnvSecret redacts secrets safely', () => {
  assert.equal(maskEnvSecret(''), null);
  assert.equal(maskEnvSecret('short'), '****');
  assert.equal(maskEnvSecret('sk-proj-1234567890abcd'), 'sk-p...abcd');
});

test('WorkspaceAndConfigStore persists accessories config with Nemotron Omni defaults', () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-cfg-test-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);
    const initial = store.getLLMConfig();
    assert.ok(initial.accessories);
    assert.equal(initial.accessories.video.provider, 'openrouter');
    assert.equal(
      initial.accessories.video.model,
      'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    );
    assert.equal(initial.accessories.audio.enabled, true);

    const updated = store.saveLLMConfig({
      ...initial,
      accessories: {
        ...getDefaultAccessoriesConfig(),
        audio: {
          ...getDefaultAccessoriesConfig().audio,
          provider: 'gemini',
          model: 'gemini-2.5-flash',
          base_url: 'https://generativelanguage.googleapis.com/v1beta/openai',
        },
      },
    });
    assert.equal(updated.accessories?.audio.provider, 'gemini');
    assert.equal(updated.accessories?.audio.model, 'gemini-2.5-flash');

    // Re-read from disk to confirm persistence
    const reloaded = new WorkspaceAndConfigStore(tmpRoot).getLLMConfig();
    assert.equal(reloaded.accessories?.audio.provider, 'gemini');
    assert.equal(reloaded.accessories?.audio.model, 'gemini-2.5-flash');
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});

test('saveUploadedMedia enforces MIME allowlist, rejects SVG, and confines paths against traversal', () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-media-test-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);

    // Valid PNG upload
    const pngAtt = store.saveUploadedMedia({
      threadId: 'thread-abc123',
      filename: 'chart.png',
      mimeType: 'image/png',
      buffer: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    });
    assert.equal(pngAtt.kind, 'image');
    assert.equal(pngAtt.mimeType, 'image/png');
    assert.ok(pngAtt.url.startsWith('/api/media/thread-abc123/'));
    const storedName = pngAtt.storageKey.split('/')[1] || '';
    const resolved = store.resolveUploadedMediaPath('thread-abc123', storedName);
    assert.ok(resolved);
    assert.ok(fs.existsSync(resolved.fullPath));
    assert.equal(store.resolveAttachmentLocalPath(pngAtt.storageKey), resolved.fullPath);

    // Valid WAV upload
    const wavAtt = store.saveUploadedMedia({
      threadId: 'thread-abc123',
      filename: 'memo.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.from('RIFF....WAVE'),
    });
    assert.equal(wavAtt.kind, 'audio');

    // Reject SVG to prevent stored XSS
    assert.throws(
      () =>
        store.saveUploadedMedia({
          threadId: 'thread-abc123',
          filename: 'evil.svg',
          mimeType: 'image/svg+xml',
          buffer: Buffer.from('<svg onload="alert(1)"/>'),
        }),
      /Unsupported media type/i,
    );

    // Reject path traversal on media resolution
    assert.equal(store.resolveUploadedMediaPath('..', 'llm_config.json'), null);
    assert.equal(store.resolveUploadedMediaPath('thread-abc123', '../secret.txt'), null);
    assert.equal(store.resolveAttachmentLocalPath('../llm_config.json'), null);
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});

test('listWorkspaceTree skips .cache/node_modules, supports lazy listWorkspaceDirectory, and readWorkspaceFile guards binary/oversized files', () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-fs-test-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);
    const ws = store.workspaceDir;

    // Create noisy cache/venv directories that should be excluded from workspace explorer
    fs.mkdirSync(path.join(ws, '.cache', 'pip', 'wheels'), { recursive: true });
    fs.writeFileSync(path.join(ws, '.cache', 'pip', 'wheels', 'huge.whl'), 'cached', 'utf-8');
    fs.mkdirSync(path.join(ws, 'node_modules', 'pkg'), { recursive: true });
    fs.writeFileSync(path.join(ws, 'node_modules', 'pkg', 'index.js'), 'module.exports = 1;', 'utf-8');

    // Create normal workspace content including nested folder and binary artifact
    fs.writeFileSync(path.join(ws, 'script.py'), 'print("hello")\n', 'utf-8');
    fs.mkdirSync(path.join(ws, 'project', 'deep', 'nested'), { recursive: true });
    fs.writeFileSync(path.join(ws, 'project', 'deep', 'nested', 'leaf.txt'), 'leaf', 'utf-8');
    fs.writeFileSync(path.join(ws, 'binary.o'), Buffer.from([0x7f, 0x45, 0x4c, 0x46, 0x00, 0x01]));

    const tree = store.listWorkspaceTree();
    const rootNames = tree.map((n) => n.name);
    assert.equal(rootNames.includes('.cache'), false);
    assert.equal(rootNames.includes('node_modules'), false);
    assert.equal(rootNames.includes('script.py'), true);
    assert.equal(rootNames.includes('project'), true);

    // Lazy directory listing works for nested paths
    const nestedListing = store.listWorkspaceDirectory('project/deep');
    assert.equal(nestedListing.entries.length, 1);
    assert.equal(nestedListing.entries[0].name, 'nested');
    assert.equal(nestedListing.entries[0].hasChildren, true);

    // Normal text file reads cleanly
    const pyFile = store.readWorkspaceFile('script.py');
    assert.equal(pyFile.content, 'print("hello")\n');

    // Binary file throws a friendly error instead of returning garbage
    assert.throws(() => store.readWorkspaceFile('binary.o'), /Binary file/i);
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});

test('WorkspaceAndConfigStore preserves blank primary.model and explicit context_window without snapping to a fallback', () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-primary-blank-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);
    const initial = store.getLLMConfig();

    const cleared = store.saveLLMConfig({
      ...initial,
      primary: {
        ...initial.primary,
        provider: 'openrouter',
        model: '',
        base_url: 'https://openrouter.ai/api/v1',
        context_window: null,
      },
    });
    assert.equal(cleared.primary.model, '');

    const reloaded = new WorkspaceAndConfigStore(tmpRoot).getLLMConfig();
    assert.equal(reloaded.primary.model, '');

    const withWindow = store.saveLLMConfig({
      ...reloaded,
      primary: {
        ...reloaded.primary,
        model: 'anthropic/claude-opus-4.6',
        context_window: 200_000,
      },
    });
    assert.equal(withWindow.primary.model, 'anthropic/claude-opus-4.6');
    assert.equal(withWindow.primary.context_window, 200_000);
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});

test('resolveWorkspacePath and workspace file ops reject symlinks escaping workspace and enforce createOnly (BUG-4, BUG-18)', () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-symlink-test-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);
    const outsideSecret = path.join(tmpRoot, 'outside-secret.txt');
    fs.writeFileSync(outsideSecret, 'TOP_SECRET', 'utf-8');

    const outsideDir = path.join(tmpRoot, 'outside-dir');
    fs.mkdirSync(outsideDir, { recursive: true });
    fs.writeFileSync(path.join(outsideDir, 'leak.txt'), 'LEAK', 'utf-8');

    fs.symlinkSync(outsideSecret, path.join(store.workspaceDir, 'link.txt'));
    fs.symlinkSync(outsideDir, path.join(store.workspaceDir, 'linked-dir'));

    assert.throws(() => store.readWorkspaceFile('link.txt'), /symlink|escapes/i);
    assert.throws(() => store.writeWorkspaceFile('link.txt', 'overwrite'), /symlink|escapes/i);
    assert.throws(() => store.writeWorkspaceFile('linked-dir/pwned.txt', 'x'), /symlink|escapes/i);
    assert.throws(() => store.listWorkspaceDirectory('linked-dir'), /symlink|escapes/i);
    assert.equal(fs.readFileSync(outsideSecret, 'utf-8'), 'TOP_SECRET');

    const treeNames = store.listWorkspaceTree().map((n) => n.name);
    assert.equal(treeNames.includes('link.txt'), false);
    assert.equal(treeNames.includes('linked-dir'), false);

    // createOnly prevents silent truncation of existing workspace files
    store.writeWorkspaceFile('notes.md', 'important content');
    assert.throws(
      () => store.writeWorkspaceFile('notes.md', '', { createOnly: true }),
      /already exists/i,
    );
    assert.equal(store.readWorkspaceFile('notes.md').content, 'important content');
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});

test('updateThreads preserves concurrent thread writes and saveMCPServers blocks internal SSRF URLs (BUG-5, BUG-16)', () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-threads-mcp-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);
    const now = new Date().toISOString();

    store.updateThreads((threads) => {
      threads.push({
        id: 'thread-1',
        title: 'First',
        createdAt: now,
        updatedAt: now,
        messages: [],
      });
    });
    store.updateThreads((threads) => {
      threads.push({
        id: 'thread-2',
        title: 'Second',
        createdAt: now,
        updatedAt: now,
        messages: [],
      });
    });

    const ids = store.listThreads().map((t) => t.id).sort();
    assert.deepEqual(ids, ['thread-1', 'thread-2']);

    for (const badUrl of [
      'http://127.0.0.1:8080/mcp',
      'http://169.254.169.254/latest/meta-data',
      'http://localhost:3000/api',
      'file:///etc/passwd',
    ]) {
      assert.throws(
        () =>
          store.saveMCPServers([
            {
              name: 'bad',
              url: badUrl,
              transport: 'streamable_http',
              enabled: true,
              timeout_seconds: 30,
            },
          ]),
        /URL|loopback|private|internal|http/i,
      );
    }

    const saved = store.saveMCPServers([
      {
        name: 'valid',
        url: 'https://mcp.example.com/sse',
        transport: 'sse',
        enabled: true,
        timeout_seconds: 30,
      },
    ]);
    assert.equal(saved.length, 1);
    assert.equal(saved[0].url, 'https://mcp.example.com/sse');
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});

