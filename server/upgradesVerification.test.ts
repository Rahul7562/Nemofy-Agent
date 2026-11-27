import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  WorkspaceAndConfigStore,
  deriveCustomProviderSecretKey,
  getDefaultAllowUnsafeLocalBash,
  isCustomProviderId,
} from './configStore.ts';
import { ProfileSecretVaultStore, isAllowedProfileSecretKey } from './profileVault.ts';
import {
  clearLiveCatalogCache,
  clearReferenceModelCatalogCache,
  resolveModelMetadataFromCatalog,
  resolveProviderApiKey,
} from './providerCatalog.ts';
import {
  computeInputBudgetMetrics,
  computeModelContextUsage,
  deriveDetectionMode,
  resolveModelContextWindow,
} from '../Nemofy Agent/src/lib/contextUtils.ts';
import type { ChatThread, PersistedLLMConfig } from '../Nemofy Agent/src/types/agent.ts';

function makeTempStore() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-upgrades-test-'));
  const store = new WorkspaceAndConfigStore(root);
  return {
    root,
    store,
    cleanup() {
      fs.rmSync(root, { recursive: true, force: true });
    },
  };
}

// =============================================================================
// 1. ALLOW_UNSAFE_LOCAL_BASH persistence and env-var consistency
// =============================================================================

test('ALLOW_UNSAFE_LOCAL_BASH defaults to false, respects env var, and persists across restarts', () => {
  assert.equal(getDefaultAllowUnsafeLocalBash({}), false);
  assert.equal(getDefaultAllowUnsafeLocalBash({ ALLOW_UNSAFE_LOCAL_BASH: 'false' }), false);
  assert.equal(getDefaultAllowUnsafeLocalBash({ ALLOW_UNSAFE_LOCAL_BASH: '0' }), false);
  assert.equal(getDefaultAllowUnsafeLocalBash({ ALLOW_UNSAFE_LOCAL_BASH: 'true' }), true);
  assert.equal(getDefaultAllowUnsafeLocalBash({ ALLOW_UNSAFE_LOCAL_BASH: '1' }), true);

  const { root, store, cleanup } = makeTempStore();
  const prevEnv = process.env.ALLOW_UNSAFE_LOCAL_BASH;
  try {
    delete process.env.ALLOW_UNSAFE_LOCAL_BASH;
    assert.equal(store.getLLMConfig().allow_unsafe_local_bash, false);

    // Enable via UI/API persistence
    const savedOn = store.saveLLMConfig({ allow_unsafe_local_bash: true });
    assert.equal(savedOn.allow_unsafe_local_bash, true);

    // Simulate process restart with fresh store instance pointing at same disk root
    const restartedStore = new WorkspaceAndConfigStore(root);
    assert.equal(restartedStore.getLLMConfig().allow_unsafe_local_bash, true);

    // Disable again and verify persistence across second restart
    restartedStore.saveLLMConfig({ allow_unsafe_local_bash: false });
    const restartedStore2 = new WorkspaceAndConfigStore(root);
    assert.equal(restartedStore2.getLLMConfig().allow_unsafe_local_bash, false);
  } finally {
    if (prevEnv === undefined) {
      delete process.env.ALLOW_UNSAFE_LOCAL_BASH;
    } else {
      process.env.ALLOW_UNSAFE_LOCAL_BASH = prevEnv;
    }
    cleanup();
  }
});

// =============================================================================
// 2. Multiple Independent Custom Providers + Vault Encryption + Legacy Migration
// =============================================================================

test('supports multiple independent custom providers with encrypted per-provider keys and legacy migration', () => {
  const { root, store, cleanup } = makeTempStore();
  try {
    const vault = new ProfileSecretVaultStore(root, () => 'test-master-key-1234567890');

    // Create two completely independent custom providers (OpenAI-compatible & Anthropic-compatible)
    const alpha = store.upsertCustomProvider({
      name: 'Alpha Gateway',
      base_url: 'https://alpha.example.com/v1',
      protocol: 'openai_compatible',
      enabled: true,
      timeout_seconds: 120,
      models: [
        { id: 'alpha-pro', name: 'Alpha Pro', context_window: 131072, max_output_tokens: 16384 },
        { id: 'alpha-mini', name: 'Alpha Mini', context_window: 65536, max_output_tokens: 8192 },
      ],
    });
    const beta = store.upsertCustomProvider({
      name: 'Beta Claude Proxy',
      base_url: 'https://beta.example.com/anthropic',
      protocol: 'anthropic_compatible',
      enabled: true,
      timeout_seconds: 240,
      models: [
        { id: 'claude-custom-4', name: 'Claude Custom 4', context_window: 200000, max_output_tokens: 32000 },
      ],
    });

    assert.equal(isCustomProviderId(alpha.id), true);
    assert.equal(isCustomProviderId(beta.id), true);
    assert.notEqual(alpha.id, beta.id);
    assert.equal(alpha.protocol, 'openai_compatible');
    assert.equal(beta.protocol, 'anthropic_compatible');

    // Store independent secrets in encrypted vault
    const alphaSecretKey = deriveCustomProviderSecretKey(alpha.id);
    const betaSecretKey = deriveCustomProviderSecretKey(beta.id);
    assert.equal(isAllowedProfileSecretKey(alphaSecretKey), true);
    assert.equal(isAllowedProfileSecretKey(betaSecretKey), true);
    assert.equal(isAllowedProfileSecretKey('FIRECRAWL_API_KEY'), true);

    vault.setSecrets('admin', {
      [alphaSecretKey]: 'sk-alpha-super-secret-value',
      [betaSecretKey]: 'sk-beta-anthropic-secret-value',
    });

    const envOverlay = vault.getDecryptedSecrets('admin');
    assert.equal(resolveProviderApiKey(alpha.id, 'primary', envOverlay), 'sk-alpha-super-secret-value');
    assert.equal(resolveProviderApiKey(beta.id, 'primary', envOverlay), 'sk-beta-anthropic-secret-value');

    // Provider availability must include both custom providers WITHOUT exposing plaintext keys
    const availability = store.getProviderAvailability(envOverlay);
    const alphaAvail = availability[alpha.id];
    const betaAvail = availability[beta.id];
    assert.ok(alphaAvail, 'Alpha custom provider must appear in provider availability');
    assert.ok(betaAvail, 'Beta custom provider must appear in provider availability');
    assert.equal(alphaAvail.is_custom, true);
    assert.equal(alphaAvail.protocol, 'openai_compatible');
    assert.equal(alphaAvail.api_key_configured, true);
    assert.equal(betaAvail.protocol, 'anthropic_compatible');
    assert.equal(betaAvail.api_key_configured, true);
    assert.equal(JSON.stringify(availability).includes('sk-alpha-super-secret-value'), false);
    assert.equal(JSON.stringify(availability).includes('sk-beta-anthropic-secret-value'), false);

    // Disable beta and verify availability reflects enabled=false
    store.upsertCustomProvider({
      id: beta.id,
      name: beta.name,
      base_url: beta.base_url,
      protocol: beta.protocol,
      enabled: false,
      models: beta.models,
    });
    const availabilityAfterDisable = store.getProviderAvailability(envOverlay);
    const betaDisabled = availabilityAfterDisable[beta.id];
    assert.equal(betaDisabled?.enabled, false);

    // Delete alpha and verify beta remains intact
    assert.equal(store.deleteCustomProvider(alpha.id), true);
    assert.equal(store.getCustomProviders().length, 1);
    assert.equal(store.getCustomProviders()[0].id, beta.id);
  } finally {
    cleanup();
  }
});

test('seamlessly migrates legacy custom provider config into custom_legacy', () => {
  const { root, cleanup } = makeTempStore();
  try {
    const llmPath = path.join(root, 'data', 'llm_config.json');
    fs.mkdirSync(path.dirname(llmPath), { recursive: true });
    fs.writeFileSync(
      llmPath,
      JSON.stringify({
        primary: {
          provider: 'custom',
          model: 'my-legacy-model',
          base_url: 'https://legacy.example.com/v1',
          temperature: 0.2,
          max_tokens: 4096,
          reasoning_effort: 'none',
        },
        worker: {
          provider: 'openai',
          model: 'gpt-4o-mini',
          base_url: 'https://api.openai.com/v1',
          temperature: 0.1,
          max_tokens: 4096,
          reasoning_effort: 'none',
        },
      }),
      'utf-8',
    );

    const store = new WorkspaceAndConfigStore(root);
    const cfg = store.getLLMConfig();
    assert.equal(cfg.custom_providers.length, 1);
    assert.equal(cfg.custom_providers[0].id, 'custom_legacy');
    assert.equal(cfg.custom_providers[0].base_url, 'https://legacy.example.com/v1');
    assert.equal(cfg.custom_providers[0].models[0]?.id, 'my-legacy-model');
  } finally {
    cleanup();
  }
});

// =============================================================================
// 3. Provider-Aware Context Window Detection, Unknown Enforcement & Manual Override
// =============================================================================

test('resolves provider-aware context windows, marks unknown endpoints explicitly, and computes input budgets accurately', async () => {
  clearLiveCatalogCache();
  clearReferenceModelCatalogCache();

  // Same model family / different provider limits:
  const groqLlama = resolveModelContextWindow('groq', 'llama-3.3-70b-versatile');
  assert.equal(groqLlama.contextWindow, 131_072);
  assert.equal(groqLlama.source, 'provider_registry');

  const deepseekNative = resolveModelContextWindow('deepseek', 'deepseek-chat');
  assert.equal(deepseekNative.contextWindow, 128_000);
  assert.equal(deepseekNative.source, 'provider_registry');

  const openrouterLlama4 = resolveModelContextWindow('openrouter', 'meta-llama/llama-4-scout');
  const groqLlama4 = resolveModelContextWindow('groq', 'meta-llama/llama-4-scout');
  assert.equal(openrouterLlama4.contextWindow, 1_048_576);
  assert.equal(groqLlama4.contextWindow, 131_072);

  const openaiGpt52 = resolveModelContextWindow('openai', 'gpt-5.2');
  assert.equal(openaiGpt52.contextWindow, 400_000);

  // Custom / local providers without explicit metadata MUST be marked unknown (never guessed!)
  const customUnknown = resolveModelContextWindow('custom_internal', 'gpt-4o');
  assert.equal(customUnknown.contextWindow, null);
  assert.equal(customUnknown.source, 'unknown');
  assert.equal(deriveDetectionMode(customUnknown.source, customUnknown.contextWindow), 'unknown');

  const ollamaUnknown = resolveModelContextWindow('ollama', 'llama-3.3-70b');
  assert.equal(ollamaUnknown.contextWindow, null);
  assert.equal(ollamaUnknown.source, 'unknown');

  // Manual override always wins and is labeled 'manual'
  const manualOverride = resolveModelContextWindow('custom_internal', 'llama-3.3-70b', null, {
    manualContextWindow: 96_000,
  });
  assert.equal(manualOverride.contextWindow, 96_000);
  assert.equal(manualOverride.source, 'manual');
  assert.equal(deriveDetectionMode(manualOverride.source, manualOverride.contextWindow), 'manual');

  // Custom provider per-model metadata resolves via resolveModelMetadataFromCatalog
  const customMeta = await resolveModelMetadataFromCatalog({
    provider: 'custom_alpha',
    model: 'alpha-pro',
    customProvider: {
      id: 'custom_alpha',
      name: 'Alpha',
      base_url: 'https://alpha.example.com/v1',
      protocol: 'openai_compatible',
      enabled: true,
      models: [{ id: 'alpha-pro', context_window: 131_072, max_output_tokens: 16_384 }],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
  });
  assert.equal(customMeta.context_window, 131_072);
  assert.equal(customMeta.max_output_tokens, 16_384);
  assert.equal(customMeta.context_window_source, 'custom_provider_config');

  // Input budget accounting: total - reservedOutput - safetyBuffer = usableInputWindow
  const budget = computeInputBudgetMetrics({
    contextWindow: 128_000,
    maxOutputTokens: 16_384,
    configuredMaxTokens: 8_192,
    usedTokens: 20_000,
  });
  assert.equal(budget.reservedOutputTokens, 8_192);
  assert.ok(budget.safetyBufferTokens >= 512);
  assert.equal(
    budget.usableInputWindow,
    128_000 - budget.reservedOutputTokens - budget.safetyBufferTokens,
  );
  assert.equal(budget.remainingInputBudget, (budget.usableInputWindow ?? 0) - 20_000);
});

// =============================================================================
// 4. CRITICAL ACCEPTANCE TEST: Cross-Session Provider & Context Isolation
// =============================================================================

test('CRITICAL: changing provider/model/context in Session 1 never alters Session 2 configuration, budget, or history', async () => {
  const { root, store, cleanup } = makeTempStore();
  try {
    const now = new Date().toISOString();
    const session1Config = store.buildThreadSessionConfig({
      provider: 'openai',
      model: 'gpt-5.2',
      base_url: 'https://api.openai.com/v1',
      context_window: 400_000,
      context_window_source: 'provider_registry',
      max_output_tokens: 128_000,
      max_tokens: 16_384,
      reasoning_effort: 'high',
      access_mode: 'default',
    });
    const session2Config = store.buildThreadSessionConfig({
      provider: 'groq',
      model: 'llama-3.3-70b-versatile',
      base_url: 'https://api.groq.com/openai/v1',
      context_window: 131_072,
      context_window_source: 'provider_registry',
      max_output_tokens: 32_768,
      max_tokens: 8_192,
      reasoning_effort: 'none',
      access_mode: 'full_access',
    });

    store.saveThreads([
      {
        id: 'session-1',
        title: 'Session One',
        createdAt: now,
        updatedAt: now,
        messages: [{ id: 'm1', role: 'user', content: 'Hello from session 1', timestamp: now }],
        sessionConfig: session1Config,
      },
      {
        id: 'session-2',
        title: 'Session Two',
        createdAt: now,
        updatedAt: now,
        messages: [
          { id: 'm2', role: 'user', content: 'Running critical analysis in session 2', timestamp: now },
          { id: 'm3', role: 'assistant', content: 'Session 2 interim result', timestamp: now },
        ],
        sessionConfig: session2Config,
      },
    ]);

    // Concurrently mutate Session 1 AND global defaults while Session 2 is active
    await Promise.all([
      Promise.resolve().then(() =>
        store.updateThreadSessionConfig('session-1', {
          provider: 'deepseek',
          model: 'deepseek-reasoner',
          manual_context_window: 96_000,
          context_window: 96_000,
          context_window_source: 'manual',
          reasoning_effort: 'medium',
          access_mode: 'full_access',
        }),
      ),
      Promise.resolve().then(() =>
        store.saveLLMConfig({
          primary: {
            provider: 'anthropic',
            model: 'claude-sonnet-4-6',
            base_url: 'https://api.anthropic.com',
            temperature: 0.4,
            max_tokens: 12_000,
            reasoning_effort: 'high',
            context_window: 1_000_000,
            context_window_source: 'provider_registry',
            max_output_tokens: 64_000,
          },
        }),
      ),
    ]);

    // Reload from disk to verify persistence
    const reloadedStore = new WorkspaceAndConfigStore(root);
    const threads = reloadedStore.listThreads();
    const t1 = threads.find((t) => t.id === 'session-1')!;
    const t2 = threads.find((t) => t.id === 'session-2')!;

    // Session 1 reflects its new provider, model, and manual context window
    assert.equal(t1.sessionConfig?.provider, 'deepseek');
    assert.equal(t1.sessionConfig?.model, 'deepseek-reasoner');
    assert.equal(t1.sessionConfig?.manual_context_window, 96_000);
    assert.equal(t1.sessionConfig?.context_window, 96_000);

    // Session 2 MUST remain 100% untouched
    assert.equal(t2.sessionConfig?.provider, 'groq');
    assert.equal(t2.sessionConfig?.model, 'llama-3.3-70b-versatile');
    assert.equal(t2.sessionConfig?.base_url, 'https://api.groq.com/openai/v1');
    assert.equal(t2.sessionConfig?.context_window, 131_072);
    assert.equal(t2.sessionConfig?.manual_context_window ?? null, null);
    assert.equal(t2.sessionConfig?.access_mode, 'full_access');
    assert.equal(t2.messages.length, 2);
    assert.equal(t2.messages[0].content, 'Running critical analysis in session 2');

    // Compute frontend context usage snapshots for both threads against the mutated global llmConfig
    const globalCfg = reloadedStore.getLLMConfig() as unknown as PersistedLLMConfig;
    const snap1 = computeModelContextUsage({
      thread: t1 as unknown as ChatThread,
      llmConfig: globalCfg,
    });
    const snap2 = computeModelContextUsage({
      thread: t2 as unknown as ChatThread,
      llmConfig: globalCfg,
    });

    assert.equal(snap1.provider, 'deepseek');
    assert.equal(snap1.model, 'deepseek-reasoner');
    assert.equal(snap1.context_window, 96_000);
    assert.equal(snap1.detection_mode, 'manual');

    assert.equal(snap2.provider, 'groq');
    assert.equal(snap2.model, 'llama-3.3-70b-versatile');
    assert.equal(snap2.context_window, 131_072);
    assert.equal(snap2.detection_mode, 'auto');
  } finally {
    cleanup();
  }
});

// =============================================================================
// 5. Docker & Firecrawl Infrastructure Validation
// =============================================================================

test('Docker and Firecrawl Local infrastructure files define clean embedded firecrawl-local stack with no legacy containers or Settings UI clutter', () => {
  const repoRoot = path.resolve(process.cwd());
  const composePath = path.join(repoRoot, 'docker-compose.yml');
  const dockerfilePath = path.join(repoRoot, 'Dockerfile');
  const envExamplePath = path.join(repoRoot, '.env.example');
  const healthScriptPath = path.join(repoRoot, 'scripts', 'check-firecrawl.sh');

  assert.equal(fs.existsSync(composePath), true, 'docker-compose.yml must exist');
  assert.equal(fs.existsSync(dockerfilePath), true, 'Dockerfile must exist');
  assert.equal(fs.existsSync(healthScriptPath), true, 'scripts/check-firecrawl.sh must exist');

  const compose = fs.readFileSync(composePath, 'utf-8');
  assert.ok(compose.includes('nemofy-app:'), 'docker-compose.yml must define nemofy-app');
  assert.ok(compose.includes('healthcheck:'), 'docker-compose.yml must define healthchecks');
  assert.ok(compose.includes('restart: unless-stopped'), 'docker-compose.yml must configure restart policies');
  assert.ok(compose.includes('FIRECRAWL_API_URL'), 'nemofy-app must receive FIRECRAWL_API_URL');
  for (const removedLegacyService of ['nuq-postgres:', 'firecrawl-worker:', 'playwright-service:']) {
    assert.equal(
      compose.includes(removedLegacyService),
      false,
      `docker-compose.yml should not contain old legacy service ${removedLegacyService}`,
    );
  }
  assert.equal(/sk-[a-zA-Z0-9]{20,}/.test(compose), false, 'docker-compose.yml must never contain hardcoded API keys');

  const dockerfile = fs.readFileSync(dockerfilePath, 'utf-8');
  assert.ok(dockerfile.includes('HEALTHCHECK'), 'Dockerfile must define HEALTHCHECK');
  assert.ok(dockerfile.includes('/api/health'), 'Dockerfile HEALTHCHECK must probe /api/health');
  assert.ok(dockerfile.includes('firecrawl-builder'), 'Dockerfile must include Rust builder stage for firecrawl-local');
  assert.ok(dockerfile.includes('firecrawl-local'), 'Dockerfile must compile and ship firecrawl-local binary');

  const installScriptPath = path.join(repoRoot, 'scripts', 'install-firecrawl-local.sh');
  const runScriptPath = path.join(repoRoot, 'scripts', 'run-firecrawl-local.sh');
  assert.equal(fs.existsSync(installScriptPath), true, 'scripts/install-firecrawl-local.sh must exist');
  assert.equal(fs.existsSync(runScriptPath), true, 'scripts/run-firecrawl-local.sh must exist');
  assert.ok(
    fs.readFileSync(installScriptPath, 'utf-8').includes('firecrawl-local'),
    'install-firecrawl-local.sh must install official firecrawl-local',
  );
  const runScript = fs.readFileSync(runScriptPath, 'utf-8');
  assert.ok(
    runScript.includes('"serve"') && runScript.includes('"--host"') && runScript.includes('firecrawl-local'),
    'run-firecrawl-local.sh must launch firecrawl-local serve',
  );
  assert.ok(
    fs.readFileSync(healthScriptPath, 'utf-8').includes('/health'),
    'check-firecrawl.sh must probe official /health endpoint',
  );

  const envExample = fs.readFileSync(envExamplePath, 'utf-8');
  assert.ok(envExample.includes('FIRECRAWL_API_URL='), '.env.example must document FIRECRAWL_API_URL');
  assert.ok(envExample.includes('FIRECRAWL_API_KEY='), '.env.example must document FIRECRAWL_API_KEY');
  assert.ok(envExample.includes('FIRECRAWL_CLOUD='), '.env.example must document FIRECRAWL_CLOUD');
  assert.ok(envExample.includes('FIRECRAWL_CHROME_PATH='), '.env.example must document FIRECRAWL_CHROME_PATH');
  assert.ok(envExample.includes('ALLOW_UNSAFE_LOCAL_BASH='), '.env.example must document ALLOW_UNSAFE_LOCAL_BASH');
  assert.equal(envExample.includes('FIRECRAWL_POSTGRES_'), false, '.env.example must not retain old Postgres Firecrawl vars');

  // Verify metadata.json reflects Nemofy Agent (not unrelated placeholder metadata)
  const metadata = JSON.parse(fs.readFileSync(path.join(repoRoot, 'metadata.json'), 'utf-8'));
  assert.equal(metadata.name, 'Nemofy Agent');
  assert.equal(/n8n/i.test(JSON.stringify(metadata)), false, 'metadata.json must not reference unrelated n8n app');

  // Verify Firecrawl Web Research card is removed from SettingsView.tsx
  const settingsView = fs.readFileSync(
    path.join(repoRoot, 'Nemofy Agent', 'src', 'components', 'SettingsView.tsx'),
    'utf-8',
  );
  assert.equal(
    settingsView.includes('firecrawl-settings-section'),
    false,
    'Firecrawl Web Research card must be removed from SettingsView.tsx',
  );
  assert.equal(
    settingsView.includes('Firecrawl Web Research'),
    false,
    'Firecrawl Web Research label must not appear in SettingsView.tsx',
  );

  // Verify every `npm run build` automatically ensures firecrawl-local AND Chromium are built/ready
  const pkg = JSON.parse(fs.readFileSync(path.join(repoRoot, 'package.json'), 'utf-8'));
  assert.ok(
    String(pkg.scripts?.build || '').includes('build-runtime-deps.sh'),
    'package.json build script must run scripts/build-runtime-deps.sh alongside vite build',
  );
  const buildDepsScript = fs.readFileSync(path.join(repoRoot, 'scripts', 'build-runtime-deps.sh'), 'utf-8');
  assert.ok(buildDepsScript.includes('install-firecrawl-local.sh'), 'build-runtime-deps.sh must build/install firecrawl-local');
  assert.ok(buildDepsScript.includes('install-chromium.sh'), 'build-runtime-deps.sh must install/verify Chromium');
  assert.equal(fs.existsSync(path.join(repoRoot, 'scripts', 'install-chromium.sh')), true, 'scripts/install-chromium.sh must exist');
  assert.ok(dockerfile.includes('chromium --version'), 'Dockerfile must verify Chromium alongside firecrawl-local');
});

test('Docker setup hardening prevents context pollution, DNS/apt failures, missing README.md, PID 1 leaks, and sandbox lockouts', () => {
  const repoRoot = path.resolve(process.cwd());
  const dockerignorePath = path.join(repoRoot, '.dockerignore');
  const dockerfile = fs.readFileSync(path.join(repoRoot, 'Dockerfile'), 'utf-8');
  const compose = fs.readFileSync(path.join(repoRoot, 'docker-compose.yml'), 'utf-8');
  const sandboxDockerfile = fs.readFileSync(path.join(repoRoot, 'Agent-Core', 'Dockerfile.sandbox'), 'utf-8');

  // 1. .dockerignore must exist and exclude host build outputs, venvs, and secrets
  assert.equal(fs.existsSync(dockerignorePath), true, '.dockerignore must exist');
  const dockerignore = fs.readFileSync(dockerignorePath, 'utf-8');
  for (const requiredEntry of ['node_modules', '.venv', 'dist', '.env', 'data', 'workspace', '.git', '.firecrawl']) {
    assert.ok(
      dockerignore.split(/\r?\n/).some((line) => line.trim() === requiredEntry),
      `.dockerignore must exclude ${requiredEntry}`,
    );
  }
  assert.ok(dockerignore.includes('!.env.example'), '.dockerignore should keep .env.example');

  // 2. Dockerfile must include apt IPv4/retry resilience, libssl-dev/perl, tini/procps, README.md before pip install, production mode, and direct tsx CMD
  assert.ok(dockerfile.includes('Acquire::ForceIPv4'), 'Dockerfile must force IPv4 for apt network resilience');
  assert.ok(dockerfile.includes('Acquire::Retries'), 'Dockerfile must configure apt retries');
  assert.ok(dockerfile.includes('libssl-dev'), 'firecrawl-builder must install libssl-dev');
  assert.ok(dockerfile.includes('CARGO_PROFILE_RELEASE_LTO=thin'), 'firecrawl-builder must avoid OOM-prone fat LTO');
  assert.ok(dockerfile.includes('tini'), 'runtime stage must install tini for PID 1 reaping');
  assert.ok(dockerfile.includes('procps'), 'runtime stage must install procps for pkill');
  assert.ok(
    dockerfile.includes('COPY pyproject.toml README.md ./'),
    'Dockerfile must copy README.md alongside pyproject.toml before pip install -e .',
  );
  assert.ok(
    dockerfile.includes('SKIP_RUNTIME_DEPS_BUILD=1 npm run build'),
    'Dockerfile frontend build must skip redundant host runtime-dep installers',
  );
  assert.ok(dockerfile.includes('ENV NODE_ENV=production'), 'Dockerfile must set NODE_ENV=production after build');
  assert.ok(dockerfile.includes('ENTRYPOINT ["/usr/bin/tini", "--"]'), 'Dockerfile must use tini as ENTRYPOINT');
  assert.ok(
    dockerfile.includes('CMD ["./node_modules/.bin/tsx", "server.ts"]'),
    'Dockerfile must execute local tsx directly instead of npx',
  );
  assert.equal(dockerfile.includes('CMD ["npx"'), false, 'Dockerfile must not use npx as container CMD');

  // 3. docker-compose.yml must configure host build network, shm_size, init, HOST=0.0.0.0, and SANDBOX_BACKEND
  assert.ok(compose.includes('network: host'), 'docker-compose.yml build must use host network to avoid bridge DNS failures');
  assert.ok(compose.includes('init: true'), 'docker-compose.yml must enable init process reaping');
  assert.ok(compose.includes('shm_size:'), 'docker-compose.yml must increase /dev/shm for Chromium');
  assert.ok(compose.includes('HOST: "0.0.0.0"'), 'docker-compose.yml must bind HOST=0.0.0.0');
  assert.ok(
    compose.includes('SANDBOX_BACKEND: ${SANDBOX_BACKEND:-local_isolated}'),
    'docker-compose.yml must default SANDBOX_BACKEND to local_isolated inside container',
  );

  // 4. Sandbox Dockerfile must pin Bookworm (avoiding Debian Trixie libasound2/libatk renames) and set apt resilience
  assert.ok(
    sandboxDockerfile.includes('FROM python:3.12-slim-bookworm'),
    'Agent-Core/Dockerfile.sandbox must pin python:3.12-slim-bookworm',
  );
  assert.ok(
    sandboxDockerfile.includes('Acquire::ForceIPv4'),
    'Agent-Core/Dockerfile.sandbox must configure apt IPv4/retry resilience',
  );
});



