import 'dotenv/config';
import crypto from 'node:crypto';
import express, { type NextFunction, type Request, type Response } from 'express';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { AuthStore, parseCookieHeader } from './server/authStore.ts';
import { AgentCoreProcessManager } from './server/agentProcessManager.ts';
import { FirecrawlLocalProcessManager } from './server/firecrawlProcessManager.ts';
import {
  WorkspaceAndConfigStore,
  maskEnvSecret,
  normalizeAccessMode,
  normalizeProviderId,
  validateMaxTurns,
  type AccessoryLLMConfig,
  type ChatAttachmentRecord,
  type ChatMessageRecord,
  type ChatThreadRecord,
  type ContextUsageSnapshot,
  type CustomProviderConfig,
  type CustomProviderModelConfig,
  type CustomProviderProtocol,
  type ExecutionAccessMode,
  type FileChangeRecord,
  type MediaModality,
  type PlanningTimeoutConfig,
  type ThreadSessionConfig,
  type ToolEventRecord,
} from './server/configStore.ts';
import {
  listProviderModelsDirect,
  probeProviderModelDirect,
  resolveModelMetadataUnified,
  resolveProviderApiKey,
  validateOutboundProviderUrl,
} from './server/providerCatalog.ts';
import { ChatJobOrchestrator } from './server/jobOrchestrator.ts';
import {
  ProfileSecretVaultStore,
  deriveCustomProviderEnvVar,
  isAllowedProfileSecretKey,
} from './server/profileVault.ts';
import { resolveServerListenConfig } from './server/serverConfig.ts';

const cwd = process.cwd();
const { port: PORT, host: HOST } = resolveServerListenConfig();
const SESSION_COOKIE = 'nemofy_session';

const authStore = new AuthStore();
const store = new WorkspaceAndConfigStore(cwd);
const agentManager = new AgentCoreProcessManager(
  cwd,
  undefined,
  () => authStore.getAgentApiToken(),
  () => {
    const cfg = store.getLLMConfig();
    return {
      ALLOW_UNSAFE_LOCAL_BASH: String(cfg.allow_unsafe_local_bash),
      ...(cfg.firecrawl_api_url ? { FIRECRAWL_API_URL: cfg.firecrawl_api_url } : {}),
    };
  },
);
const firecrawlManager = new FirecrawlLocalProcessManager(cwd, undefined, () => {
  const cfg = store.getLLMConfig();
  return {
    ...(cfg.firecrawl_api_url ? { FIRECRAWL_API_URL: cfg.firecrawl_api_url } : {}),
  };
});
const profileVault = new ProfileSecretVaultStore(
  path.join(cwd, 'data', 'vaults'),
  () => authStore.getVaultKeyMaterial(),
);
const migratedProfiles = new Set<string>();

function loadProfileSecrets(username: string): Record<string, string> {
  const cleanUser = username.trim().toLowerCase();
  if (!migratedProfiles.has(cleanUser)) {
    // Automatically migrate any legacy plaintext provider keys from .env into admin's encrypted vault
    if (cleanUser === authStore.getUsername().toLowerCase()) {
      try {
        profileVault.migrateLegacyEnvSecrets(cleanUser, store.envFilePath, process.env);
      } catch (err) {
        console.warn('[profile-vault] Legacy .env migration skipped:', err instanceof Error ? err.message : String(err));
      }
    }
    migratedProfiles.add(cleanUser);
  }
  try {
    return profileVault.getDecryptedSecrets(cleanUser);
  } catch (err) {
    console.warn(
      '[profile-vault] Unable to decrypt vault for user (vault key may have rotated):',
      err instanceof Error ? err.message : String(err),
    );
    return {};
  }
}

function buildVaultSummary(username: string) {
  let entries: ReturnType<typeof profileVault.listSecretMetadata> = [];
  try {
    entries = profileVault.listSecretMetadata(username);
  } catch (err) {
    console.warn(
      '[profile-vault] Unable to list vault metadata:',
      err instanceof Error ? err.message : String(err),
    );
  }
  return {
    username,
    encrypted: true,
    algorithm: 'AES-256-GCM',
    entries,
  };
}

async function syncProfileSecretsToCore(username: string, waitForReady = false): Promise<boolean> {
  const secrets = loadProfileSecrets(username);
  let resp = await callAgentCore('/config/secrets', {
    method: 'POST',
    body: { secrets },
    timeoutMs: 5000,
  });
  if (!resp.ok && (resp.status === 503 || resp.status === 401)) {
    const ready = await agentManager.ensureRunning(waitForReady ? 15_000 : 4_000);
    if (ready) {
      resp = await callAgentCore('/config/secrets', {
        method: 'POST',
        body: { secrets },
        timeoutMs: 5000,
      });
    }
  }
  return resp.ok;
}

function getAgentApiToken(): string {
  return authStore.getAgentApiToken();
}

async function enrichRoleLimits<
  T extends {
    provider: string;
    model: string;
    base_url?: string;
    context_window?: number | null;
    context_window_source?: string | null;
    manual_context_window?: number | null;
    max_output_tokens?: number | null;
  },
>(
  roleCfg: T,
  options: {
    env?: Record<string, string | undefined>;
    allowLiveFetch?: boolean;
    forceRefresh?: boolean;
    role?: 'primary' | 'worker';
    customProviders?: CustomProviderConfig[];
  } = {},
): Promise<T> {
  if (!roleCfg.model || !roleCfg.model.trim()) return roleCfg;
  if (roleCfg.manual_context_window && roleCfg.manual_context_window >= 512) {
    return {
      ...roleCfg,
      context_window: Math.round(roleCfg.manual_context_window),
      context_window_source: 'manual',
    };
  }
  if (!options.forceRefresh && roleCfg.context_window && roleCfg.max_output_tokens && roleCfg.context_window_source) {
    return roleCfg;
  }
  const normalizedProv = normalizeProviderId(roleCfg.provider);
  const customProvider =
    (options.customProviders || store.getCustomProviders()).find((cp) => cp.id === normalizedProv) || null;
  const meta = await resolveModelMetadataUnified({
    provider: normalizedProv,
    model: roleCfg.model,
    base_url: roleCfg.base_url || customProvider?.base_url,
    customProvider,
    role: options.role,
    env: options.env,
    manualContextWindow: roleCfg.manual_context_window,
    explicitContextWindow: options.forceRefresh ? null : roleCfg.context_window,
    explicitMaxOutputTokens: options.forceRefresh ? null : roleCfg.max_output_tokens,
    allowLiveFetch: options.allowLiveFetch ?? false,
    forceRefresh: options.forceRefresh ?? false,
  });
  return {
    ...roleCfg,
    context_window: meta.context_window,
    context_window_source: meta.context_window_source,
    max_output_tokens: meta.max_output_tokens,
  };
}

async function enrichLLMConfigLimits(
  cfg: ReturnType<typeof store.getLLMConfig>,
  options: { env?: Record<string, string | undefined>; allowLiveFetch?: boolean } = {},
): Promise<ReturnType<typeof store.getLLMConfig>> {
  const opts = { ...options, customProviders: cfg.custom_providers };
  const [primary, worker, image, video, audio] = await Promise.all([
    enrichRoleLimits(cfg.primary, { ...opts, role: 'primary' }),
    enrichRoleLimits(cfg.worker, { ...opts, role: 'worker' }),
    enrichRoleLimits(cfg.accessories.image, { ...opts, role: 'worker' }),
    enrichRoleLimits(cfg.accessories.video, { ...opts, role: 'worker' }),
    enrichRoleLimits(cfg.accessories.audio, { ...opts, role: 'worker' }),
  ]);
  return {
    ...cfg,
    primary,
    worker,
    accessories: { image, video, audio },
  };
}

async function resolveThreadExecutionPayload(params: {
  thread: ChatThreadRecord | null;
  globalCfg: ReturnType<typeof store.getLLMConfig>;
  profileSecrets: Record<string, string>;
  reasoningEffortOverride?: string;
  accessModeOverride?: ExecutionAccessMode;
}) {
  const { thread, globalCfg, profileSecrets, reasoningEffortOverride, accessModeOverride } = params;
  const baseSession: ThreadSessionConfig = thread?.sessionConfig
    ? { ...thread.sessionConfig }
    : store.buildThreadSessionConfig(undefined, globalCfg);

  const requestedEffort =
    typeof reasoningEffortOverride === 'string' && reasoningEffortOverride.trim()
      ? reasoningEffortOverride.trim().toLowerCase()
      : baseSession.reasoning_effort;
  const effectiveAccessMode: ExecutionAccessMode =
    accessModeOverride === 'default' || accessModeOverride === 'full_access'
      ? accessModeOverride
      : baseSession.access_mode;

  const sessionPrimary = await enrichRoleLimits(
    { ...baseSession, reasoning_effort: requestedEffort },
    {
      env: profileSecrets,
      role: 'primary',
      customProviders: globalCfg.custom_providers,
    },
  );
  const customPrimary =
    globalCfg.custom_providers.find((cp) => cp.id === normalizeProviderId(sessionPrimary.provider)) || null;
  const primaryApiKey = resolveProviderApiKey(
    sessionPrimary.provider,
    'primary',
    profileSecrets,
  );
  const timeoutCfg = globalCfg.planning_timeout;

  const customWorker =
    globalCfg.custom_providers.find((cp) => cp.id === normalizeProviderId(globalCfg.worker.provider)) || null;
  const workerApiKey = resolveProviderApiKey(
    globalCfg.worker.provider,
    'worker',
    profileSecrets,
  );

  return {
    sessionConfig: {
      ...sessionPrimary,
      access_mode: effectiveAccessMode,
    } as ThreadSessionConfig,
    effectiveAccessMode,
    primaryModelPayload: {
      provider: sessionPrimary.provider,
      model: sessionPrimary.model,
      base_url: sessionPrimary.base_url || customPrimary?.base_url || '',
      temperature: sessionPrimary.temperature,
      max_tokens: sessionPrimary.max_tokens,
      reasoning_effort: sessionPrimary.reasoning_effort,
      ...(customPrimary?.protocol ? { protocol: customPrimary.protocol } : {}),
      ...(primaryApiKey ? { api_key: primaryApiKey } : {}),
      ...(sessionPrimary.context_window ? { context_window: sessionPrimary.context_window } : {}),
      ...(sessionPrimary.context_window_source
        ? { context_window_source: sessionPrimary.context_window_source }
        : {}),
      ...(sessionPrimary.manual_context_window
        ? { manual_context_window: sessionPrimary.manual_context_window }
        : {}),
      ...(sessionPrimary.max_output_tokens ? { max_output_tokens: sessionPrimary.max_output_tokens } : {}),
      timeout_seconds: customPrimary?.timeout_seconds ?? timeoutCfg.read_timeout_seconds,
      connect_timeout_seconds: timeoutCfg.connect_timeout_seconds,
      max_retries: timeoutCfg.max_retries,
      fallback_on_timeout: timeoutCfg.fallback_on_timeout,
      ...(timeoutCfg.fallback_provider ? { fallback_provider: timeoutCfg.fallback_provider } : {}),
      ...(timeoutCfg.fallback_model ? { fallback_model: timeoutCfg.fallback_model } : {}),
    },
    workerModelPayload: {
      provider: globalCfg.worker.provider,
      model: globalCfg.worker.model,
      base_url: globalCfg.worker.base_url || customWorker?.base_url || '',
      temperature: globalCfg.worker.temperature,
      max_tokens: globalCfg.worker.max_tokens,
      reasoning_effort: globalCfg.worker.reasoning_effort,
      ...(customWorker?.protocol ? { protocol: customWorker.protocol } : {}),
      ...(workerApiKey ? { api_key: workerApiKey } : {}),
      ...(globalCfg.worker.context_window ? { context_window: globalCfg.worker.context_window } : {}),
      ...(globalCfg.worker.context_window_source
        ? { context_window_source: globalCfg.worker.context_window_source }
        : {}),
      ...(globalCfg.worker.manual_context_window
        ? { manual_context_window: globalCfg.worker.manual_context_window }
        : {}),
      ...(globalCfg.worker.max_output_tokens ? { max_output_tokens: globalCfg.worker.max_output_tokens } : {}),
      timeout_seconds: customWorker?.timeout_seconds ?? timeoutCfg.read_timeout_seconds,
      connect_timeout_seconds: timeoutCfg.connect_timeout_seconds,
      max_retries: timeoutCfg.max_retries,
    },
  };
}

function cleanThreadTitle(raw: string): string {
  return raw
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[*_~`#>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 56);
}

function extractSessionToken(req: Request): string | null {
  const authHeader = req.headers.authorization;
  if (typeof authHeader === 'string' && authHeader.toLowerCase().startsWith('bearer ')) {
    const tok = authHeader.slice(7).trim();
    if (tok) return tok;
  }
  const cookies = parseCookieHeader(req.headers.cookie);
  return cookies[SESSION_COOKIE] || null;
}

function isSecureRequest(req: Request): boolean {
  const forwardedProto = String(req.headers['x-forwarded-proto'] || '').toLowerCase();
  return req.secure || forwardedProto.includes('https');
}

function setSessionCookie(req: Request, res: Response, token: string): void {
  const secure = isSecureRequest(req);
  const parts = [
    `${SESSION_COOKIE}=${encodeURIComponent(token)}`,
    'Path=/',
    'HttpOnly',
    secure ? 'SameSite=None' : 'SameSite=Lax',
    ...(secure ? ['Secure', 'Partitioned'] : []),
    `Max-Age=${24 * 60 * 60}`,
  ];
  res.setHeader('Set-Cookie', parts.join('; '));
}

function clearSessionCookie(req: Request, res: Response): void {
  const secure = isSecureRequest(req);
  const parts = [
    `${SESSION_COOKIE}=`,
    'Path=/',
    'HttpOnly',
    secure ? 'SameSite=None' : 'SameSite=Lax',
    ...(secure ? ['Secure', 'Partitioned'] : []),
    'Max-Age=0',
  ];
  res.setHeader('Set-Cookie', parts.join('; '));
}

function isSameOriginMutation(req: Request): boolean {
  const method = req.method.toUpperCase();
  if (method === 'GET' || method === 'HEAD' || method === 'OPTIONS') return true;
  // Bearer token auth or explicit X-Requested-With header is not vulnerable to cross-site form CSRF
  const authHeader = req.headers.authorization;
  if (typeof authHeader === 'string' && authHeader.toLowerCase().startsWith('bearer ')) return true;
  if (req.headers['x-requested-with']) return true;
  const origin = req.headers.origin;
  if (!origin) return true;
  try {
    const originHost = new URL(origin).host.toLowerCase();
    const expectedHost = String(req.headers['x-forwarded-host'] || req.headers.host || '').split(',')[0].trim().toLowerCase();
    return Boolean(expectedHost && originHost === expectedHost);
  } catch {
    return false;
  }
}

function requireSession(req: Request, res: Response, next: NextFunction): void {
  const token = extractSessionToken(req);
  const session = authStore.verifySessionToken(token);
  if (!session) {
    res.status(401).json({ error: 'Authentication required. Please sign in.' });
    return;
  }
  if (!isSameOriginMutation(req)) {
    res.status(403).json({ error: 'Cross-origin request blocked.' });
    return;
  }
  res.locals.session = session;
  next();
}

function getSessionUsername(res: Response): string {
  const session = res.locals.session as { username?: string } | undefined;
  return session?.username || authStore.getUsername();
}

async function callAgentCore(
  endpoint: string,
  options: { method?: string; body?: unknown; timeoutMs?: number } = {},
): Promise<{ ok: boolean; status: number; data: Record<string, unknown> }> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  const apiToken = getAgentApiToken();
  if (apiToken) {
    headers.Authorization = `Bearer ${apiToken}`;
  }

  try {
    const resp = await fetch(`${agentManager.baseUrl}${endpoint}`, {
      method: options.method || 'GET',
      headers,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
      signal: AbortSignal.timeout(options.timeoutMs ?? 120_000),
    });
    const text = await resp.text();
    let parsed: Record<string, unknown> = {};
    try {
      parsed = text ? (JSON.parse(text) as Record<string, unknown>) : {};
    } catch {
      parsed = { detail: text };
    }
    return { ok: resp.ok, status: resp.status, data: parsed };
  } catch (err) {
    void agentManager.ensureRunning(0);
    const status = agentManager.getStatus();
    const detail =
      status.state === 'bootstrapping'
        ? 'Python Agent-Core is setting up its virtual environment. Please retry in a few seconds.'
        : status.state === 'starting'
          ? 'Python Agent-Core is starting up. Please retry in a moment.'
          : `Python Agent-Core service unreachable: ${err instanceof Error ? err.message : String(err)}`;
    return {
      ok: false,
      status: 503,
      data: { detail },
    };
  }
}

async function startServer(): Promise<void> {
  const app = express();
  app.disable('x-powered-by');
  app.use(express.json({ limit: '5mb' }));

  const jobOrchestrator = new ChatJobOrchestrator({
    store,
    callAgentCore,
    pollIntervalMs: 650,
  });

  // ---------------------------------------------------------------------------
  // Public Health & Auth Endpoints
  // ---------------------------------------------------------------------------
  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'ok',
      service: 'nemofy-gateway',
      agent_core: agentManager.getStatus().state,
    });
  });

  app.get('/api/auth/me', (req, res) => {
    const token = extractSessionToken(req);
    const session = authStore.verifySessionToken(token);
    if (!session) {
      res.json({ authenticated: false, needsSetup: authStore.needsSetup() });
      return;
    }
    const secrets = loadProfileSecrets(session.username);
    void syncProfileSecretsToCore(session.username);
    res.json({
      authenticated: true,
      needsSetup: false,
      username: session.username,
      updatedAt: authStore.getUpdatedAt(),
      vaultKeysLoaded: Object.keys(secrets).length,
    });
  });

  app.post('/api/auth/setup', async (req, res) => {
    if (!isSameOriginMutation(req)) {
      res.status(403).json({ error: 'Cross-origin request blocked.' });
      return;
    }
    if (!authStore.needsSetup()) {
      res.status(403).json({ error: 'Administrator account is already configured.' });
      return;
    }
    const { username, password } = (req.body || {}) as { username?: string; password?: string };
    const result = authStore.setupInitialPassword(password || '', username);
    if (!result.ok || !result.token || !result.username) {
      res.status(400).json({ error: result.error || 'Unable to complete initial setup.' });
      return;
    }
    setSessionCookie(req, res, result.token);
    const secrets = loadProfileSecrets(result.username);
    await syncProfileSecretsToCore(result.username);
    res.status(201).json({
      authenticated: true,
      needsSetup: false,
      username: result.username,
      token: result.token,
      vaultKeysLoaded: Object.keys(secrets).length,
    });
  });

  app.post('/api/auth/login', async (req, res) => {
    if (authStore.needsSetup()) {
      res.status(403).json({
        error: 'Initial administrator setup is required before signing in.',
        needsSetup: true,
      });
      return;
    }
    const { username, password } = (req.body || {}) as { username?: string; password?: string };
    if (!username || !password) {
      res.status(400).json({ error: 'Username and password are required.' });
      return;
    }
    if (!authStore.verifyCredentials(username, password)) {
      res.status(401).json({ error: 'Invalid username or password.' });
      return;
    }
    const authenticatedUser = authStore.getUsername();
    const token = authStore.createSessionToken();
    setSessionCookie(req, res, token);

    // Automatically unlock and hydrate all encrypted API keys for this user profile
    const secrets = loadProfileSecrets(authenticatedUser);
    await syncProfileSecretsToCore(authenticatedUser);

    res.json({
      authenticated: true,
      username: authenticatedUser,
      token,
      vaultKeysLoaded: Object.keys(secrets).length,
    });
  });

  app.post('/api/auth/logout', (req, res) => {
    const token = extractSessionToken(req);
    const session = authStore.verifySessionToken(token);
    clearSessionCookie(req, res);
    if (session && isSameOriginMutation(req)) {
      authStore.revokeSessionToken(token);
      void callAgentCore('/config/secrets', { method: 'DELETE', timeoutMs: 3000 });
    }
    res.json({ ok: true });
  });

  // ---------------------------------------------------------------------------
  // Protected Endpoints (All require valid session!)
  // ---------------------------------------------------------------------------
  app.post('/api/auth/change-password', requireSession, (req, res) => {
    const { currentPassword, newPassword } = (req.body || {}) as {
      currentPassword?: string;
      newPassword?: string;
    };
    if (!currentPassword || !newPassword) {
      res.status(400).json({ error: 'Current password and new password are required.' });
      return;
    }
    const result = authStore.changePassword(currentPassword, newPassword);
    if (!result.ok || !result.token) {
      res.status(400).json({ error: result.error || 'Password update failed.' });
      return;
    }
    setSessionCookie(req, res, result.token);
    res.json({ ok: true, token: result.token, message: 'Password updated successfully.' });
  });

  app.get('/api/agent/health', requireSession, async (_req, res) => {
    const health = await callAgentCore('/health', { timeoutMs: 3000 });
    res.json({
      connected: health.ok,
      core: health.ok ? health.data : null,
      error: health.ok ? null : String(health.data.detail || 'Agent-Core offline'),
    });
  });

  // ---------------------------------------------------------------------------
  // Chat Threads & Pure-Python Agent-Core Execution
  // ---------------------------------------------------------------------------
  app.get('/api/threads', requireSession, (_req, res) => {
    res.json({ threads: store.listThreads() });
  });

  app.post('/api/threads', requireSession, async (req, res) => {
    const { title, sessionConfig } = (req.body || {}) as {
      title?: string;
      sessionConfig?: Partial<ThreadSessionConfig>;
    };
    const username = getSessionUsername(res);
    const profileSecrets = loadProfileSecrets(username);
    const globalCfg = await enrichLLMConfigLimits(store.getLLMConfig(), { env: profileSecrets });
    const initialSession = store.buildThreadSessionConfig(sessionConfig, globalCfg);
    const enrichedSession = await enrichRoleLimits(initialSession, {
      env: profileSecrets,
      role: 'primary',
      customProviders: globalCfg.custom_providers,
    });
    const now = new Date().toISOString();
    const thread: ChatThreadRecord = {
      id: `thread-${crypto.randomBytes(5).toString('hex')}`,
      title: (title || 'New Agent Session').trim().slice(0, 80),
      createdAt: now,
      updatedAt: now,
      messages: [],
      pendingApproval: null,
      sessionConfig: {
        ...enrichedSession,
        access_mode: initialSession.access_mode,
      },
    };
    const threads = [thread, ...store.listThreads()];
    store.saveThreads(threads);
    res.status(201).json({ thread });
  });

  app.patch('/api/threads/:id/config', requireSession, async (req, res) => {
    const id = req.params.id;
    const body = (req.body || {}) as Partial<ThreadSessionConfig> & { force_refresh?: boolean };
    const username = getSessionUsername(res);
    const profileSecrets = loadProfileSecrets(username);
    const globalCfg = store.getLLMConfig();

    const patch: Partial<ThreadSessionConfig> = {};
    if (typeof body.provider === 'string' && body.provider.trim()) {
      const nextProv = normalizeProviderId(body.provider);
      const customMatch = globalCfg.custom_providers.find((cp) => cp.id === nextProv);
      if (customMatch && !customMatch.enabled) {
        res.status(400).json({ error: `Custom provider '${customMatch.name}' is currently disabled.` });
        return;
      }
      patch.provider = nextProv;
    }
    if (typeof body.model === 'string') patch.model = body.model.trim();
    if (typeof body.base_url === 'string') patch.base_url = body.base_url.trim().replace(/\/+$/, '');
    if (typeof body.temperature === 'number') patch.temperature = body.temperature;
    if (typeof body.max_tokens === 'number') patch.max_tokens = body.max_tokens;
    if (typeof body.reasoning_effort === 'string') patch.reasoning_effort = body.reasoning_effort.trim().toLowerCase();
    if (body.access_mode === 'default' || body.access_mode === 'full_access') {
      patch.access_mode = body.access_mode;
    }
    if ('manual_context_window' in body) {
      if (body.manual_context_window === null || body.manual_context_window === undefined) {
        patch.manual_context_window = null;
        patch.context_window = null;
        patch.context_window_source = null;
      } else {
        const num = Number(body.manual_context_window);
        if (!Number.isFinite(num) || num < 512) {
          res.status(400).json({ error: 'Manual context window must be at least 512 tokens.' });
          return;
        }
        patch.manual_context_window = Math.round(num);
        patch.context_window = Math.round(num);
        patch.context_window_source = 'manual';
      }
    }
    if ('context_window' in body && !('manual_context_window' in body)) {
      patch.context_window = body.context_window;
    }
    if ('context_window_source' in body && !('manual_context_window' in body)) {
      patch.context_window_source = body.context_window_source;
    }
    if ('max_output_tokens' in body) {
      patch.max_output_tokens = body.max_output_tokens;
    }

    const updated = store.updateThreadSessionConfig(id, patch);
    if (!updated || !updated.sessionConfig) {
      res.status(404).json({ error: 'Thread not found.' });
      return;
    }

    const enriched = await enrichRoleLimits(updated.sessionConfig, {
      env: profileSecrets,
      allowLiveFetch: Boolean(body.force_refresh),
      forceRefresh: Boolean(body.force_refresh),
      role: 'primary',
      customProviders: globalCfg.custom_providers,
    });

    const finalThread = store.updateThreadSessionConfig(id, {
      context_window: enriched.context_window ?? null,
      context_window_source: enriched.context_window_source ?? 'unknown',
      max_output_tokens: enriched.max_output_tokens ?? null,
      ...(enriched.manual_context_window !== undefined
        ? { manual_context_window: enriched.manual_context_window }
        : {}),
    });

    res.json({ thread: finalThread || updated });
  });

  app.delete('/api/threads/:id', requireSession, async (req, res) => {
    const id = req.params.id;
    jobOrchestrator.markThreadDeleted(id);
    await jobOrchestrator.cancelThreadJob(id);
    const threads = store.listThreads().filter((t) => t.id !== id);
    store.saveThreads(threads);
    res.json({ ok: true });
  });

  // ---------------------------------------------------------------------------
  // Multimodal Media Upload & Authenticated Serving (image / video / audio)
  // ---------------------------------------------------------------------------
  app.post(
    '/api/media/upload',
    requireSession,
    express.raw({ type: '*/*', limit: '42mb' }),
    (req, res) => {
      try {
        const rawBuf = Buffer.isBuffer(req.body) ? req.body : Buffer.alloc(0);
        const rawThreadId = String(req.headers['x-thread-id'] || req.query.threadId || 'draft');
        const rawFilenameHeader = String(req.headers['x-file-name'] || req.query.filename || 'upload');
        let decodedFilename = rawFilenameHeader;
        try {
          decodedFilename = decodeURIComponent(rawFilenameHeader);
        } catch {
          decodedFilename = rawFilenameHeader;
        }
        const mimeType = String(req.headers['x-file-type'] || req.headers['content-type'] || '');
        const saved = store.saveUploadedMedia({
          threadId: rawThreadId,
          filename: decodedFilename,
          mimeType,
          buffer: rawBuf,
        });
        res.status(201).json({
          attachment: {
            id: saved.id,
            name: saved.name,
            kind: saved.kind,
            mimeType: saved.mimeType,
            size: saved.size,
            url: saved.url,
            storageKey: saved.storageKey,
          },
        });
      } catch (err) {
        res.status(400).json({
          error: err instanceof Error ? err.message : 'Unable to upload media attachment.',
        });
      }
    },
  );

  app.get('/api/media/:threadId/:filename', requireSession, (req, res) => {
    const resolved = store.resolveUploadedMediaPath(req.params.threadId, req.params.filename);
    if (!resolved) {
      res.status(404).json({ error: 'Media attachment not found.' });
      return;
    }
    res.setHeader('Content-Type', resolved.mimeType);
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Cache-Control', 'private, max-age=3600');
    res.sendFile(resolved.fullPath);
  });

  const normalizeIncomingAttachments = (
    rawList: unknown,
  ): {
    publicAttachments: ChatAttachmentRecord[];
    agentAttachments: Array<{
      id: string;
      name: string;
      kind: MediaModality;
      mime_type: string;
      size: number;
      local_path: string;
    }>;
  } => {
    if (!Array.isArray(rawList) || rawList.length === 0) {
      return { publicAttachments: [], agentAttachments: [] };
    }
    const publicAttachments: ChatAttachmentRecord[] = [];
    const agentAttachments: Array<{
      id: string;
      name: string;
      kind: MediaModality;
      mime_type: string;
      size: number;
      local_path: string;
    }> = [];

    for (const item of rawList.slice(0, 8)) {
      if (!item || typeof item !== 'object') continue;
      const rec = item as Partial<ChatAttachmentRecord>;
      const kind = rec.kind === 'image' || rec.kind === 'video' || rec.kind === 'audio' ? rec.kind : null;
      const storageKey = typeof rec.storageKey === 'string' ? rec.storageKey.trim() : '';
      if (!kind || !storageKey) continue;
      const localPath = store.resolveAttachmentLocalPath(storageKey);
      if (!localPath) continue;

      const att: ChatAttachmentRecord = {
        id: typeof rec.id === 'string' && rec.id.trim() ? rec.id.trim() : `att-${crypto.randomBytes(5).toString('hex')}`,
        name: typeof rec.name === 'string' && rec.name.trim() ? rec.name.trim().slice(0, 120) : 'attachment',
        kind,
        mimeType: typeof rec.mimeType === 'string' && rec.mimeType.trim() ? rec.mimeType.trim() : 'application/octet-stream',
        size: typeof rec.size === 'number' && Number.isFinite(rec.size) ? Math.max(0, Math.round(rec.size)) : 0,
        url: typeof rec.url === 'string' && rec.url.startsWith('/api/media/') ? rec.url : `/api/media/${storageKey}`,
        storageKey,
      };
      publicAttachments.push(att);
      agentAttachments.push({
        id: att.id,
        name: att.name,
        kind: att.kind,
        mime_type: att.mimeType,
        size: att.size,
        local_path: localPath,
      });
    }
    return { publicAttachments, agentAttachments };
  };

  app.post('/api/chat/send', requireSession, async (req, res) => {
    const { threadId, message, reasoningEffort, accessMode, attachments } = (req.body || {}) as {
      threadId?: string;
      message?: string;
      reasoningEffort?: string;
      accessMode?: ExecutionAccessMode;
      attachments?: ChatAttachmentRecord[];
    };
    const { publicAttachments, agentAttachments } = normalizeIncomingAttachments(attachments);
    const rawText = (message || '').trim();
    const text =
      rawText ||
      (publicAttachments.length > 0
        ? `Analyze the attached ${Array.from(new Set(publicAttachments.map((a) => a.kind))).join(', ')} file(s).`
        : '');
    if (!text) {
      res.status(400).json({ error: 'Message cannot be empty.' });
      return;
    }

    const username = getSessionUsername(res);
    const profileSecrets = loadProfileSecrets(username);
    const llmCfg = await enrichLLMConfigLimits(store.getLLMConfig(), { env: profileSecrets });

    const now = new Date().toISOString();
    const derivedTitle = cleanThreadTitle(text) || 'New Agent Session';
    const userMsg: ChatMessageRecord = {
      id: `msg-${crypto.randomBytes(6).toString('hex')}`,
      role: 'user' as const,
      content: text,
      timestamp: now,
      ...(publicAttachments.length > 0 ? { attachments: publicAttachments } : {}),
    };

    const existingThread = threadId ? store.listThreads().find((t) => t.id === threadId) || null : null;
    const preResolved = await resolveThreadExecutionPayload({
      thread: existingThread,
      globalCfg: llmCfg,
      profileSecrets,
      reasoningEffortOverride: reasoningEffort,
      accessModeOverride: accessMode,
    });
    if (!preResolved.sessionConfig.model.trim()) {
      res.status(400).json({
        error: 'No model is selected for this session. Pick a provider and model in the chat header or Settings.',
      });
      return;
    }

    const resolvedThreadId = store.updateThreads((threads) => {
      let target = threads.find((t) => t.id === threadId);
      if (!target) {
        target = {
          id: threadId || `thread-${crypto.randomBytes(5).toString('hex')}`,
          title: derivedTitle,
          createdAt: now,
          updatedAt: now,
          messages: [],
          pendingApproval: null,
          sessionConfig: preResolved.sessionConfig,
        };
        threads.unshift(target);
      } else {
        if (target.messages.length === 0) {
          target.title = derivedTitle;
        }
        target.sessionConfig = preResolved.sessionConfig;
      }
      target.messages.push(userMsg);
      target.updatedAt = now;
      return target.id;
    });

    const enabledMcps = store.listMCPServers().filter((s) => s.enabled);
    await syncProfileSecretsToCore(username, true);

    const agentResp = await callAgentCore('/agent/run', {
      method: 'POST',
      body: {
        task: text,
        thread_id: resolvedThreadId,
        max_turns: llmCfg.max_turns,
        access_mode: preResolved.effectiveAccessMode,
        primary_model: preResolved.primaryModelPayload,
        worker_model: preResolved.workerModelPayload,
        accessory_models: llmCfg.accessories,
        attachments: agentAttachments.length > 0 ? agentAttachments : undefined,
        mcp_servers: enabledMcps.length > 0 ? enabledMcps : undefined,
      },
    });

    const replyNow = new Date().toISOString();
    if (!agentResp.ok) {
      const errDetail = String(
        agentResp.data.detail || agentResp.data.error || `Agent-Core returned HTTP ${agentResp.status}`,
      );
      const assistantMsg = {
        id: `msg-${crypto.randomBytes(6).toString('hex')}`,
        role: 'assistant' as const,
        content: `Agent-Core could not complete this run: ${errDetail}`,
        timestamp: replyNow,
        status: 'error',
      };
      const updatedThread = store.updateThreads((freshThreads) => {
        const target = freshThreads.find((t) => t.id === resolvedThreadId);
        // Do NOT resurrect a thread deleted while awaiting Agent-Core
        if (!target) return null;
        target.messages.push(assistantMsg);
        target.updatedAt = replyNow;
        return target;
      });
      res.status(200).json({ thread: updatedThread, message: assistantMsg, error: errDetail });
      return;
    }

    const data = agentResp.data;
    const statusVal = String(data.status || 'completed');
    const pendingApproval = (data.pending_approval as Record<string, unknown> | null) || null;
    const finalText =
      String(data.final_response || '').trim() ||
      (statusVal === 'awaiting_approval'
        ? 'Execution paused for your approval before running a sensitive tool action.'
        : 'Task completed.');

    const assistantMsg: ChatMessageRecord = {
      id: `msg-${crypto.randomBytes(6).toString('hex')}`,
      role: 'assistant' as const,
      content: finalText,
      timestamp: replyNow,
      plan: Array.isArray(data.plan) ? (data.plan as ChatThreadRecord['messages'][number]['plan']) : [],
      activeSkills: Array.isArray(data.active_skills) ? (data.active_skills as string[]) : [],
      subagentReports: Array.isArray(data.subagent_reports)
        ? (data.subagent_reports as Array<Record<string, unknown>>)
        : [],
      activeModels: data.active_models as ChatThreadRecord['messages'][number]['activeModels'],
      thinking: typeof data.thinking === 'string' && data.thinking.trim() ? data.thinking.trim() : null,
      toolEvents: Array.isArray(data.tool_events) ? (data.tool_events as ToolEventRecord[]) : [],
      fileChanges: Array.isArray(data.file_changes) ? (data.file_changes as FileChangeRecord[]) : [],
      pendingApproval,
      status: statusVal,
    };

    const updatedThread = store.updateThreads((freshThreads) => {
      const target = freshThreads.find((t) => t.id === resolvedThreadId);
      // Do NOT resurrect a thread deleted while awaiting Agent-Core
      if (!target) return null;
      target.messages.push(assistantMsg);
      target.pendingApproval = pendingApproval;
      if (data.context_usage && typeof data.context_usage === 'object') {
        target.contextUsage = data.context_usage as ContextUsageSnapshot;
      }
      target.updatedAt = replyNow;
      return target;
    });
    res.json({ thread: updatedThread, message: assistantMsg });
  });

  // Live async chat jobs backed by server-side ChatJobOrchestrator
  // Jobs continue running and persist results even if the browser navigates away, switches threads, or reloads.
  app.get('/api/chat/jobs/active', requireSession, (_req, res) => {
    res.json({ jobs: jobOrchestrator.getActiveJobs() });
  });

  app.post('/api/chat/jobs/start', requireSession, async (req, res) => {
    jobOrchestrator.pruneOldJobs();
    const { threadId, message, reasoningEffort, accessMode, attachments } = (req.body || {}) as {
      threadId?: string;
      message?: string;
      reasoningEffort?: string;
      accessMode?: ExecutionAccessMode;
      attachments?: ChatAttachmentRecord[];
    };
    const { publicAttachments, agentAttachments } = normalizeIncomingAttachments(attachments);
    const rawText = (message || '').trim();
    const text =
      rawText ||
      (publicAttachments.length > 0
        ? `Analyze the attached ${Array.from(new Set(publicAttachments.map((a) => a.kind))).join(', ')} file(s).`
        : '');
    if (!text) {
      res.status(400).json({ error: 'Message cannot be empty.' });
      return;
    }

    let lockedThreadId = threadId ? String(threadId) : '';
    if (lockedThreadId) {
      if (!jobOrchestrator.tryAcquireThreadLaunch(lockedThreadId)) {
        const existingActive = jobOrchestrator.getActiveJobForThread(lockedThreadId);
        res.status(409).json({
          error: 'An agent task is already running in this session.',
          ...(existingActive ? { activeJob: existingActive } : {}),
        });
        return;
      }
    }

    try {
      const username = getSessionUsername(res);
      const profileSecrets = loadProfileSecrets(username);
      const llmCfg = await enrichLLMConfigLimits(store.getLLMConfig(), { env: profileSecrets });

      if (lockedThreadId && jobOrchestrator.isThreadDeleted(lockedThreadId)) {
        res.status(404).json({ error: 'Thread was deleted before execution started.' });
        return;
      }

      const existingThread = threadId ? store.listThreads().find((t) => t.id === threadId) || null : null;
      const resolvedExec = await resolveThreadExecutionPayload({
        thread: existingThread,
        globalCfg: llmCfg,
        profileSecrets,
        reasoningEffortOverride: reasoningEffort,
        accessModeOverride: accessMode,
      });

      if (!resolvedExec.sessionConfig.model.trim()) {
        res.status(400).json({
          error: 'No model is selected for this session. Pick a provider and model in the chat header or Settings.',
        });
        return;
      }

      const now = new Date().toISOString();
      const derivedTitle = cleanThreadTitle(text) || 'New Agent Session';
      const userMsg: ChatMessageRecord = {
        id: `msg-${crypto.randomBytes(6).toString('hex')}`,
        role: 'user',
        content: text,
        timestamp: now,
        ...(publicAttachments.length > 0 ? { attachments: publicAttachments } : {}),
      };

      const thread = store.updateThreads((threads) => {
        let target = threads.find((t) => t.id === threadId);
        if (!target) {
          if (threadId && jobOrchestrator.isThreadDeleted(threadId)) {
            return null;
          }
          target = {
            id: threadId || `thread-${crypto.randomBytes(5).toString('hex')}`,
            title: derivedTitle,
            createdAt: now,
            updatedAt: now,
            messages: [],
            pendingApproval: null,
            sessionConfig: resolvedExec.sessionConfig,
          };
          threads.unshift(target);
        } else {
          if (target.messages.length === 0) {
            target.title = derivedTitle;
          }
          target.sessionConfig = resolvedExec.sessionConfig;
        }
        target.messages.push(userMsg);
        target.updatedAt = now;
        return target;
      });

      if (!thread) {
        res.status(404).json({ error: 'Thread was deleted before execution started.' });
        return;
      }

      if (!lockedThreadId) {
        lockedThreadId = thread.id;
        jobOrchestrator.tryAcquireThreadLaunch(lockedThreadId);
      }

      const enabledMcps = store.listMCPServers().filter((s) => s.enabled);
      await syncProfileSecretsToCore(username, true);

      if (
        jobOrchestrator.isThreadDeleted(thread.id) ||
        !store.listThreads().some((t) => t.id === thread.id)
      ) {
        res.status(404).json({ error: 'Thread was deleted before execution started.' });
        return;
      }

      const agentResp = await callAgentCore('/agent/jobs/run', {
        method: 'POST',
        timeoutMs: 15_000,
        body: {
          task: text,
          thread_id: thread.id,
          max_turns: llmCfg.max_turns,
          access_mode: resolvedExec.effectiveAccessMode,
          primary_model: resolvedExec.primaryModelPayload,
          worker_model: resolvedExec.workerModelPayload,
          accessory_models: llmCfg.accessories,
          attachments: agentAttachments.length > 0 ? agentAttachments : undefined,
          mcp_servers: enabledMcps.length > 0 ? enabledMcps : undefined,
        },
      });

      if (!agentResp.ok || !agentResp.data.job_id) {
        const replyNow = new Date().toISOString();
        const errDetail = String(
          agentResp.data.detail || agentResp.data.error || `Agent-Core returned HTTP ${agentResp.status}`,
        );
        const assistantMsg: ChatMessageRecord = {
          id: `msg-${crypto.randomBytes(6).toString('hex')}`,
          role: 'assistant',
          content: `Agent-Core could not complete this run: ${errDetail}`,
          timestamp: replyNow,
          status: 'error',
        };
        const updatedThread = store.updateThreads((freshThreads) => {
          const target = freshThreads.find((t) => t.id === thread.id);
          // Do NOT resurrect a thread that was deleted while starting the job
          if (!target) return null;
          target.messages.push(assistantMsg);
          target.updatedAt = replyNow;
          return target;
        });
        res.status(200).json({
          jobId: null,
          status: 'error',
          thread: updatedThread,
          message: assistantMsg,
          error: errDetail,
        });
        return;
      }

      const jobId = String(agentResp.data.job_id);
      if (
        jobOrchestrator.isThreadDeleted(thread.id) ||
        !store.listThreads().some((t) => t.id === thread.id)
      ) {
        // Thread was deleted while Agent-Core was starting the job; cancel immediately to avoid an orphan
        void callAgentCore(`/agent/jobs/${encodeURIComponent(jobId)}/cancel`, {
          method: 'POST',
          timeoutMs: 5_000,
        }).catch(() => undefined);
        res.status(404).json({ error: 'Thread was deleted while starting execution.' });
        return;
      }

      const activity = jobOrchestrator.registerJob({
        jobId,
        threadId: thread.id,
        initialPayload: agentResp.data,
      });
      res.status(202).json({
        jobId,
        thread,
        activity,
      });
    } finally {
      if (lockedThreadId) {
        jobOrchestrator.releaseThreadLaunch(lockedThreadId);
      }
    }
  });

  const reconcileThreadPendingApproval = async (threadId: string) => {
    const inspectResp = await callAgentCore(`/agent/threads/${encodeURIComponent(threadId)}`, {
      timeoutMs: 10_000,
    });
    const inspectedApproval =
      inspectResp.ok && inspectResp.data && typeof inspectResp.data === 'object'
        ? ((inspectResp.data.pending_approval as Record<string, unknown> | null) ?? null)
        : null;
    const now = new Date().toISOString();
    const updatedThread = store.updateThreads((freshThreads) => {
      const target = freshThreads.find((t) => t.id === threadId);
      if (!target) return null;
      target.pendingApproval = inspectedApproval;
      if (
        inspectResp.ok &&
        inspectResp.data.context_usage &&
        typeof inspectResp.data.context_usage === 'object'
      ) {
        target.contextUsage = inspectResp.data.context_usage as ContextUsageSnapshot;
      }
      target.updatedAt = now;
      return target;
    });
    return updatedThread;
  };

  app.post('/api/chat/jobs/resume', requireSession, async (req, res) => {
    jobOrchestrator.pruneOldJobs();
    const { threadId, decision, arguments: editedArgs, reason, accessMode } = (req.body || {}) as {
      threadId?: string;
      decision?: 'approve' | 'reject' | 'edit';
      arguments?: Record<string, unknown>;
      reason?: string;
      accessMode?: ExecutionAccessMode;
    };
    if (!threadId || !decision) {
      res.status(400).json({ error: 'threadId and decision are required.' });
      return;
    }

    if (!jobOrchestrator.tryAcquireThreadLaunch(threadId)) {
      const existingActive = jobOrchestrator.getActiveJobForThread(threadId);
      res.status(409).json({
        error: 'An agent task is already running in this session.',
        ...(existingActive ? { activeJob: existingActive } : {}),
      });
      return;
    }

    try {
      const username = getSessionUsername(res);
      const profileSecrets = loadProfileSecrets(username);
      const llmCfg = await enrichLLMConfigLimits(store.getLLMConfig(), { env: profileSecrets });
      const targetThread = store.listThreads().find((t) => t.id === threadId) || null;
      if (jobOrchestrator.isThreadDeleted(threadId) || !targetThread) {
        res.status(404).json({ error: 'Thread was deleted before resume started.' });
        return;
      }
      const resolvedExec = await resolveThreadExecutionPayload({
        thread: targetThread,
        globalCfg: llmCfg,
        profileSecrets,
        accessModeOverride: accessMode,
      });
      const enabledMcps = store.listMCPServers().filter((s) => s.enabled);
      await syncProfileSecretsToCore(username, true);

      const agentResp = await callAgentCore('/agent/jobs/resume', {
        method: 'POST',
        timeoutMs: 15_000,
        body: {
          thread_id: threadId,
          decision,
          arguments: editedArgs,
          reason,
          access_mode: resolvedExec.effectiveAccessMode,
          primary_model: resolvedExec.primaryModelPayload,
          worker_model: resolvedExec.workerModelPayload,
          accessory_models: llmCfg.accessories,
          mcp_servers: enabledMcps.length > 0 ? enabledMcps : undefined,
        },
      });

      if (agentResp.status === 409) {
        const reconciled = await reconcileThreadPendingApproval(threadId);
        res.status(200).json({
          jobId: null,
          alreadyResolved: true,
          thread: reconciled,
        });
        return;
      }

      if (!agentResp.ok || !agentResp.data.job_id) {
        res.status(agentResp.status || 502).json({
          error: String(
            agentResp.data.detail ||
              agentResp.data.error ||
              `Failed to start approval execution (HTTP ${agentResp.status}).`,
          ),
        });
        return;
      }

      const jobId = String(agentResp.data.job_id);
      if (
        jobOrchestrator.isThreadDeleted(threadId) ||
        !store.listThreads().some((t) => t.id === threadId)
      ) {
        void callAgentCore(`/agent/jobs/${encodeURIComponent(jobId)}/cancel`, {
          method: 'POST',
          timeoutMs: 5_000,
        }).catch(() => undefined);
        res.status(404).json({ error: 'Thread was deleted while resuming execution.' });
        return;
      }

      const activity = jobOrchestrator.registerJob({
        jobId,
        threadId,
        initialPayload: agentResp.data,
      });
      const thread = store.listThreads().find((t) => t.id === threadId) || null;
      res.status(202).json({
        jobId,
        thread,
        activity,
      });
    } finally {
      jobOrchestrator.releaseThreadLaunch(threadId);
    }
  });

  app.get('/api/chat/jobs/:jobId', requireSession, async (req, res) => {
    const jobId = req.params.jobId;
    const waitMs = Math.min(2000, Math.max(0, Number(req.query.wait_ms || 300) || 300));
    const result = await jobOrchestrator.pollJobForClient(jobId, waitMs);

    if (!result.ok) {
      res.status(result.status).json({ error: result.error });
      return;
    }

    if (!result.done) {
      res.json({ done: false, activity: result.activity });
      return;
    }

    res.json({
      done: true,
      activity: result.activity,
      thread: result.thread ?? null,
      ...(result.message ? { message: result.message } : {}),
    });
  });

  app.post('/api/chat/jobs/:jobId/cancel', requireSession, async (req, res) => {
    const jobId = req.params.jobId;
    const result = await jobOrchestrator.cancelJob(jobId);
    res.json({
      ok: true,
      activity: result.activity,
      thread: result.thread,
      ...(result.message ? { message: result.message } : {}),
    });
  });

  app.post('/api/chat/resume', requireSession, async (req, res) => {
    const { threadId, decision, arguments: editedArgs, reason, accessMode } = (req.body || {}) as {
      threadId?: string;
      decision?: 'approve' | 'reject' | 'edit';
      arguments?: Record<string, unknown>;
      reason?: string;
      accessMode?: ExecutionAccessMode;
    };
    if (!threadId || !decision) {
      res.status(400).json({ error: 'threadId and decision are required.' });
      return;
    }

    const username = getSessionUsername(res);
    const profileSecrets = loadProfileSecrets(username);
    const llmCfg = await enrichLLMConfigLimits(store.getLLMConfig(), { env: profileSecrets });
    const targetThread = store.listThreads().find((t) => t.id === threadId) || null;
    const resolvedExec = await resolveThreadExecutionPayload({
      thread: targetThread,
      globalCfg: llmCfg,
      profileSecrets,
      accessModeOverride: accessMode,
    });
    const enabledMcps = store.listMCPServers().filter((s) => s.enabled);
    await syncProfileSecretsToCore(username, true);
    const agentResp = await callAgentCore('/agent/resume', {
      method: 'POST',
      timeoutMs: 300_000,
      body: {
        thread_id: threadId,
        decision,
        arguments: editedArgs,
        reason,
        access_mode: resolvedExec.effectiveAccessMode,
        primary_model: resolvedExec.primaryModelPayload,
        worker_model: resolvedExec.workerModelPayload,
        accessory_models: llmCfg.accessories,
        mcp_servers: enabledMcps.length > 0 ? enabledMcps : undefined,
      },
    });

    if (agentResp.status === 409) {
      const reconciled = await reconcileThreadPendingApproval(threadId);
      res.json({ thread: reconciled, alreadyResolved: true });
      return;
    }

    const now = new Date().toISOString();

    if (!agentResp.ok) {
      res.status(agentResp.status).json({
        error: String(agentResp.data.detail || agentResp.data.error || 'Failed to resume approval.'),
      });
      return;
    }

    const data = agentResp.data;
    const pendingApproval = (data.pending_approval as Record<string, unknown> | null) || null;
    const assistantMsg = {
      id: `msg-${crypto.randomBytes(6).toString('hex')}`,
      role: 'assistant' as const,
      content:
        String(data.final_response || '').trim() ||
        (pendingApproval
          ? 'Updated arguments validated. Please review and approve execution.'
          : 'Action processed.'),
      timestamp: now,
      plan: Array.isArray(data.plan) ? (data.plan as ChatThreadRecord['messages'][number]['plan']) : [],
      activeSkills: Array.isArray(data.active_skills) ? (data.active_skills as string[]) : [],
      subagentReports: Array.isArray(data.subagent_reports)
        ? (data.subagent_reports as Array<Record<string, unknown>>)
        : [],
      activeModels: data.active_models as ChatThreadRecord['messages'][number]['activeModels'],
      thinking: typeof data.thinking === 'string' && data.thinking.trim() ? data.thinking.trim() : null,
      toolEvents: Array.isArray(data.tool_events) ? (data.tool_events as ToolEventRecord[]) : [],
      fileChanges: Array.isArray(data.file_changes) ? (data.file_changes as FileChangeRecord[]) : [],
      pendingApproval,
      status: String(data.status || 'completed'),
    };

    const updatedThread = store.updateThreads((freshThreads) => {
      const target = freshThreads.find((t) => t.id === threadId);
      if (!target) return null;
      target.messages.push(assistantMsg);
      target.pendingApproval = pendingApproval;
      if (data.context_usage && typeof data.context_usage === 'object') {
        target.contextUsage = data.context_usage as ContextUsageSnapshot;
      }
      target.updatedAt = now;
      return target;
    });

    res.json({ thread: updatedThread, message: assistantMsg });
  });

  // ---------------------------------------------------------------------------
  // Workspace Filesystem API (/workspace)
  // ---------------------------------------------------------------------------
  app.get('/api/filesystem/tree', requireSession, (req, res) => {
    const dirParam = typeof req.query.dir === 'string' ? req.query.dir.trim() : '';
    try {
      if (dirParam) {
        const listed = store.listWorkspaceDirectory(dirParam);
        res.json({
          root: '/workspace',
          dir: dirParam,
          entries: listed.entries,
          totalCount: listed.totalCount,
          truncated: listed.truncated,
        });
        return;
      }
      res.json({
        root: '/workspace',
        entries: store.listWorkspaceTree(),
      });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to read workspace tree.' });
    }
  });

  app.get('/api/filesystem/file', requireSession, (req, res) => {
    const filePath = String(req.query.path || '');
    try {
      res.json(store.readWorkspaceFile(filePath));
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to read file.' });
    }
  });

  app.put('/api/filesystem/file', requireSession, (req, res) => {
    const { path: filePath, content, createOnly } = (req.body || {}) as {
      path?: string;
      content?: string;
      createOnly?: boolean;
    };
    try {
      const saved = store.writeWorkspaceFile(filePath || '', content ?? '', {
        createOnly: Boolean(createOnly),
      });
      res.json({ ok: true, file: saved });
    } catch (err) {
      const status = (err as NodeJS.ErrnoException)?.code === 'EEXIST' ? 409 : 400;
      res.status(status).json({ error: err instanceof Error ? err.message : 'Unable to save file.' });
    }
  });

  app.post('/api/filesystem/folder', requireSession, (req, res) => {
    const { path: folderPath } = (req.body || {}) as { path?: string };
    try {
      const created = store.createWorkspaceFolder(folderPath || '');
      res.status(201).json({ ok: true, folder: created });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to create folder.' });
    }
  });

  app.delete('/api/filesystem/entry', requireSession, (req, res) => {
    const targetPath = String(req.query.path || '');
    try {
      store.deleteWorkspaceEntry(targetPath);
      res.json({ ok: true });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to delete entry.' });
    }
  });

  // ---------------------------------------------------------------------------
  // Settings: LLM Config, Skills, and MCP Servers
  // ---------------------------------------------------------------------------
  app.get('/api/settings/llm', requireSession, async (_req, res) => {
    const username = getSessionUsername(res);
    const secrets = loadProfileSecrets(username);
    const rawConfig = await enrichLLMConfigLimits(store.getLLMConfig(), { env: secrets });
    const config = {
      ...rawConfig,
      custom_providers: (rawConfig.custom_providers || []).map((cp) => {
        const envVar = deriveCustomProviderEnvVar(cp.id);
        const rawKey = (
          secrets[envVar] ||
          (cp.id === 'custom_legacy' ? secrets.PRIMARY_API_KEY : '') ||
          ''
        ).trim();
        return {
          ...cp,
          api_key_configured: Boolean(rawKey),
          masked_key: maskEnvSecret(rawKey),
        };
      }),
    };
    res.json({
      config,
      providers: store.getProviderAvailability(secrets),
      vault: buildVaultSummary(username),
    });
  });

  app.get('/api/settings/access-mode', requireSession, (_req, res) => {
    const cfg = store.getLLMConfig();
    res.json({ access_mode: cfg.access_mode });
  });

  app.put('/api/settings/access-mode', requireSession, async (req, res) => {
    const { access_mode } = (req.body || {}) as { access_mode?: unknown };
    if (access_mode !== 'default' && access_mode !== 'full_access') {
      res.status(400).json({
        error: "Invalid access_mode. Expected 'default' or 'full_access'.",
      });
      return;
    }
    const saved = store.saveLLMConfig({ access_mode });
    const syncResp = await callAgentCore('/config/access-mode', {
      method: 'PUT',
      body: { access_mode: saved.access_mode },
      timeoutMs: 4000,
    });
    res.json({
      ok: true,
      access_mode: saved.access_mode,
      coreSynced: syncResp.ok,
    });
  });

  app.get('/api/settings/sandbox', requireSession, (_req, res) => {
    const cfg = store.getLLMConfig();
    res.json({
      allow_unsafe_local_bash: cfg.allow_unsafe_local_bash,
    });
  });

  app.put('/api/settings/sandbox', requireSession, async (req, res) => {
    const { allow_unsafe_local_bash } = (req.body || {}) as { allow_unsafe_local_bash?: unknown };
    if (typeof allow_unsafe_local_bash !== 'boolean') {
      res.status(400).json({ error: 'allow_unsafe_local_bash must be a boolean.' });
      return;
    }
    const saved = store.saveLLMConfig({ allow_unsafe_local_bash });
    const syncResp = await callAgentCore('/config/sandbox', {
      method: 'PUT',
      body: { allow_unsafe_local_bash: saved.allow_unsafe_local_bash },
      timeoutMs: 4000,
    });
    res.json({
      ok: true,
      allow_unsafe_local_bash: saved.allow_unsafe_local_bash,
      coreSynced: syncResp.ok,
    });
  });

  const enrichCustomProvidersWithVaultStatus = (
    providers: CustomProviderConfig[],
    profileSecrets: Record<string, string>,
  ) =>
    providers.map((cp) => {
      const envVar = deriveCustomProviderEnvVar(cp.id);
      const rawKey = (
        profileSecrets[envVar] ||
        (cp.id === 'custom_legacy' ? profileSecrets.PRIMARY_API_KEY : '') ||
        ''
      ).trim();
      return {
        ...cp,
        api_key_configured: Boolean(rawKey),
        masked_key: maskEnvSecret(rawKey),
      };
    });

  app.get('/api/settings/custom-providers', requireSession, (_req, res) => {
    const username = getSessionUsername(res);
    const secrets = loadProfileSecrets(username);
    const list = enrichCustomProvidersWithVaultStatus(store.getCustomProviders(), secrets);
    res.json({ custom_providers: list, providers: store.getProviderAvailability(secrets) });
  });

  app.post('/api/settings/custom-providers', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const body = (req.body || {}) as {
      id?: string;
      name?: string;
      base_url?: string;
      protocol?: CustomProviderProtocol;
      enabled?: boolean;
      models?: Array<string | CustomProviderModelConfig>;
      timeout_seconds?: number | null;
      api_key?: string;
    };
    const cleanBase = (body.base_url || '').trim().replace(/\/+$/, '');
    const urlCheck = await validateOutboundProviderUrl(cleanBase, {
      allowLoopback: true,
      resolveDns: false,
    });
    if (!urlCheck.ok) {
      res.status(400).json({ error: urlCheck.error });
      return;
    }
    try {
      const created = store.upsertCustomProvider({
        id: body.id,
        name: body.name || '',
        base_url: cleanBase,
        protocol: body.protocol,
        enabled: body.enabled,
        models: body.models,
        timeout_seconds: body.timeout_seconds,
      });
      if (typeof body.api_key === 'string' && body.api_key.trim()) {
        const envVar = deriveCustomProviderEnvVar(created.id);
        loadProfileSecrets(username);
        profileVault.setSecrets(username, { [envVar]: body.api_key.trim() }, { resetIfUnreadable: true });
      }
      const updatedSecrets = loadProfileSecrets(username);
      await syncProfileSecretsToCore(username, true);
      await callAgentCore('/config/llm', {
        method: 'PUT',
        body: store.getLLMConfig(),
        timeoutMs: 5000,
      });
      const list = enrichCustomProvidersWithVaultStatus(store.getCustomProviders(), updatedSecrets);
      res.status(201).json({
        ok: true,
        provider: list.find((p) => p.id === created.id) || created,
        custom_providers: list,
        providers: store.getProviderAvailability(updatedSecrets),
        vault: buildVaultSummary(username),
      });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to save custom provider.' });
    }
  });

  app.put('/api/settings/custom-providers/:id', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const targetId = normalizeProviderId(req.params.id);
    const existing = store.getCustomProvider(targetId);
    if (!existing) {
      res.status(404).json({ error: 'Custom provider not found.' });
      return;
    }
    const body = (req.body || {}) as {
      name?: string;
      base_url?: string;
      protocol?: CustomProviderProtocol;
      enabled?: boolean;
      models?: Array<string | CustomProviderModelConfig>;
      timeout_seconds?: number | null;
      api_key?: string;
      clear_api_key?: boolean;
    };
    const nextBase = (body.base_url ?? existing.base_url).trim().replace(/\/+$/, '');
    const urlCheck = await validateOutboundProviderUrl(nextBase, {
      allowLoopback: true,
      resolveDns: false,
    });
    if (!urlCheck.ok) {
      res.status(400).json({ error: urlCheck.error });
      return;
    }
    try {
      const updated = store.upsertCustomProvider({
        id: targetId,
        name: body.name ?? existing.name,
        base_url: nextBase,
        protocol: body.protocol ?? existing.protocol,
        enabled: body.enabled ?? existing.enabled,
        models: body.models ?? existing.models,
        timeout_seconds: 'timeout_seconds' in body ? body.timeout_seconds : existing.timeout_seconds,
      });
      const envVar = deriveCustomProviderEnvVar(updated.id);
      if (body.clear_api_key) {
        loadProfileSecrets(username);
        profileVault.deleteSecret(username, envVar, { resetIfUnreadable: true });
      } else if (typeof body.api_key === 'string' && body.api_key.trim()) {
        loadProfileSecrets(username);
        profileVault.setSecrets(username, { [envVar]: body.api_key.trim() }, { resetIfUnreadable: true });
      }
      const updatedSecrets = loadProfileSecrets(username);
      await syncProfileSecretsToCore(username, true);
      await callAgentCore('/config/llm', {
        method: 'PUT',
        body: store.getLLMConfig(),
        timeoutMs: 5000,
      });
      const list = enrichCustomProvidersWithVaultStatus(store.getCustomProviders(), updatedSecrets);
      res.json({
        ok: true,
        provider: list.find((p) => p.id === updated.id) || updated,
        custom_providers: list,
        providers: store.getProviderAvailability(updatedSecrets),
        vault: buildVaultSummary(username),
      });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to update custom provider.' });
    }
  });

  app.delete('/api/settings/custom-providers/:id', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const targetId = normalizeProviderId(req.params.id);
    const deleted = store.deleteCustomProvider(targetId);
    if (!deleted) {
      res.status(404).json({ error: 'Custom provider not found.' });
      return;
    }
    const envVar = deriveCustomProviderEnvVar(targetId);
    try {
      loadProfileSecrets(username);
      profileVault.deleteSecret(username, envVar, { resetIfUnreadable: true });
    } catch {
      // ignore if key did not exist
    }
    const updatedSecrets = loadProfileSecrets(username);
    await syncProfileSecretsToCore(username, true);
    await callAgentCore('/config/llm', {
      method: 'PUT',
      body: store.getLLMConfig(),
      timeoutMs: 5000,
    });
    const list = enrichCustomProvidersWithVaultStatus(store.getCustomProviders(), updatedSecrets);
    res.json({
      ok: true,
      custom_providers: list,
      providers: store.getProviderAvailability(updatedSecrets),
      vault: buildVaultSummary(username),
    });
  });

  app.post('/api/settings/resolve-model', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const { provider, model, base_url, api_key, role, manual_context_window, force_refresh } = (req.body || {}) as {
      provider?: string;
      model?: string;
      base_url?: string;
      api_key?: string;
      role?: 'primary' | 'worker' | 'image' | 'video' | 'audio';
      manual_context_window?: number | null;
      force_refresh?: boolean;
    };
    const cleanModel = (model || '').trim();
    if (!cleanModel) {
      res.status(400).json({ error: 'Model ID is required.' });
      return;
    }
    const cleanProvider = normalizeProviderId(provider || 'river_ai');
    const customProvider = store.getCustomProvider(cleanProvider);
    const profileSecrets = loadProfileSecrets(username);
    const resolved = await resolveModelMetadataUnified({
      provider: cleanProvider,
      model: cleanModel,
      base_url: base_url?.trim() || customProvider?.base_url || undefined,
      api_key: api_key?.trim() || undefined,
      customProvider,
      manualContextWindow: manual_context_window ?? null,
      forceRefresh: Boolean(force_refresh),
      role: role === 'primary' ? 'primary' : 'worker',
      env: profileSecrets,
      allowLiveFetch: true,
      allowReferenceFetch: true,
      timeoutMs: 3500,
    });
    res.json(resolved);
  });

  app.put('/api/settings/llm', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const body = (req.body || {}) as {
      primary?: Partial<ReturnType<typeof store.getLLMConfig>['primary']>;
      worker?: Partial<ReturnType<typeof store.getLLMConfig>['worker']>;
      accessories?: Partial<{
        image: Partial<AccessoryLLMConfig>;
        video: Partial<AccessoryLLMConfig>;
        audio: Partial<AccessoryLLMConfig>;
      }>;
      max_turns?: unknown;
      access_mode?: unknown;
      allow_unsafe_local_bash?: unknown;
      custom_providers?: CustomProviderConfig[];
      planning_timeout?: Partial<PlanningTimeoutConfig>;
      api_keys?: {
        primary?: string;
        worker?: string;
        image?: string;
        video?: string;
        audio?: string;
      };
    };

    let validatedMaxTurns: number | undefined;
    if (body.max_turns !== undefined) {
      const check = validateMaxTurns(body.max_turns);
      if (!check.ok) {
        res.status(400).json({ error: check.error });
        return;
      }
      validatedMaxTurns = check.value;
    }

    let validatedAccessMode: ExecutionAccessMode | undefined;
    if (body.access_mode !== undefined) {
      if (body.access_mode !== 'default' && body.access_mode !== 'full_access') {
        res.status(400).json({
          error: "Invalid access_mode. Expected 'default' or 'full_access'.",
        });
        return;
      }
      validatedAccessMode = normalizeAccessMode(body.access_mode);
    }

    const currentSecrets = loadProfileSecrets(username);
    const initialSaved = store.saveLLMConfig({
      primary: body.primary as ReturnType<typeof store.getLLMConfig>['primary'],
      worker: body.worker as ReturnType<typeof store.getLLMConfig>['worker'],
      accessories: body.accessories,
      ...(validatedMaxTurns !== undefined ? { max_turns: validatedMaxTurns } : {}),
      ...(validatedAccessMode !== undefined ? { access_mode: validatedAccessMode } : {}),
      ...(typeof body.allow_unsafe_local_bash === 'boolean'
        ? { allow_unsafe_local_bash: body.allow_unsafe_local_bash }
        : {}),
      ...(Array.isArray(body.custom_providers) ? { custom_providers: body.custom_providers } : {}),
      ...(body.planning_timeout ? { planning_timeout: body.planning_timeout } : {}),
    });

    const vaultUpdates: Record<string, string> = {};
    if (body.api_keys?.primary && body.api_keys.primary.trim()) {
      const envVar = store.getEnvVarForProvider(initialSaved.primary.provider, 'primary');
      vaultUpdates[envVar] = body.api_keys.primary.trim();
    }
    if (body.api_keys?.worker && body.api_keys.worker.trim()) {
      const envVar = store.getEnvVarForProvider(initialSaved.worker.provider, 'worker');
      vaultUpdates[envVar] = body.api_keys.worker.trim();
    }
    for (const mod of ['image', 'video', 'audio'] as const) {
      const keyVal = body.api_keys?.[mod];
      if (keyVal && keyVal.trim()) {
        const envVar = store.getEnvVarForProvider(initialSaved.accessories[mod].provider, 'worker');
        vaultUpdates[envVar] = keyVal.trim();
      }
    }

    const enriched = await enrichLLMConfigLimits(initialSaved, {
      env: { ...currentSecrets, ...vaultUpdates },
      allowLiveFetch: true,
    });
    const saved =
      JSON.stringify(enriched) !== JSON.stringify(initialSaved)
        ? store.saveLLMConfig(enriched)
        : initialSaved;

    // Store API keys encrypted at rest inside this user's profile vault (AES-256-GCM)
    let savedVaultKeys: string[] = [];
    try {
      loadProfileSecrets(username);
      savedVaultKeys = profileVault.setSecrets(username, vaultUpdates, { resetIfUnreadable: true });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Unable to save API keys to encrypted profile vault.',
      });
      return;
    }
    const updatedSecrets = loadProfileSecrets(username);
    await syncProfileSecretsToCore(username, true);

    // Sync LLM config inside running Agent-Core
    const syncResp = await callAgentCore('/config/llm', {
      method: 'PUT',
      body: saved,
      timeoutMs: 6000,
    });

    if (!syncResp.ok && syncResp.status >= 400 && syncResp.status < 500) {
      res.status(syncResp.status).json({
        error: String(syncResp.data.detail || syncResp.data.error || 'Agent-Core rejected LLM configuration.'),
        config: saved,
        providers: store.getProviderAvailability(updatedSecrets),
        savedEnvKeys: savedVaultKeys,
        vault: buildVaultSummary(username),
      });
      return;
    }

    const coreStatus = agentManager.getStatus();
    const syncWarning = syncResp.ok
      ? null
      : coreStatus.state === 'bootstrapping' || coreStatus.state === 'starting'
        ? 'Agent-Core is initializing and will apply these settings automatically once ready.'
        : String(syncResp.data.detail || 'Saved to your encrypted profile vault; Agent-Core will reload once online.');

    res.json({
      ok: true,
      config: saved,
      providers: store.getProviderAvailability(updatedSecrets),
      savedEnvKeys: savedVaultKeys,
      vault: buildVaultSummary(username),
      coreSynced: syncResp.ok,
      syncWarning,
    });
  });

  app.get('/api/settings/vault', requireSession, (_req, res) => {
    const username = getSessionUsername(res);
    const secrets = loadProfileSecrets(username);
    res.json({
      vault: buildVaultSummary(username),
      providers: store.getProviderAvailability(secrets),
    });
  });

  app.put('/api/settings/vault', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const { provider, env_var, api_key } = (req.body || {}) as {
      provider?: string;
      env_var?: string;
      api_key?: string;
    };
    const cleanKey = (api_key || '').trim();
    if (!cleanKey) {
      res.status(400).json({ error: 'API key value is required.' });
      return;
    }
    const targetEnvVar = (
      env_var?.trim().toUpperCase() ||
      (provider ? store.getEnvVarForProvider(provider, 'primary') : '')
    ).trim();
    if (!isAllowedProfileSecretKey(targetEnvVar)) {
      res.status(400).json({ error: 'Unsupported provider API key name.' });
      return;
    }

    try {
      loadProfileSecrets(username);
      profileVault.setSecrets(username, { [targetEnvVar]: cleanKey }, { resetIfUnreadable: true });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Unable to save API key to encrypted profile vault.',
      });
      return;
    }
    const updatedSecrets = loadProfileSecrets(username);
    await syncProfileSecretsToCore(username, true);

    res.json({
      ok: true,
      savedKey: targetEnvVar,
      vault: buildVaultSummary(username),
      providers: store.getProviderAvailability(updatedSecrets),
    });
  });

  app.delete('/api/settings/vault/:envVar', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const targetEnvVar = String(req.params.envVar || '').trim().toUpperCase();
    if (!isAllowedProfileSecretKey(targetEnvVar)) {
      res.status(400).json({ error: 'Unsupported provider API key name.' });
      return;
    }

    let deleted = false;
    try {
      loadProfileSecrets(username);
      deleted = profileVault.deleteSecret(username, targetEnvVar, { resetIfUnreadable: true });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Unable to remove API key from profile vault.',
      });
      return;
    }
    const updatedSecrets = loadProfileSecrets(username);
    await syncProfileSecretsToCore(username, true);

    res.json({
      ok: true,
      deleted,
      vault: buildVaultSummary(username),
      providers: store.getProviderAvailability(updatedSecrets),
    });
  });

  app.post('/api/settings/models', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const { provider, base_url, api_key, role, force_refresh } = (req.body || {}) as {
      provider?: string;
      base_url?: string;
      api_key?: string;
      role?: 'primary' | 'worker' | 'image' | 'video' | 'audio';
      force_refresh?: boolean;
    };
    const cleanProvider = normalizeProviderId(provider || '');
    if (!cleanProvider) {
      res.status(400).json({ ok: false, models: [], count: 0, error: 'Provider is required.' });
      return;
    }

    const customProvider = store.getCustomProvider(cleanProvider);
    const profileSecrets = loadProfileSecrets(username);
    const catalog = await listProviderModelsDirect({
      provider: cleanProvider,
      base_url: base_url?.trim() || customProvider?.base_url || undefined,
      api_key: api_key?.trim() || undefined,
      protocol: customProvider?.protocol,
      customProvider,
      forceRefresh: Boolean(force_refresh),
      role: role === 'primary' ? 'primary' : 'worker',
      env: profileSecrets,
    });

    res.json(catalog);
  });

  app.post('/api/settings/probe', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const { role, override, apiKey } = (req.body || {}) as {
      role?: 'primary' | 'worker' | 'image' | 'video' | 'audio';
      override?: Partial<ReturnType<typeof store.getLLMConfig>['primary']> & { api_key?: string };
      apiKey?: string;
    };
    const profileSecrets = loadProfileSecrets(username);
    const persisted = store.getLLMConfig();
    const isAccessory = role === 'image' || role === 'video' || role === 'audio';
    const targetRole: 'primary' | 'worker' = role === 'primary' ? 'primary' : 'worker';
    const persistedRoleConfig = isAccessory
      ? persisted.accessories[role]
      : persisted[targetRole];
    const targetProvider = normalizeProviderId(override?.provider || persistedRoleConfig.provider);
    const customProvider = store.getCustomProvider(targetProvider);
    const result = await probeProviderModelDirect({
      role: targetRole,
      override,
      persistedRoleConfig,
      customProvider,
      apiKey,
      env: profileSecrets,
    });
    res.json(result);
  });

  app.get('/api/skills', requireSession, (_req, res) => {
    res.json({ skills: store.listSkills() });
  });

  app.put('/api/skills/:name', requireSession, (req, res) => {
    const { content } = (req.body || {}) as { content?: string };
    try {
      const skill = store.saveSkill(req.params.name, content || '');
      res.json({ ok: true, skill });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to save skill.' });
    }
  });

  app.delete('/api/skills/:name', requireSession, (req, res) => {
    store.deleteSkill(req.params.name);
    res.json({ ok: true });
  });

  app.get('/api/mcps', requireSession, (_req, res) => {
    res.json({ servers: store.listMCPServers() });
  });

  app.put('/api/mcps', requireSession, (req, res) => {
    const { servers } = (req.body || {}) as { servers?: Parameters<typeof store.saveMCPServers>[0] };
    try {
      const saved = store.saveMCPServers(Array.isArray(servers) ? servers : []);
      res.json({ ok: true, servers: saved });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to save MCP servers.' });
    }
  });

  // ---------------------------------------------------------------------------
  // Frontend SPA (Nemofy Agent via Vite dev middleware or built dist/)
  // ---------------------------------------------------------------------------
  const isProd = process.env.NODE_ENV === 'production';
  const distDir = path.resolve(cwd, 'dist');
  const distIndex = path.join(distDir, 'index.html');

  const server = http.createServer(app);
  server.requestTimeout = 300_000;

  let viteServer: { close: () => Promise<void> } | null = null;

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const disableHmr = process.env.DISABLE_HMR === 'true';
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        ...(disableHmr ? { ws: false, hmr: false, watch: null } : { hmr: { server }, watch: {} }),
      },
      appType: 'spa',
    });
    viteServer = vite;
    app.use(vite.middlewares);
  } else if (fs.existsSync(distIndex)) {
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(distIndex);
    });
  }

  server.on('error', (err) => {
    console.error(`[nemofy-agent] HTTP server error on ${HOST}:${PORT}:`, err);
    process.exit(1);
  });

  server.listen(PORT, HOST, () => {
    console.log(`[nemofy-agent] Server running on http://${HOST}:${PORT}`);
    void firecrawlManager.start();
    void agentManager.start();
  });

  let shuttingDown = false;
  const handleShutdown = async (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`[nemofy-agent] Received ${signal}; shutting down...`);

    const forceExitTimer = setTimeout(() => {
      process.exit(0);
    }, 2500);
    forceExitTimer.unref();

    // Immediately stop accepting new connections and close keep-alive sockets so restarts don't hit EADDRINUSE.
    const httpClosePromise = new Promise<void>((resolve) => {
      server.close(() => resolve());
      if (typeof server.closeAllConnections === 'function') {
        server.closeAllConnections();
      }
    });

    jobOrchestrator.shutdown();

    await Promise.allSettled([
      httpClosePromise,
      viteServer ? viteServer.close() : Promise.resolve(),
      firecrawlManager.stop(),
      agentManager.stop(),
    ]);

    clearTimeout(forceExitTimer);
    process.exit(0);
  };

  process.once('SIGINT', () => void handleShutdown('SIGINT'));
  process.once('SIGTERM', () => void handleShutdown('SIGTERM'));
}

startServer().catch((err) => {
  console.error('[nemofy-agent] Failed to start server:', err);
  process.exit(1);
});
