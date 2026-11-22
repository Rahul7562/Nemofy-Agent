import 'dotenv/config';
import crypto from 'node:crypto';
import express, { type NextFunction, type Request, type Response } from 'express';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { AuthStore, parseCookieHeader } from './server/authStore.ts';
import { AgentCoreProcessManager } from './server/agentProcessManager.ts';
import {
  WorkspaceAndConfigStore,
  type AccessoryLLMConfig,
  type ChatAttachmentRecord,
  type ChatMessageRecord,
  type ChatThreadRecord,
  type ContextUsageSnapshot,
  type FileChangeRecord,
  type MediaModality,
  type ToolEventRecord,
} from './server/configStore.ts';
import {
  listProviderModelsDirect,
  probeProviderModelDirect,
  resolveModelMetadataFromCatalog,
} from './server/providerCatalog.ts';
import {
  ALLOWED_PROFILE_SECRET_KEYS,
  ProfileSecretVaultStore,
} from './server/profileVault.ts';
import { resolveServerListenConfig } from './server/serverConfig.ts';

const cwd = process.cwd();
const { port: PORT, host: HOST } = resolveServerListenConfig();
const SESSION_COOKIE = 'nemofy_session';

const authStore = new AuthStore();
const store = new WorkspaceAndConfigStore(cwd);
const agentManager = new AgentCoreProcessManager(cwd, undefined, () => authStore.getAgentApiToken());
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

async function enrichRoleLimits<T extends { provider: string; model: string; context_window?: number | null; max_output_tokens?: number | null }>(
  roleCfg: T,
): Promise<T> {
  if (!roleCfg.model || !roleCfg.model.trim()) return roleCfg;
  if (roleCfg.context_window && roleCfg.max_output_tokens) return roleCfg;
  const meta = await resolveModelMetadataFromCatalog({
    provider: roleCfg.provider,
    model: roleCfg.model,
    explicitContextWindow: roleCfg.context_window,
    explicitMaxOutputTokens: roleCfg.max_output_tokens,
  });
  return {
    ...roleCfg,
    ...(meta.context_window ? { context_window: meta.context_window } : {}),
    ...(meta.max_output_tokens ? { max_output_tokens: meta.max_output_tokens } : {}),
  };
}

async function enrichLLMConfigLimits(
  cfg: ReturnType<typeof store.getLLMConfig>,
): Promise<ReturnType<typeof store.getLLMConfig>> {
  const [primary, worker, image, video, audio] = await Promise.all([
    enrichRoleLimits(cfg.primary),
    enrichRoleLimits(cfg.worker),
    enrichRoleLimits(cfg.accessories.image),
    enrichRoleLimits(cfg.accessories.video),
    enrichRoleLimits(cfg.accessories.audio),
  ]);
  return {
    ...cfg,
    primary,
    worker,
    accessories: { image, video, audio },
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

  // ---------------------------------------------------------------------------
  // Public Auth Endpoints
  // ---------------------------------------------------------------------------
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

  app.post('/api/threads', requireSession, (req, res) => {
    const { title } = (req.body || {}) as { title?: string };
    const now = new Date().toISOString();
    const thread: ChatThreadRecord = {
      id: `thread-${crypto.randomBytes(5).toString('hex')}`,
      title: (title || 'New Agent Session').trim().slice(0, 80),
      createdAt: now,
      updatedAt: now,
      messages: [],
      pendingApproval: null,
    };
    const threads = [thread, ...store.listThreads()];
    store.saveThreads(threads);
    res.status(201).json({ thread });
  });

  app.delete('/api/threads/:id', requireSession, (req, res) => {
    const id = req.params.id;
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
    const { threadId, message, reasoningEffort, attachments } = (req.body || {}) as {
      threadId?: string;
      message?: string;
      reasoningEffort?: string;
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

    let llmCfg = await enrichLLMConfigLimits(store.getLLMConfig());
    if (!llmCfg.primary.model.trim()) {
      res.status(400).json({
        error: 'No Primary model is selected yet. Open Settings → LLM Config and choose your Primary model first.',
      });
      return;
    }

    const now = new Date().toISOString();
    const derivedTitle = cleanThreadTitle(text) || 'New Agent Session';
    const userMsg: ChatMessageRecord = {
      id: `msg-${crypto.randomBytes(6).toString('hex')}`,
      role: 'user' as const,
      content: text,
      timestamp: now,
      ...(publicAttachments.length > 0 ? { attachments: publicAttachments } : {}),
    };

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
        };
        threads.unshift(target);
      } else if (target.messages.length === 0) {
        target.title = derivedTitle;
      }
      target.messages.push(userMsg);
      target.updatedAt = now;
      return target.id;
    });

    const requestedEffort =
      typeof reasoningEffort === 'string' && reasoningEffort.trim()
        ? reasoningEffort.trim().toLowerCase()
        : llmCfg.primary.reasoning_effort;

    if (requestedEffort && requestedEffort !== llmCfg.primary.reasoning_effort) {
      llmCfg = store.saveLLMConfig({
        primary: { ...llmCfg.primary, reasoning_effort: requestedEffort },
      });
    }

    const enabledMcps = store.listMCPServers().filter((s) => s.enabled);
    await syncProfileSecretsToCore(getSessionUsername(res), true);

    const agentResp = await callAgentCore('/agent/run', {
      method: 'POST',
      body: {
        task: text,
        thread_id: resolvedThreadId,
        max_turns: llmCfg.max_turns,
        primary_model: {
          provider: llmCfg.primary.provider,
          model: llmCfg.primary.model,
          base_url: llmCfg.primary.base_url,
          temperature: llmCfg.primary.temperature,
          max_tokens: llmCfg.primary.max_tokens,
          reasoning_effort: requestedEffort,
          ...(llmCfg.primary.context_window ? { context_window: llmCfg.primary.context_window } : {}),
          ...(llmCfg.primary.max_output_tokens ? { max_output_tokens: llmCfg.primary.max_output_tokens } : {}),
        },
        worker_model: {
          provider: llmCfg.worker.provider,
          model: llmCfg.worker.model,
          base_url: llmCfg.worker.base_url,
          temperature: llmCfg.worker.temperature,
          max_tokens: llmCfg.worker.max_tokens,
          reasoning_effort: llmCfg.worker.reasoning_effort,
          ...(llmCfg.worker.context_window ? { context_window: llmCfg.worker.context_window } : {}),
          ...(llmCfg.worker.max_output_tokens ? { max_output_tokens: llmCfg.worker.max_output_tokens } : {}),
        },
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
        let target = freshThreads.find((t) => t.id === resolvedThreadId);
        if (!target) {
          target = {
            id: resolvedThreadId,
            title: derivedTitle,
            createdAt: now,
            updatedAt: replyNow,
            messages: [userMsg],
            pendingApproval: null,
          };
          freshThreads.unshift(target);
        }
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
      let target = freshThreads.find((t) => t.id === resolvedThreadId);
      if (!target) {
        target = {
          id: resolvedThreadId,
          title: derivedTitle,
          createdAt: now,
          updatedAt: replyNow,
          messages: [userMsg],
          pendingApproval: null,
        };
        freshThreads.unshift(target);
      }
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

  // Live async chat jobs (enables real-time thinking, tool calls, and workspace file/dir changes)
  const activeChatJobs = new Map<string, { threadId: string; finalized: boolean; createdAt: number }>();

  const pruneChatJobs = () => {
    const cutoff = Date.now() - 30 * 60 * 1000;
    for (const [jid, meta] of activeChatJobs.entries()) {
      if (meta.finalized && meta.createdAt < cutoff) {
        activeChatJobs.delete(jid);
      }
    }
  };

  const extractActivityFromJobPayload = (jobId: string, data: Record<string, unknown>) => {
    const resultObj =
      data.result && typeof data.result === 'object'
        ? (data.result as Record<string, unknown>)
        : null;
    const rawThinking =
      typeof data.thinking === 'string' && data.thinking.trim()
        ? data.thinking.trim()
        : typeof resultObj?.thinking === 'string' && resultObj.thinking.trim()
          ? resultObj.thinking.trim()
          : null;
    const toolEvents = Array.isArray(data.tool_events)
      ? (data.tool_events as ToolEventRecord[])
      : Array.isArray(resultObj?.tool_events)
        ? (resultObj.tool_events as ToolEventRecord[])
        : [];
    const fileChanges = Array.isArray(data.file_changes)
      ? (data.file_changes as FileChangeRecord[])
      : Array.isArray(resultObj?.file_changes)
        ? (resultObj.file_changes as FileChangeRecord[])
        : [];
    const activityFeed = Array.isArray(data.activity_feed)
      ? (data.activity_feed as Array<{ id: string; phase: string; text: string; timestamp?: string }>)
      : [];
    const contextUsage =
      data.context_usage && typeof data.context_usage === 'object'
        ? (data.context_usage as ContextUsageSnapshot)
        : resultObj?.context_usage && typeof resultObj.context_usage === 'object'
          ? (resultObj.context_usage as ContextUsageSnapshot)
          : null;
    const activeTasks = Array.isArray(data.active_tasks)
      ? (data.active_tasks as Array<{
          id: string;
          kind: string;
          label: string;
          elapsed_seconds: number;
          expected_seconds?: number;
        }>)
      : [];
    return {
      jobId,
      status: String(data.status || 'running'),
      currentPhase: String(data.current_phase || 'thinking'),
      lastEvent: String(data.last_event || 'Nemo is working...'),
      elapsedSeconds: typeof data.elapsed_seconds === 'number' ? data.elapsed_seconds : 0,
      turnCount: typeof data.turn_count === 'number' ? data.turn_count : 0,
      thinking: rawThinking,
      activityFeed,
      toolEvents,
      fileChanges,
      contextUsage,
      activeTasks,
    };
  };

  app.post('/api/chat/jobs/start', requireSession, async (req, res) => {
    pruneChatJobs();
    const { threadId, message, reasoningEffort, attachments } = (req.body || {}) as {
      threadId?: string;
      message?: string;
      reasoningEffort?: string;
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

    let llmCfg = await enrichLLMConfigLimits(store.getLLMConfig());
    if (!llmCfg.primary.model.trim()) {
      res.status(400).json({
        error: 'No Primary model is selected yet. Open Settings → LLM Config and choose your Primary model first.',
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
        target = {
          id: threadId || `thread-${crypto.randomBytes(5).toString('hex')}`,
          title: derivedTitle,
          createdAt: now,
          updatedAt: now,
          messages: [],
          pendingApproval: null,
        };
        threads.unshift(target);
      } else if (target.messages.length === 0) {
        target.title = derivedTitle;
      }
      target.messages.push(userMsg);
      target.updatedAt = now;
      return target;
    });

    const requestedEffort =
      typeof reasoningEffort === 'string' && reasoningEffort.trim()
        ? reasoningEffort.trim().toLowerCase()
        : llmCfg.primary.reasoning_effort;

    if (requestedEffort && requestedEffort !== llmCfg.primary.reasoning_effort) {
      llmCfg = store.saveLLMConfig({
        primary: { ...llmCfg.primary, reasoning_effort: requestedEffort },
      });
    }

    const enabledMcps = store.listMCPServers().filter((s) => s.enabled);
    await syncProfileSecretsToCore(getSessionUsername(res), true);

    const agentResp = await callAgentCore('/agent/jobs/run', {
      method: 'POST',
      timeoutMs: 15_000,
      body: {
        task: text,
        thread_id: thread.id,
        max_turns: llmCfg.max_turns,
        primary_model: {
          provider: llmCfg.primary.provider,
          model: llmCfg.primary.model,
          base_url: llmCfg.primary.base_url,
          temperature: llmCfg.primary.temperature,
          max_tokens: llmCfg.primary.max_tokens,
          reasoning_effort: requestedEffort,
          ...(llmCfg.primary.context_window ? { context_window: llmCfg.primary.context_window } : {}),
          ...(llmCfg.primary.max_output_tokens ? { max_output_tokens: llmCfg.primary.max_output_tokens } : {}),
        },
        worker_model: {
          provider: llmCfg.worker.provider,
          model: llmCfg.worker.model,
          base_url: llmCfg.worker.base_url,
          temperature: llmCfg.worker.temperature,
          max_tokens: llmCfg.worker.max_tokens,
          reasoning_effort: llmCfg.worker.reasoning_effort,
          ...(llmCfg.worker.context_window ? { context_window: llmCfg.worker.context_window } : {}),
          ...(llmCfg.worker.max_output_tokens ? { max_output_tokens: llmCfg.worker.max_output_tokens } : {}),
        },
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
        let target = freshThreads.find((t) => t.id === thread.id);
        if (!target) {
          target = {
            id: thread.id,
            title: derivedTitle,
            createdAt: now,
            updatedAt: replyNow,
            messages: [userMsg],
            pendingApproval: null,
          };
          freshThreads.unshift(target);
        }
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
    activeChatJobs.set(jobId, { threadId: thread.id, finalized: false, createdAt: Date.now() });
    res.status(202).json({
      jobId,
      thread,
      activity: extractActivityFromJobPayload(jobId, agentResp.data),
    });
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
    pruneChatJobs();
    const { threadId, decision, arguments: editedArgs, reason } = (req.body || {}) as {
      threadId?: string;
      decision?: 'approve' | 'reject' | 'edit';
      arguments?: Record<string, unknown>;
      reason?: string;
    };
    if (!threadId || !decision) {
      res.status(400).json({ error: 'threadId and decision are required.' });
      return;
    }

    const llmCfg = await enrichLLMConfigLimits(store.getLLMConfig());
    const enabledMcps = store.listMCPServers().filter((s) => s.enabled);
    await syncProfileSecretsToCore(getSessionUsername(res), true);

    const agentResp = await callAgentCore('/agent/jobs/resume', {
      method: 'POST',
      timeoutMs: 15_000,
      body: {
        thread_id: threadId,
        decision,
        arguments: editedArgs,
        reason,
        primary_model: {
          provider: llmCfg.primary.provider,
          model: llmCfg.primary.model,
          base_url: llmCfg.primary.base_url,
          temperature: llmCfg.primary.temperature,
          max_tokens: llmCfg.primary.max_tokens,
          reasoning_effort: llmCfg.primary.reasoning_effort,
          ...(llmCfg.primary.context_window ? { context_window: llmCfg.primary.context_window } : {}),
          ...(llmCfg.primary.max_output_tokens ? { max_output_tokens: llmCfg.primary.max_output_tokens } : {}),
        },
        worker_model: {
          provider: llmCfg.worker.provider,
          model: llmCfg.worker.model,
          base_url: llmCfg.worker.base_url,
          temperature: llmCfg.worker.temperature,
          max_tokens: llmCfg.worker.max_tokens,
          reasoning_effort: llmCfg.worker.reasoning_effort,
          ...(llmCfg.worker.context_window ? { context_window: llmCfg.worker.context_window } : {}),
          ...(llmCfg.worker.max_output_tokens ? { max_output_tokens: llmCfg.worker.max_output_tokens } : {}),
        },
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
    activeChatJobs.set(jobId, { threadId, finalized: false, createdAt: Date.now() });
    const thread = store.listThreads().find((t) => t.id === threadId) || null;
    res.status(202).json({
      jobId,
      thread,
      activity: extractActivityFromJobPayload(jobId, agentResp.data),
    });
  });

  app.get('/api/chat/jobs/:jobId', requireSession, async (req, res) => {
    const jobId = req.params.jobId;
    const waitMs = Math.min(2000, Math.max(0, Number(req.query.wait_ms || 300) || 300));
    const poll = await callAgentCore(`/agent/jobs/${encodeURIComponent(jobId)}?wait_ms=${waitMs}`, {
      timeoutMs: 15_000,
    });

    if (!poll.ok) {
      res.status(poll.status).json({
        error: String(poll.data.detail || 'Unable to poll live agent job.'),
      });
      return;
    }

    const data = poll.data;
    const activity = extractActivityFromJobPayload(jobId, data);
    const isTerminal = ['completed', 'awaiting_approval', 'error', 'stalled', 'cancelled'].includes(
      activity.status,
    );

    if (!isTerminal) {
      res.json({ done: false, activity });
      return;
    }

    const jobMeta = activeChatJobs.get(jobId);
    const targetThreadId = jobMeta?.threadId || String(data.thread_id || '');

    if (jobMeta && !jobMeta.finalized && targetThreadId) {
      jobMeta.finalized = true;
      const replyNow = new Date().toISOString();
      const resultObj =
        data.result && typeof data.result === 'object'
          ? (data.result as Record<string, unknown>)
          : {};
      const pendingApproval = (resultObj.pending_approval as Record<string, unknown> | null) || null;
      const errText =
        typeof data.error === 'string' && data.error.trim()
          ? data.error.trim()
          : typeof resultObj.error === 'string' && resultObj.error.trim()
            ? resultObj.error.trim()
            : null;
      const finalText =
        String(resultObj.final_response || '').trim() ||
        (activity.status === 'awaiting_approval'
          ? 'Execution paused for your approval before running a sensitive tool action.'
          : activity.status === 'cancelled'
            ? 'Stopped by user.'
            : errText
              ? `Agent-Core could not complete this run: ${errText}`
              : activity.status === 'stalled'
                ? 'Agent-Core stopped because the task became unresponsive with no active background work.'
                : 'Task completed.');

      const assistantMsg: ChatMessageRecord = {
        id: `msg-${crypto.randomBytes(6).toString('hex')}`,
        role: 'assistant',
        content: finalText,
        timestamp: replyNow,
        plan: Array.isArray(resultObj.plan)
          ? (resultObj.plan as ChatThreadRecord['messages'][number]['plan'])
          : [],
        activeSkills: Array.isArray(resultObj.active_skills)
          ? (resultObj.active_skills as string[])
          : [],
        subagentReports: Array.isArray(resultObj.subagent_reports)
          ? (resultObj.subagent_reports as Array<Record<string, unknown>>)
          : [],
        activeModels: resultObj.active_models as ChatThreadRecord['messages'][number]['activeModels'],
        thinking: activity.thinking,
        toolEvents: activity.toolEvents,
        fileChanges: activity.fileChanges,
        pendingApproval,
        status: activity.status,
      };

      const updatedThread = store.updateThreads((freshThreads) => {
        const target = freshThreads.find((t) => t.id === targetThreadId);
        if (!target) return null;
        target.messages.push(assistantMsg);
        target.pendingApproval = pendingApproval;
        if (activity.contextUsage) {
          target.contextUsage = activity.contextUsage;
        }
        target.updatedAt = replyNow;
        return target;
      });
      res.json({ done: true, activity, thread: updatedThread, message: assistantMsg });
      return;
    }

    const thread = store.listThreads().find((t) => t.id === targetThreadId) || null;
    res.json({ done: true, activity, thread });
  });

  app.post('/api/chat/jobs/:jobId/cancel', requireSession, async (req, res) => {
    const jobId = req.params.jobId;
    const cancelResp = await callAgentCore(`/agent/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
      timeoutMs: 10_000,
    });

    const data = cancelResp.data || {};
    const activity = extractActivityFromJobPayload(jobId, {
      ...data,
      status: data.status || 'cancelled',
      current_phase: data.current_phase || 'cancelled',
      last_event: data.last_event || 'Stopped by user.',
    });

    const jobMeta = activeChatJobs.get(jobId);
    const targetThreadId = jobMeta?.threadId || String(data.thread_id || '');

    if (jobMeta && !jobMeta.finalized && targetThreadId) {
      jobMeta.finalized = true;
      const replyNow = new Date().toISOString();
      const assistantMsg: ChatMessageRecord = {
        id: `msg-${crypto.randomBytes(6).toString('hex')}`,
        role: 'assistant',
        content: 'Stopped by user.',
        timestamp: replyNow,
        plan: [],
        activeSkills: [],
        subagentReports: [],
        thinking: activity.thinking,
        toolEvents: activity.toolEvents,
        fileChanges: activity.fileChanges,
        pendingApproval: null,
        status: 'cancelled',
      };
      const updatedThread = store.updateThreads((freshThreads) => {
        const target = freshThreads.find((t) => t.id === targetThreadId);
        if (!target) return null;
        target.messages.push(assistantMsg);
        target.pendingApproval = null;
        target.updatedAt = replyNow;
        return target;
      });
      res.json({ ok: true, activity, thread: updatedThread, message: assistantMsg });
      return;
    }

    const thread = store.listThreads().find((t) => t.id === targetThreadId) || null;
    res.json({ ok: true, activity, thread });
  });

  app.post('/api/chat/resume', requireSession, async (req, res) => {
    const { threadId, decision, arguments: editedArgs, reason } = (req.body || {}) as {
      threadId?: string;
      decision?: 'approve' | 'reject' | 'edit';
      arguments?: Record<string, unknown>;
      reason?: string;
    };
    if (!threadId || !decision) {
      res.status(400).json({ error: 'threadId and decision are required.' });
      return;
    }

    const llmCfg = await enrichLLMConfigLimits(store.getLLMConfig());
    const enabledMcps = store.listMCPServers().filter((s) => s.enabled);
    await syncProfileSecretsToCore(getSessionUsername(res), true);
    const agentResp = await callAgentCore('/agent/resume', {
      method: 'POST',
      timeoutMs: 300_000,
      body: {
        thread_id: threadId,
        decision,
        arguments: editedArgs,
        reason,
        primary_model: {
          provider: llmCfg.primary.provider,
          model: llmCfg.primary.model,
          base_url: llmCfg.primary.base_url,
          temperature: llmCfg.primary.temperature,
          max_tokens: llmCfg.primary.max_tokens,
          reasoning_effort: llmCfg.primary.reasoning_effort,
          ...(llmCfg.primary.context_window ? { context_window: llmCfg.primary.context_window } : {}),
          ...(llmCfg.primary.max_output_tokens ? { max_output_tokens: llmCfg.primary.max_output_tokens } : {}),
        },
        worker_model: {
          provider: llmCfg.worker.provider,
          model: llmCfg.worker.model,
          base_url: llmCfg.worker.base_url,
          temperature: llmCfg.worker.temperature,
          max_tokens: llmCfg.worker.max_tokens,
          reasoning_effort: llmCfg.worker.reasoning_effort,
          ...(llmCfg.worker.context_window ? { context_window: llmCfg.worker.context_window } : {}),
          ...(llmCfg.worker.max_output_tokens ? { max_output_tokens: llmCfg.worker.max_output_tokens } : {}),
        },
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
    const config = await enrichLLMConfigLimits(store.getLLMConfig());
    res.json({
      config,
      providers: store.getProviderAvailability(secrets),
      vault: buildVaultSummary(username),
    });
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
      max_turns?: number;
      api_keys?: {
        primary?: string;
        worker?: string;
        image?: string;
        video?: string;
        audio?: string;
      };
    };

    const initialSaved = store.saveLLMConfig({
      primary: body.primary as ReturnType<typeof store.getLLMConfig>['primary'],
      worker: body.worker as ReturnType<typeof store.getLLMConfig>['worker'],
      accessories: body.accessories,
      max_turns: body.max_turns,
    });
    const enriched = await enrichLLMConfigLimits(initialSaved);
    const saved =
      JSON.stringify(enriched) !== JSON.stringify(initialSaved)
        ? store.saveLLMConfig(enriched)
        : initialSaved;

    const vaultUpdates: Record<string, string> = {};
    if (body.api_keys?.primary && body.api_keys.primary.trim()) {
      const envVar = store.getEnvVarForProvider(saved.primary.provider, 'primary');
      vaultUpdates[envVar] = body.api_keys.primary.trim();
    }
    if (body.api_keys?.worker && body.api_keys.worker.trim()) {
      const envVar = store.getEnvVarForProvider(saved.worker.provider, 'worker');
      vaultUpdates[envVar] = body.api_keys.worker.trim();
    }
    for (const mod of ['image', 'video', 'audio'] as const) {
      const keyVal = body.api_keys?.[mod];
      if (keyVal && keyVal.trim()) {
        const envVar = store.getEnvVarForProvider(saved.accessories[mod].provider, 'worker');
        vaultUpdates[envVar] = keyVal.trim();
      }
    }

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
    if (!ALLOWED_PROFILE_SECRET_KEYS.has(targetEnvVar)) {
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
    if (!ALLOWED_PROFILE_SECRET_KEYS.has(targetEnvVar)) {
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
    const { provider, base_url, api_key, role } = (req.body || {}) as {
      provider?: string;
      base_url?: string;
      api_key?: string;
      role?: 'primary' | 'worker' | 'image' | 'video' | 'audio';
    };
    const cleanProvider = (provider || '').trim();
    if (!cleanProvider) {
      res.status(400).json({ ok: false, models: [], count: 0, error: 'Provider is required.' });
      return;
    }

    const profileSecrets = loadProfileSecrets(username);
    const catalog = await listProviderModelsDirect({
      provider: cleanProvider,
      base_url: base_url?.trim() || undefined,
      api_key: api_key?.trim() || undefined,
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
    const result = await probeProviderModelDirect({
      role: targetRole,
      override,
      persistedRoleConfig,
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

    await Promise.allSettled([
      httpClosePromise,
      viteServer ? viteServer.close() : Promise.resolve(),
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
