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
  type ChatMessageRecord,
  type ChatThreadRecord,
  type FileChangeRecord,
  type ToolEventRecord,
} from './server/configStore.ts';
import {
  listProviderModelsDirect,
  probeProviderModelDirect,
} from './server/providerCatalog.ts';
import {
  ALLOWED_PROFILE_SECRET_KEYS,
  ProfileSecretVaultStore,
} from './server/profileVault.ts';

const cwd = process.cwd();
const PORT = Number(process.env.PORT || '3000');
const SESSION_COOKIE = 'nemofy_session';

const authStore = new AuthStore();
const store = new WorkspaceAndConfigStore(cwd);
const agentManager = new AgentCoreProcessManager(cwd);
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
  return profileVault.getDecryptedSecrets(cleanUser);
}

function buildVaultSummary(username: string) {
  return {
    username,
    encrypted: true,
    algorithm: 'AES-256-GCM',
    entries: profileVault.listSecretMetadata(username),
  };
}

async function syncProfileSecretsToCore(username: string): Promise<boolean> {
  const secrets = loadProfileSecrets(username);
  const resp = await callAgentCore('/config/secrets', {
    method: 'POST',
    body: { secrets },
    timeoutMs: 5000,
  });
  return resp.ok;
}

function getAgentApiToken(): string {
  return (process.env.AGENT_API_TOKEN || '').trim();
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
      res.json({ authenticated: false });
      return;
    }
    const secrets = loadProfileSecrets(session.username);
    void syncProfileSecretsToCore(session.username);
    res.json({
      authenticated: true,
      username: session.username,
      updatedAt: authStore.getUpdatedAt(),
      vaultKeysLoaded: Object.keys(secrets).length,
    });
  });

  app.post('/api/auth/login', async (req, res) => {
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
    clearSessionCookie(req, res);
    void callAgentCore('/config/secrets', { method: 'DELETE', timeoutMs: 3000 });
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

  app.post('/api/chat/send', requireSession, async (req, res) => {
    const { threadId, message, reasoningEffort } = (req.body || {}) as {
      threadId?: string;
      message?: string;
      reasoningEffort?: string;
    };
    const text = (message || '').trim();
    if (!text) {
      res.status(400).json({ error: 'Message cannot be empty.' });
      return;
    }

    const threads = store.listThreads();
    const now = new Date().toISOString();
    const derivedTitle = cleanThreadTitle(text) || 'New Agent Session';
    let thread = threads.find((t) => t.id === threadId);
    if (!thread) {
      thread = {
        id: threadId || `thread-${crypto.randomBytes(5).toString('hex')}`,
        title: derivedTitle,
        createdAt: now,
        updatedAt: now,
        messages: [],
        pendingApproval: null,
      };
      threads.unshift(thread);
    } else if (thread.messages.length === 0) {
      thread.title = derivedTitle;
    }

    const userMsg = {
      id: `msg-${crypto.randomBytes(6).toString('hex')}`,
      role: 'user' as const,
      content: text,
      timestamp: now,
    };
    thread.messages.push(userMsg);
    thread.updatedAt = now;
    store.saveThreads(threads);

    let llmCfg = store.getLLMConfig();
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
    await syncProfileSecretsToCore(getSessionUsername(res));

    const agentResp = await callAgentCore('/agent/run', {
      method: 'POST',
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
        },
        worker_model: {
          provider: llmCfg.worker.provider,
          model: llmCfg.worker.model,
          base_url: llmCfg.worker.base_url,
          temperature: llmCfg.worker.temperature,
          max_tokens: llmCfg.worker.max_tokens,
          reasoning_effort: llmCfg.worker.reasoning_effort,
        },
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
      thread.messages.push(assistantMsg);
      thread.updatedAt = replyNow;
      store.saveThreads(threads);
      res.status(200).json({ thread, message: assistantMsg, error: errDetail });
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

    thread.messages.push(assistantMsg);
    thread.pendingApproval = pendingApproval;
    thread.updatedAt = replyNow;
    store.saveThreads(threads);
    res.json({ thread, message: assistantMsg });
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
    };
  };

  app.post('/api/chat/jobs/start', requireSession, async (req, res) => {
    pruneChatJobs();
    const { threadId, message, reasoningEffort } = (req.body || {}) as {
      threadId?: string;
      message?: string;
      reasoningEffort?: string;
    };
    const text = (message || '').trim();
    if (!text) {
      res.status(400).json({ error: 'Message cannot be empty.' });
      return;
    }

    const threads = store.listThreads();
    const now = new Date().toISOString();
    const derivedTitle = cleanThreadTitle(text) || 'New Agent Session';
    let thread = threads.find((t) => t.id === threadId);
    if (!thread) {
      thread = {
        id: threadId || `thread-${crypto.randomBytes(5).toString('hex')}`,
        title: derivedTitle,
        createdAt: now,
        updatedAt: now,
        messages: [],
        pendingApproval: null,
      };
      threads.unshift(thread);
    } else if (thread.messages.length === 0) {
      thread.title = derivedTitle;
    }

    const userMsg: ChatMessageRecord = {
      id: `msg-${crypto.randomBytes(6).toString('hex')}`,
      role: 'user',
      content: text,
      timestamp: now,
    };
    thread.messages.push(userMsg);
    thread.updatedAt = now;
    store.saveThreads(threads);

    let llmCfg = store.getLLMConfig();
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
    await syncProfileSecretsToCore(getSessionUsername(res));

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
        },
        worker_model: {
          provider: llmCfg.worker.provider,
          model: llmCfg.worker.model,
          base_url: llmCfg.worker.base_url,
          temperature: llmCfg.worker.temperature,
          max_tokens: llmCfg.worker.max_tokens,
          reasoning_effort: llmCfg.worker.reasoning_effort,
        },
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
      thread.messages.push(assistantMsg);
      thread.updatedAt = replyNow;
      store.saveThreads(threads);
      res.status(200).json({
        jobId: null,
        status: 'error',
        thread,
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
    const threads = store.listThreads();
    const thread = threads.find((t) => t.id === targetThreadId);

    if (jobMeta && !jobMeta.finalized && thread) {
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

      thread.messages.push(assistantMsg);
      thread.pendingApproval = pendingApproval;
      thread.updatedAt = replyNow;
      store.saveThreads(threads);
      res.json({ done: true, activity, thread, message: assistantMsg });
      return;
    }

    res.json({ done: true, activity, thread: thread || null });
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
    const threads = store.listThreads();
    const thread = threads.find((t) => t.id === targetThreadId);

    if (jobMeta && !jobMeta.finalized && thread) {
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
      thread.messages.push(assistantMsg);
      thread.pendingApproval = null;
      thread.updatedAt = replyNow;
      store.saveThreads(threads);
      res.json({ ok: true, activity, thread, message: assistantMsg });
      return;
    }

    res.json({ ok: true, activity, thread: thread || null });
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

    const llmCfg = store.getLLMConfig();
    await syncProfileSecretsToCore(getSessionUsername(res));
    const agentResp = await callAgentCore('/agent/resume', {
      method: 'POST',
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
        },
        worker_model: {
          provider: llmCfg.worker.provider,
          model: llmCfg.worker.model,
          base_url: llmCfg.worker.base_url,
          temperature: llmCfg.worker.temperature,
          max_tokens: llmCfg.worker.max_tokens,
          reasoning_effort: llmCfg.worker.reasoning_effort,
        },
      },
    });

    const threads = store.listThreads();
    const thread = threads.find((t) => t.id === threadId);
    const now = new Date().toISOString();

    if (!agentResp.ok) {
      res.status(agentResp.status).json({
        error: String(agentResp.data.detail || 'Failed to resume approval.'),
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

    if (thread) {
      thread.messages.push(assistantMsg);
      thread.pendingApproval = pendingApproval;
      thread.updatedAt = now;
      store.saveThreads(threads);
    }

    res.json({ thread, message: assistantMsg });
  });

  // ---------------------------------------------------------------------------
  // Workspace Filesystem API (/workspace)
  // ---------------------------------------------------------------------------
  app.get('/api/filesystem/tree', requireSession, (_req, res) => {
    res.json({
      root: '/workspace',
      entries: store.listWorkspaceTree(),
    });
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
    const { path: filePath, content } = (req.body || {}) as { path?: string; content?: string };
    try {
      const saved = store.writeWorkspaceFile(filePath || '', content ?? '');
      res.json({ ok: true, file: saved });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unable to save file.' });
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
  app.get('/api/settings/llm', requireSession, (_req, res) => {
    const username = getSessionUsername(res);
    const secrets = loadProfileSecrets(username);
    res.json({
      config: store.getLLMConfig(),
      providers: store.getProviderAvailability(secrets),
      vault: buildVaultSummary(username),
    });
  });

  app.put('/api/settings/llm', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const body = (req.body || {}) as {
      primary?: Partial<ReturnType<typeof store.getLLMConfig>['primary']>;
      worker?: Partial<ReturnType<typeof store.getLLMConfig>['worker']>;
      max_turns?: number;
      api_keys?: {
        primary?: string;
        worker?: string;
      };
    };

    const saved = store.saveLLMConfig({
      primary: body.primary as ReturnType<typeof store.getLLMConfig>['primary'],
      worker: body.worker as ReturnType<typeof store.getLLMConfig>['worker'],
      max_turns: body.max_turns,
    });

    const vaultUpdates: Record<string, string> = {};
    if (body.api_keys?.primary && body.api_keys.primary.trim()) {
      const envVar = store.getEnvVarForProvider(saved.primary.provider, 'primary');
      vaultUpdates[envVar] = body.api_keys.primary.trim();
    }
    if (body.api_keys?.worker && body.api_keys.worker.trim()) {
      const envVar = store.getEnvVarForProvider(saved.worker.provider, 'worker');
      vaultUpdates[envVar] = body.api_keys.worker.trim();
    }

    // Store API keys encrypted at rest inside this user's profile vault (AES-256-GCM)
    loadProfileSecrets(username);
    const savedVaultKeys = profileVault.setSecrets(username, vaultUpdates);
    const updatedSecrets = loadProfileSecrets(username);
    await syncProfileSecretsToCore(username);

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

    loadProfileSecrets(username);
    profileVault.setSecrets(username, { [targetEnvVar]: cleanKey });
    const updatedSecrets = loadProfileSecrets(username);
    await syncProfileSecretsToCore(username);

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

    loadProfileSecrets(username);
    const deleted = profileVault.deleteSecret(username, targetEnvVar);
    const updatedSecrets = loadProfileSecrets(username);
    await syncProfileSecretsToCore(username);

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
      role?: 'primary' | 'worker';
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
      role: role === 'worker' ? 'worker' : 'primary',
      env: profileSecrets,
    });

    res.json(catalog);
  });

  app.post('/api/settings/probe', requireSession, async (req, res) => {
    const username = getSessionUsername(res);
    const { role, override, apiKey } = (req.body || {}) as {
      role?: 'primary' | 'worker';
      override?: Partial<ReturnType<typeof store.getLLMConfig>['primary']> & { api_key?: string };
      apiKey?: string;
    };
    const targetRole = role === 'worker' ? 'worker' : 'primary';
    const profileSecrets = loadProfileSecrets(username);
    const persisted = store.getLLMConfig();
    const result = await probeProviderModelDirect({
      role: targetRole,
      override,
      persistedRoleConfig: persisted[targetRole],
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
    const saved = store.saveMCPServers(Array.isArray(servers) ? servers : []);
    res.json({ ok: true, servers: saved });
  });

  // ---------------------------------------------------------------------------
  // Frontend SPA (Nemofy Agent via Vite dev middleware or built dist/)
  // ---------------------------------------------------------------------------
  const isProd = process.env.NODE_ENV === 'production';
  const distDir = path.resolve(cwd, 'dist');
  const distIndex = path.join(distDir, 'index.html');

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else if (fs.existsSync(distIndex)) {
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(distIndex);
    });
  }

  const server = http.createServer(app);
  server.requestTimeout = 300_000;

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`[nemofy-agent] Server running on http://0.0.0.0:${PORT}`);
    void agentManager.start();
  });

  const handleShutdown = async (signal: string) => {
    console.log(`[nemofy-agent] Received ${signal}; shutting down...`);
    await agentManager.stop();
    server.close(() => process.exit(0));
  };

  process.once('SIGINT', () => void handleShutdown('SIGINT'));
  process.once('SIGTERM', () => void handleShutdown('SIGTERM'));
}

void startServer();
