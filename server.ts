import 'dotenv/config';
import crypto from 'node:crypto';
import express, { type NextFunction, type Request, type Response } from 'express';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { AuthStore, parseCookieHeader } from './server/authStore.ts';
import { AgentCoreProcessManager } from './server/agentProcessManager.ts';
import { WorkspaceAndConfigStore, type ChatThreadRecord } from './server/configStore.ts';

const cwd = process.cwd();
const PORT = Number(process.env.PORT || '3000');
const SESSION_COOKIE = 'nemofy_session';

const authStore = new AuthStore();
const store = new WorkspaceAndConfigStore(cwd);
const agentManager = new AgentCoreProcessManager(cwd);

function getAgentApiToken(): string {
  return (process.env.AGENT_API_TOKEN || '').trim();
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

function setSessionCookie(res: Response, token: string): void {
  const parts = [
    `${SESSION_COOKIE}=${encodeURIComponent(token)}`,
    'Path=/',
    'HttpOnly',
    'SameSite=Lax',
    `Max-Age=${24 * 60 * 60}`,
  ];
  res.setHeader('Set-Cookie', parts.join('; '));
}

function clearSessionCookie(res: Response): void {
  res.setHeader(
    'Set-Cookie',
    `${SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`,
  );
}

function requireSession(req: Request, res: Response, next: NextFunction): void {
  const token = extractSessionToken(req);
  const session = authStore.verifySessionToken(token);
  if (!session) {
    res.status(401).json({ error: 'Authentication required. Please sign in.' });
    return;
  }
  next();
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
    return {
      ok: false,
      status: 503,
      data: {
        detail: `Python Agent-Core service unreachable: ${err instanceof Error ? err.message : String(err)}`,
      },
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
    res.json({
      authenticated: true,
      username: session.username,
      updatedAt: authStore.getUpdatedAt(),
    });
  });

  app.post('/api/auth/login', (req, res) => {
    const { username, password } = (req.body || {}) as { username?: string; password?: string };
    if (!username || !password) {
      res.status(400).json({ error: 'Username and password are required.' });
      return;
    }
    if (!authStore.verifyCredentials(username, password)) {
      res.status(401).json({ error: 'Invalid username or password.' });
      return;
    }
    const token = authStore.createSessionToken();
    setSessionCookie(res, token);
    res.json({
      authenticated: true,
      username: authStore.getUsername(),
      token,
    });
  });

  app.post('/api/auth/logout', (_req, res) => {
    clearSessionCookie(res);
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
    setSessionCookie(res, result.token);
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
    const { threadId, message } = (req.body || {}) as { threadId?: string; message?: string };
    const text = (message || '').trim();
    if (!text) {
      res.status(400).json({ error: 'Message cannot be empty.' });
      return;
    }

    const threads = store.listThreads();
    const now = new Date().toISOString();
    let thread = threads.find((t) => t.id === threadId);
    if (!thread) {
      thread = {
        id: threadId || `thread-${crypto.randomBytes(5).toString('hex')}`,
        title: text.slice(0, 56),
        createdAt: now,
        updatedAt: now,
        messages: [],
        pendingApproval: null,
      };
      threads.unshift(thread);
    } else if (thread.messages.length === 0) {
      thread.title = text.slice(0, 56);
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

    const llmCfg = store.getLLMConfig();
    const enabledMcps = store.listMCPServers().filter((s) => s.enabled);

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
          reasoning_effort:
            llmCfg.primary.reasoning_effort !== 'off' ? llmCfg.primary.reasoning_effort : undefined,
        },
        worker_model: {
          provider: llmCfg.worker.provider,
          model: llmCfg.worker.model,
          base_url: llmCfg.worker.base_url,
          temperature: llmCfg.worker.temperature,
          max_tokens: llmCfg.worker.max_tokens,
          reasoning_effort:
            llmCfg.worker.reasoning_effort !== 'off' ? llmCfg.worker.reasoning_effort : undefined,
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

    const assistantMsg = {
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
      pendingApproval,
      status: statusVal,
    };

    thread.messages.push(assistantMsg);
    thread.pendingApproval = pendingApproval;
    thread.updatedAt = replyNow;
    store.saveThreads(threads);
    res.json({ thread, message: assistantMsg });
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
    const agentResp = await callAgentCore('/agent/resume', {
      method: 'POST',
      body: {
        thread_id: threadId,
        decision,
        arguments: editedArgs,
        reason,
        primary_model: llmCfg.primary,
        worker_model: llmCfg.worker,
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
    res.json({
      config: store.getLLMConfig(),
      providers: store.getProviderAvailability(),
    });
  });

  app.put('/api/settings/llm', requireSession, async (req, res) => {
    const body = (req.body || {}) as {
      primary?: Partial<ReturnType<typeof store.getLLMConfig>['primary']>;
      worker?: Partial<ReturnType<typeof store.getLLMConfig>['worker']>;
      max_turns?: number;
    };
    const saved = store.saveLLMConfig(body as Parameters<typeof store.saveLLMConfig>[0]);
    // Also sync to running Agent-Core if reachable
    await callAgentCore('/config/llm', { method: 'PUT', body: saved, timeoutMs: 3000 });
    res.json({
      ok: true,
      config: saved,
      providers: store.getProviderAvailability(),
    });
  });

  app.post('/api/settings/probe', requireSession, async (req, res) => {
    const { role, override } = (req.body || {}) as {
      role?: 'primary' | 'worker';
      override?: Record<string, unknown>;
    };
    const probe = await callAgentCore('/models/probe', {
      method: 'POST',
      body: { role: role || 'primary', override, run_inference: false },
      timeoutMs: 15_000,
    });
    res.status(probe.ok ? 200 : probe.status).json(probe.data);
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
