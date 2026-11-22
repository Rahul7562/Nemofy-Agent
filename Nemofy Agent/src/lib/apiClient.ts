import type {
  AgentHealthResponse,
  AuthSession,
  ChatAttachment,
  ChatMessage,
  ChatThread,
  LiveAgentActivity,
  MCPServerItem,
  ModelRoleKey,
  PersistedLLMConfig,
  ProfileVaultSummary,
  ProviderAvailability,
  ProviderModelsResponse,
  RoleLLMConfig,
  SkillDocument,
  WorkspaceNode,
} from '../types/agent.ts';
import {
  DEFAULT_ADMIN_USERNAME,
  hashStandaloneCredential,
} from './authValidation.ts';
import {
  LLM_PROVIDER_PRESETS,
  classifyFileModality,
  getDefaultAccessoriesConfig,
  getProviderPreset,
  normalizeAccessoriesConfig,
} from './llmPresets.ts';

const SESSION_TOKEN_KEY = 'nemofy_session_token';
const STANDALONE_DIGEST_KEY = 'nemofy_standalone_digest';
const STANDALONE_SESSION_KEY = 'nemofy_standalone_session';
const REMOTE_BACKEND_URL_KEY = 'nemofy_remote_backend_url';
const STANDALONE_LLM_KEY = 'nemofy_standalone_llm';
const STANDALONE_THREADS_KEY = 'nemofy_standalone_threads';
const STANDALONE_FILES_KEY = 'nemofy_standalone_files';
const STANDALONE_SKILLS_KEY = 'nemofy_standalone_skills';
const STANDALONE_MCPS_KEY = 'nemofy_standalone_mcps';

let detectedStandaloneMode = false;
let inMemorySessionToken: string | null = null;

export function getRemoteBackendUrl(): string {
  const fromStorage =
    typeof window !== 'undefined' ? window.localStorage.getItem(REMOTE_BACKEND_URL_KEY) || '' : '';
  const fromEnv =
    typeof import.meta !== 'undefined' && import.meta.env?.VITE_AGENT_API_URL
      ? String(import.meta.env.VITE_AGENT_API_URL)
      : '';
  return (fromStorage || fromEnv).trim().replace(/\/+$/, '');
}

export function setRemoteBackendUrl(url: string): void {
  const clean = url.trim().replace(/\/+$/, '');
  if (typeof window !== 'undefined') {
    if (clean) {
      window.localStorage.setItem(REMOTE_BACKEND_URL_KEY, clean);
    } else {
      window.localStorage.removeItem(REMOTE_BACKEND_URL_KEY);
    }
  }
  detectedStandaloneMode = false;
}

export function isStandaloneMode(): boolean {
  return detectedStandaloneMode;
}

function getSessionToken(): string | null {
  return inMemorySessionToken;
}

function setSessionToken(token: string | null): void {
  inMemorySessionToken = token || null;
  if (typeof window !== 'undefined') {
    window.sessionStorage.removeItem(SESSION_TOKEN_KEY);
  }
}

function buildUrl(apiPath: string): string {
  const base = getRemoteBackendUrl();
  return base ? `${base}${apiPath}` : apiPath;
}

async function apiFetch<T>(
  apiPath: string,
  options: RequestInit = {},
): Promise<{ ok: boolean; status: number; data: T; isHtmlFallback: boolean }> {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  if (!headers.has('X-Requested-With')) {
    headers.set('X-Requested-With', 'NemofyAgent');
  }
  const token = getSessionToken();
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const resp = await fetch(buildUrl(apiPath), {
    ...options,
    headers,
    credentials: 'include',
  });

  const contentType = resp.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    return {
      ok: false,
      status: resp.status,
      data: {} as T,
      isHtmlFallback: true,
    };
  }

  const data = (await resp.json()) as T;
  return { ok: resp.ok, status: resp.status, data, isHtmlFallback: false };
}

function extractApiError(
  res: { status: number; data: unknown; isHtmlFallback: boolean },
  fallback: string,
): string {
  const dataObj = res.data && typeof res.data === 'object' ? (res.data as { error?: string; detail?: string }) : null;
  if (dataObj?.error && typeof dataObj.error === 'string' && dataObj.error.trim()) {
    return dataObj.error.trim();
  }
  if (dataObj?.detail && typeof dataObj.detail === 'string' && dataObj.detail.trim()) {
    return dataObj.detail.trim();
  }
  if (res.isHtmlFallback && res.status >= 400) {
    return `${fallback} (HTTP ${res.status} gateway/proxy response).`;
  }
  return fallback;
}

function getDefaultLLMConfig(): PersistedLLMConfig {
  return {
    primary: {
      provider: 'river_ai',
      model: '',
      base_url: 'https://api.river.ai/v1',
      temperature: 0.2,
      max_tokens: 16384,
      reasoning_effort: 'off',
    },
    worker: {
      provider: 'river_ai',
      model: 'deepseek-ai/DeepSeek-V4.1-Flash',
      base_url: 'https://api.river.ai/v1',
      temperature: 0.1,
      max_tokens: 8192,
      reasoning_effort: 'off',
    },
    accessories: getDefaultAccessoriesConfig(),
    max_turns: 30,
  };
}

function readLocalJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeLocalJson(key: string, val: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(val));
  } catch {
    // ignore quota errors
  }
}

// ---------------------------------------------------------------------------
// Auth API
// ---------------------------------------------------------------------------
export async function checkAuthSession(): Promise<AuthSession> {
  try {
    const res = await apiFetch<AuthSession>('/api/auth/me');
    if (!res.isHtmlFallback && res.status !== 404) {
      detectedStandaloneMode = false;
      return { ...res.data, mode: 'server' };
    }
  } catch {
    // fall through to standalone static Vercel mode
  }

  detectedStandaloneMode = true;
  const savedDigest = window.localStorage.getItem(STANDALONE_DIGEST_KEY);
  const active = window.sessionStorage.getItem(STANDALONE_SESSION_KEY);
  if (savedDigest && active) {
    return { authenticated: true, username: active, mode: 'standalone' };
  }
  return { authenticated: false, needsSetup: !savedDigest, mode: 'standalone' };
}

export async function setupAdminPassword(
  username: string,
  password: string,
): Promise<{ ok: boolean; username?: string; error?: string }> {
  const cleanUser = username.trim() || DEFAULT_ADMIN_USERNAME;
  if (!detectedStandaloneMode) {
    try {
      const res = await apiFetch<{
        authenticated?: boolean;
        username?: string;
        token?: string;
        error?: string;
      }>('/api/auth/setup', {
        method: 'POST',
        body: JSON.stringify({ username: cleanUser, password }),
      });
      if (!res.isHtmlFallback && res.status !== 404) {
        if (!res.ok || !res.data.authenticated) {
          return { ok: false, error: res.data.error || 'Unable to complete administrator setup.' };
        }
        if (res.data.token) setSessionToken(res.data.token);
        return { ok: true, username: res.data.username || cleanUser };
      }
      detectedStandaloneMode = true;
    } catch {
      detectedStandaloneMode = true;
    }
  }

  if (window.localStorage.getItem(STANDALONE_DIGEST_KEY)) {
    return { ok: false, error: 'Administrator account is already configured.' };
  }
  const digest = await hashStandaloneCredential(cleanUser, password);
  window.localStorage.setItem(STANDALONE_DIGEST_KEY, digest);
  window.sessionStorage.setItem(STANDALONE_SESSION_KEY, cleanUser);
  return { ok: true, username: cleanUser };
}

export async function loginWithCredentials(
  username: string,
  password: string,
): Promise<{ ok: boolean; username?: string; error?: string }> {
  if (!detectedStandaloneMode) {
    try {
      const res = await apiFetch<{
        authenticated?: boolean;
        username?: string;
        token?: string;
        error?: string;
      }>('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
      });
      if (!res.isHtmlFallback && res.status !== 404) {
        if (!res.ok || !res.data.authenticated) {
          return { ok: false, error: res.data.error || 'Invalid username or password.' };
        }
        if (res.data.token) setSessionToken(res.data.token);
        return { ok: true, username: res.data.username || DEFAULT_ADMIN_USERNAME };
      }
      detectedStandaloneMode = true;
    } catch {
      detectedStandaloneMode = true;
    }
  }

  const expectedDigest = window.localStorage.getItem(STANDALONE_DIGEST_KEY);
  if (!expectedDigest) {
    return { ok: false, error: 'Please complete first-time administrator setup first.' };
  }
  const candidate = await hashStandaloneCredential(username, password);
  if (candidate === expectedDigest) {
    const cleanUser = username.trim() || DEFAULT_ADMIN_USERNAME;
    window.sessionStorage.setItem(STANDALONE_SESSION_KEY, cleanUser);
    return { ok: true, username: cleanUser };
  }
  return { ok: false, error: 'Invalid username or password.' };
}

export async function logoutSession(): Promise<void> {
  if (!detectedStandaloneMode) {
    try {
      await apiFetch('/api/auth/logout', { method: 'POST' });
    } catch {
      // ignore
    }
  }
  setSessionToken(null);
  window.sessionStorage.removeItem(STANDALONE_SESSION_KEY);
}

export async function changeAdminPassword(
  currentPassword: string,
  newPassword: string,
): Promise<{ ok: boolean; message?: string; error?: string }> {
  if (!detectedStandaloneMode) {
    const res = await apiFetch<{ ok?: boolean; token?: string; message?: string; error?: string }>(
      '/api/auth/change-password',
      {
        method: 'POST',
        body: JSON.stringify({ currentPassword, newPassword }),
      },
    );
    if (!res.ok) {
      return { ok: false, error: res.data.error || 'Unable to update password.' };
    }
    if (res.data.token) setSessionToken(res.data.token);
    return { ok: true, message: res.data.message || 'Password updated successfully.' };
  }

  const expectedDigest = window.localStorage.getItem(STANDALONE_DIGEST_KEY);
  if (!expectedDigest) {
    return { ok: false, error: 'Administrator account is not configured yet.' };
  }
  const activeUser = window.sessionStorage.getItem(STANDALONE_SESSION_KEY) || DEFAULT_ADMIN_USERNAME;
  const currentHash = await hashStandaloneCredential(activeUser, currentPassword);
  if (currentHash !== expectedDigest) {
    return { ok: false, error: 'Current password is incorrect.' };
  }
  const nextHash = await hashStandaloneCredential(activeUser, newPassword);
  window.localStorage.setItem(STANDALONE_DIGEST_KEY, nextHash);
  return { ok: true, message: 'Password updated for this browser deployment.' };
}

// ---------------------------------------------------------------------------
// Agent Health & Chat Threads
// ---------------------------------------------------------------------------
export async function fetchAgentHealth(): Promise<AgentHealthResponse> {
  if (detectedStandaloneMode) {
    return {
      connected: false,
      core: null,
      error: 'Standalone UI mode. Configure a remote Agent-Core backend URL in Settings to run live Python agent tasks.',
    };
  }
  const res = await apiFetch<AgentHealthResponse>('/api/agent/health');
  if (!res.ok) {
    return { connected: false, core: null, error: 'Unable to reach server.' };
  }
  return res.data;
}

export async function listThreads(): Promise<ChatThread[]> {
  if (detectedStandaloneMode) {
    return readLocalJson<ChatThread[]>(STANDALONE_THREADS_KEY, []);
  }
  const res = await apiFetch<{ threads: ChatThread[] }>('/api/threads');
  if (!res.ok) throw new Error('Unable to load chat sessions.');
  return res.data.threads || [];
}

export async function createThread(title?: string): Promise<ChatThread> {
  if (detectedStandaloneMode) {
    const now = new Date().toISOString();
    const thread: ChatThread = {
      id: `thread-${Math.random().toString(16).slice(2, 10)}`,
      title: title || 'New Agent Session',
      createdAt: now,
      updatedAt: now,
      messages: [],
      pendingApproval: null,
    };
    const next = [thread, ...readLocalJson<ChatThread[]>(STANDALONE_THREADS_KEY, [])];
    writeLocalJson(STANDALONE_THREADS_KEY, next);
    return thread;
  }
  const res = await apiFetch<{ thread: ChatThread }>('/api/threads', {
    method: 'POST',
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error('Unable to create session.');
  return res.data.thread;
}

export async function deleteThread(threadId: string): Promise<void> {
  if (detectedStandaloneMode) {
    const next = readLocalJson<ChatThread[]>(STANDALONE_THREADS_KEY, []).filter(
      (t) => t.id !== threadId,
    );
    writeLocalJson(STANDALONE_THREADS_KEY, next);
    return;
  }
  await apiFetch(`/api/threads/${encodeURIComponent(threadId)}`, { method: 'DELETE' });
}

export async function uploadChatMedia(
  file: File,
  threadId?: string,
): Promise<ChatAttachment> {
  if (detectedStandaloneMode) {
    const kind = classifyFileModality(file.type, file.name);
    if (!kind) {
      throw new Error('Unsupported file type. Upload an image, audio, or video file.');
    }
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('Failed to read media file.'));
      reader.readAsDataURL(file);
    });
    return {
      id: `att-${Math.random().toString(16).slice(2, 12)}`,
      name: file.name,
      kind,
      mimeType: file.type || `${kind}/octet-stream`,
      size: file.size,
      url: dataUrl,
    };
  }

  const headers = new Headers();
  headers.set('Content-Type', file.type || 'application/octet-stream');
  headers.set('X-File-Name', encodeURIComponent(file.name));
  if (file.type) headers.set('X-File-Type', file.type);
  if (threadId) headers.set('X-Thread-Id', threadId);
  headers.set('X-Requested-With', 'NemofyAgent');
  const token = getSessionToken();
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const resp = await fetch(buildUrl('/api/media/upload'), {
    method: 'POST',
    headers,
    body: file,
    credentials: 'include',
  });
  const payload = (await resp.json().catch(() => ({}))) as {
    attachment?: ChatAttachment;
    error?: string;
  };
  if (!resp.ok || !payload.attachment) {
    throw new Error(payload.error || 'Failed to upload media attachment.');
  }
  return payload.attachment;
}

export async function sendChatMessage(
  threadId: string,
  message: string,
  reasoningEffort?: string,
  attachments?: ChatAttachment[],
): Promise<{ thread: ChatThread; message: ChatMessage; error?: string }> {
  if (detectedStandaloneMode) {
    const now = new Date().toISOString();
    const threads = readLocalJson<ChatThread[]>(STANDALONE_THREADS_KEY, []);
    let thread = threads.find((t) => t.id === threadId);
    if (!thread) {
      thread = {
        id: threadId,
        title: (message || attachments?.[0]?.name || 'Media Session').slice(0, 56),
        createdAt: now,
        updatedAt: now,
        messages: [],
      };
      threads.unshift(thread);
    }
    thread.messages.push({
      id: `msg-${Date.now()}-u`,
      role: 'user',
      content: message,
      timestamp: now,
      attachments: attachments && attachments.length > 0 ? attachments : undefined,
    });
    const reply: ChatMessage = {
      id: `msg-${Date.now()}-a`,
      role: 'assistant',
      content:
        'This frontend is currently running in static mode without a connected Agent-Core API server. Open Settings > Appearance & Runtime to configure your hosted Agent-Core backend URL.',
      timestamp: new Date().toISOString(),
      status: 'error',
    };
    thread.messages.push(reply);
    thread.updatedAt = reply.timestamp;
    writeLocalJson(STANDALONE_THREADS_KEY, threads);
    return { thread, message: reply, error: 'Backend URL not configured' };
  }

  const res = await apiFetch<{ thread: ChatThread; message: ChatMessage; error?: string }>(
    '/api/chat/send',
    {
      method: 'POST',
      body: JSON.stringify({
        threadId,
        message,
        reasoningEffort,
        attachments: attachments && attachments.length > 0 ? attachments : undefined,
      }),
    },
  );
  if (!res.ok) {
    throw new Error((res.data as { error?: string }).error || 'Failed to send prompt to Agent-Core.');
  }
  return res.data;
}

export async function startChatJob(
  threadId: string,
  message: string,
  reasoningEffort?: string,
  signal?: AbortSignal,
  attachments?: ChatAttachment[],
): Promise<{
  jobId: string | null;
  thread: ChatThread;
  activity?: LiveAgentActivity;
  message?: ChatMessage;
  error?: string;
}> {
  if (detectedStandaloneMode) {
    const fallback = await sendChatMessage(threadId, message, reasoningEffort, attachments);
    return {
      jobId: null,
      thread: fallback.thread,
      message: fallback.message,
      error: fallback.error,
    };
  }

  const res = await apiFetch<{
    jobId: string | null;
    thread: ChatThread;
    activity?: LiveAgentActivity;
    message?: ChatMessage;
    error?: string;
  }>('/api/chat/jobs/start', {
    method: 'POST',
    body: JSON.stringify({
      threadId,
      message,
      reasoningEffort,
      attachments: attachments && attachments.length > 0 ? attachments : undefined,
    }),
    signal,
  });

  if (!res.ok) {
    throw new Error(extractApiError(res, 'Failed to start agent task.'));
  }
  return res.data;
}

export async function startResumeApprovalJob(
  threadId: string,
  decision: 'approve' | 'reject' | 'edit',
  args?: Record<string, unknown>,
  reason?: string,
  signal?: AbortSignal,
): Promise<{
  jobId: string | null;
  alreadyResolved?: boolean;
  thread?: ChatThread | null;
  activity?: LiveAgentActivity;
  message?: ChatMessage;
  error?: string;
}> {
  const res = await apiFetch<{
    jobId: string | null;
    alreadyResolved?: boolean;
    thread?: ChatThread | null;
    activity?: LiveAgentActivity;
    message?: ChatMessage;
    error?: string;
  }>('/api/chat/jobs/resume', {
    method: 'POST',
    body: JSON.stringify({ threadId, decision, arguments: args, reason }),
    signal,
  });

  if (!res.ok) {
    throw new Error(extractApiError(res, 'Failed to process approval.'));
  }
  return res.data;
}

export async function pollChatJob(
  jobId: string,
  waitMs = 650,
  signal?: AbortSignal,
): Promise<{
  done: boolean;
  activity: LiveAgentActivity;
  thread?: ChatThread | null;
  message?: ChatMessage;
}> {
  const res = await apiFetch<{
    done: boolean;
    activity: LiveAgentActivity;
    thread?: ChatThread | null;
    message?: ChatMessage;
    error?: string;
  }>(`/api/chat/jobs/${encodeURIComponent(jobId)}?wait_ms=${waitMs}`, { signal });

  if (!res.ok) {
    throw new Error(extractApiError(res, 'Failed to poll agent progress.'));
  }
  return res.data;
}

export async function cancelChatJob(
  jobId: string,
): Promise<{
  ok: boolean;
  activity?: LiveAgentActivity;
  thread?: ChatThread | null;
  message?: ChatMessage;
}> {
  if (detectedStandaloneMode) {
    return { ok: true };
  }
  const res = await apiFetch<{
    ok: boolean;
    activity?: LiveAgentActivity;
    thread?: ChatThread | null;
    message?: ChatMessage;
    error?: string;
  }>(`/api/chat/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) {
    return { ok: false };
  }
  return res.data;
}

export async function resumeApproval(
  threadId: string,
  decision: 'approve' | 'reject' | 'edit',
  args?: Record<string, unknown>,
  reason?: string,
): Promise<{ thread?: ChatThread; message?: ChatMessage; alreadyResolved?: boolean }> {
  const res = await apiFetch<{
    thread?: ChatThread;
    message?: ChatMessage;
    alreadyResolved?: boolean;
    error?: string;
  }>('/api/chat/resume', {
    method: 'POST',
    body: JSON.stringify({ threadId, decision, arguments: args, reason }),
  });
  if (!res.ok) {
    throw new Error(extractApiError(res, 'Failed to process approval.'));
  }
  return res.data;
}

// ---------------------------------------------------------------------------
// Workspace Filesystem API
// ---------------------------------------------------------------------------
export async function fetchWorkspaceTree(dir?: string): Promise<{
  root: string;
  dir?: string;
  entries: WorkspaceNode[];
  totalCount?: number;
  truncated?: boolean;
}> {
  if (detectedStandaloneMode) {
    const files = readLocalJson<Record<string, string>>(STANDALONE_FILES_KEY, {
      'README.md': '# Nemofy Agent Workspace\n\nConnect a hosted Agent-Core server in Settings to sync with the live Python sandbox `/workspace` directory.\n',
    });
    const entries: WorkspaceNode[] = Object.keys(files)
      .sort()
      .map((p) => ({
        name: p.split('/').pop() || p,
        path: p,
        type: 'file',
        size: files[p].length,
        modifiedAt: new Date().toISOString(),
      }));
    return { root: '/workspace', entries, totalCount: entries.length, truncated: false };
  }
  const cleanDir = (dir || '').trim();
  const query = cleanDir ? `?dir=${encodeURIComponent(cleanDir)}` : '';
  const res = await apiFetch<{
    root: string;
    dir?: string;
    entries: WorkspaceNode[];
    totalCount?: number;
    truncated?: boolean;
    error?: string;
  }>(`/api/filesystem/tree${query}`);
  if (!res.ok) throw new Error(res.data?.error || 'Unable to read workspace tree.');
  return res.data;
}

export async function readWorkspaceFile(
  filePath: string,
): Promise<{ path: string; content: string; size: number; modifiedAt: string }> {
  if (detectedStandaloneMode) {
    const files = readLocalJson<Record<string, string>>(STANDALONE_FILES_KEY, {});
    const content = files[filePath] ?? '';
    return { path: filePath, content, size: content.length, modifiedAt: new Date().toISOString() };
  }
  const res = await apiFetch<{ path: string; content: string; size: number; modifiedAt: string; error?: string }>(
    `/api/filesystem/file?path=${encodeURIComponent(filePath)}`,
  );
  if (!res.ok) throw new Error(res.data.error || 'Unable to read file.');
  return res.data;
}

export async function saveWorkspaceFile(
  filePath: string,
  content: string,
  options?: { createOnly?: boolean },
): Promise<void> {
  const createOnly = Boolean(options?.createOnly);
  if (detectedStandaloneMode) {
    const files = readLocalJson<Record<string, string>>(STANDALONE_FILES_KEY, {});
    if (createOnly && Object.prototype.hasOwnProperty.call(files, filePath)) {
      throw new Error(`File already exists: /workspace/${filePath}`);
    }
    files[filePath] = content;
    writeLocalJson(STANDALONE_FILES_KEY, files);
    return;
  }
  const res = await apiFetch<{ ok?: boolean; error?: string }>('/api/filesystem/file', {
    method: 'PUT',
    body: JSON.stringify({ path: filePath, content, ...(createOnly ? { createOnly: true } : {}) }),
  });
  if (!res.ok) throw new Error(res.data.error || 'Unable to save file.');
}

export async function createWorkspaceFolder(folderPath: string): Promise<void> {
  if (detectedStandaloneMode) {
    const files = readLocalJson<Record<string, string>>(STANDALONE_FILES_KEY, {});
    files[`${folderPath.replace(/\/+$/, '')}/.keep`] = '';
    writeLocalJson(STANDALONE_FILES_KEY, files);
    return;
  }
  const res = await apiFetch<{ ok?: boolean; error?: string }>('/api/filesystem/folder', {
    method: 'POST',
    body: JSON.stringify({ path: folderPath }),
  });
  if (!res.ok) throw new Error(res.data.error || 'Unable to create folder.');
}

export async function deleteWorkspaceEntry(targetPath: string): Promise<void> {
  if (detectedStandaloneMode) {
    const files = readLocalJson<Record<string, string>>(STANDALONE_FILES_KEY, {});
    delete files[targetPath];
    writeLocalJson(STANDALONE_FILES_KEY, files);
    return;
  }
  const res = await apiFetch<{ ok?: boolean; error?: string }>(
    `/api/filesystem/entry?path=${encodeURIComponent(targetPath)}`,
    { method: 'DELETE' },
  );
  if (!res.ok) throw new Error(res.data.error || 'Unable to delete entry.');
}

// ---------------------------------------------------------------------------
// Settings: LLM Config, Skills, and MCP Servers
// ---------------------------------------------------------------------------
export async function fetchLLMSettings(): Promise<{
  config: PersistedLLMConfig;
  providers: Record<string, ProviderAvailability>;
  vault?: ProfileVaultSummary;
}> {
  if (detectedStandaloneMode) {
    const providers: Record<string, ProviderAvailability> = {};
    for (const p of LLM_PROVIDER_PRESETS) {
      providers[p.id] = {
        base_url: p.baseUrl,
        env_keys: p.envVars,
        primary_env_var: p.envVars[0] || 'PRIMARY_API_KEY',
        api_key_configured: !p.requiresApiKey,
        masked_key: null,
        requires_api_key: p.requiresApiKey,
      };
    }
    const rawConfig = readLocalJson<PersistedLLMConfig>(STANDALONE_LLM_KEY, getDefaultLLMConfig());
    return {
      config: {
        ...rawConfig,
        accessories: normalizeAccessoriesConfig(rawConfig.accessories),
      },
      providers,
      vault: {
        username: DEFAULT_ADMIN_USERNAME,
        encrypted: false,
        algorithm: 'standalone',
        entries: [],
      },
    };
  }
  const res = await apiFetch<{
    config: PersistedLLMConfig;
    providers: Record<string, ProviderAvailability>;
    vault?: ProfileVaultSummary;
  }>('/api/settings/llm');
  if (!res.ok) throw new Error('Unable to load LLM settings.');
  return {
    ...res.data,
    config: {
      ...res.data.config,
      accessories: normalizeAccessoriesConfig(res.data.config?.accessories),
    },
  };
}

export async function saveLLMSettings(
  config: PersistedLLMConfig,
  apiKeys?: Partial<Record<ModelRoleKey, string>>,
): Promise<{
  config: PersistedLLMConfig;
  providers: Record<string, ProviderAvailability>;
  savedEnvKeys?: string[];
  vault?: ProfileVaultSummary;
  syncWarning?: string | null;
}> {
  if (detectedStandaloneMode) {
    writeLocalJson(STANDALONE_LLM_KEY, config);
    return fetchLLMSettings();
  }
  const res = await apiFetch<{
    config: PersistedLLMConfig;
    providers: Record<string, ProviderAvailability>;
    savedEnvKeys?: string[];
    vault?: ProfileVaultSummary;
    syncWarning?: string | null;
    error?: string;
  }>('/api/settings/llm', {
    method: 'PUT',
    body: JSON.stringify({
      ...config,
      api_keys: apiKeys,
    }),
  });
  if (!res.ok) throw new Error(res.data.error || 'Unable to save LLM settings.');
  return {
    ...res.data,
    config: {
      ...res.data.config,
      accessories: normalizeAccessoriesConfig(res.data.config?.accessories),
    },
  };
}

export async function saveProfileVaultKey(params: {
  provider?: string;
  env_var?: string;
  api_key: string;
}): Promise<{
  ok: boolean;
  savedKey: string;
  vault: ProfileVaultSummary;
  providers: Record<string, ProviderAvailability>;
}> {
  if (detectedStandaloneMode) {
    throw new Error('Encrypted profile key vault requires a connected Nemofy Agent server.');
  }
  const res = await apiFetch<{
    ok: boolean;
    savedKey: string;
    vault: ProfileVaultSummary;
    providers: Record<string, ProviderAvailability>;
    error?: string;
  }>('/api/settings/vault', {
    method: 'PUT',
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(res.data.error || 'Unable to save API key to profile vault.');
  return res.data;
}

export async function deleteProfileVaultKey(envVar: string): Promise<{
  ok: boolean;
  deleted: boolean;
  vault: ProfileVaultSummary;
  providers: Record<string, ProviderAvailability>;
}> {
  if (detectedStandaloneMode) {
    throw new Error('Encrypted profile key vault requires a connected Nemofy Agent server.');
  }
  const res = await apiFetch<{
    ok: boolean;
    deleted: boolean;
    vault: ProfileVaultSummary;
    providers: Record<string, ProviderAvailability>;
    error?: string;
  }>(`/api/settings/vault/${encodeURIComponent(envVar)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(res.data.error || 'Unable to remove API key from profile vault.');
  return res.data;
}

export async function fetchProviderModels(params: {
  provider: string;
  base_url?: string;
  api_key?: string;
  role?: ModelRoleKey;
}): Promise<ProviderModelsResponse> {
  if (detectedStandaloneMode) {
    const preset = getProviderPreset(params.provider);
    const models = (preset?.defaultModels || []).map((id) => ({
      id,
      is_free: id.toLowerCase().endsWith(':free'),
      source: 'preset' as const,
    }));
    return {
      ok: true,
      provider: params.provider,
      base_url: params.base_url || preset?.baseUrl || '',
      source: 'preset',
      models,
      count: models.length,
      error: null,
    };
  }

  const res = await apiFetch<ProviderModelsResponse>('/api/settings/models', {
    method: 'POST',
    body: JSON.stringify({
      provider: params.provider,
      base_url: params.base_url?.trim() || undefined,
      api_key: params.api_key?.trim() || undefined,
      role: params.role || 'primary',
    }),
  });

  if (!res.ok || !res.data || !Array.isArray(res.data.models)) {
    return {
      ok: false,
      provider: params.provider,
      base_url: params.base_url || '',
      source: 'error',
      models: [],
      count: 0,
      error:
        (res.data as { error?: string } | undefined)?.error ||
        'Unable to query live models from this provider.',
    };
  }

  return res.data;
}

export async function probeLLMModel(
  role: ModelRoleKey,
  override: RoleLLMConfig,
  apiKey?: string,
): Promise<Record<string, unknown>> {
  if (detectedStandaloneMode) {
    return {
      ok: false,
      detail: 'Connect an Agent-Core backend URL in Settings to run live model probes.',
    };
  }
  const res = await apiFetch<Record<string, unknown>>('/api/settings/probe', {
    method: 'POST',
    body: JSON.stringify({ role, override, apiKey }),
  });
  return res.data;
}

export async function fetchSkills(): Promise<SkillDocument[]> {
  if (detectedStandaloneMode) {
    return readLocalJson<SkillDocument[]>(STANDALONE_SKILLS_KEY, []);
  }
  const res = await apiFetch<{ skills: SkillDocument[] }>('/api/skills');
  if (!res.ok) throw new Error('Unable to load skills.');
  return res.data.skills || [];
}

export async function saveSkillDocument(name: string, content: string): Promise<SkillDocument> {
  if (detectedStandaloneMode) {
    const list = readLocalJson<SkillDocument[]>(STANDALONE_SKILLS_KEY, []);
    const doc: SkillDocument = {
      name,
      description: 'Custom skill',
      filename: `${name}.md`,
      content,
    };
    const next = [...list.filter((s) => s.name !== name), doc];
    writeLocalJson(STANDALONE_SKILLS_KEY, next);
    return doc;
  }
  const res = await apiFetch<{ skill: SkillDocument; error?: string }>(
    `/api/skills/${encodeURIComponent(name)}`,
    {
      method: 'PUT',
      body: JSON.stringify({ content }),
    },
  );
  if (!res.ok) throw new Error(res.data.error || 'Unable to save skill.');
  return res.data.skill;
}

export async function deleteSkillDocument(name: string): Promise<void> {
  if (detectedStandaloneMode) {
    const next = readLocalJson<SkillDocument[]>(STANDALONE_SKILLS_KEY, []).filter(
      (s) => s.name !== name,
    );
    writeLocalJson(STANDALONE_SKILLS_KEY, next);
    return;
  }
  const res = await apiFetch<{ ok?: boolean; error?: string }>(
    `/api/skills/${encodeURIComponent(name)}`,
    { method: 'DELETE' },
  );
  if (!res.ok) throw new Error(res.data.error || 'Unable to delete skill.');
}

export async function fetchMCPServers(): Promise<MCPServerItem[]> {
  if (detectedStandaloneMode) {
    return readLocalJson<MCPServerItem[]>(STANDALONE_MCPS_KEY, []);
  }
  const res = await apiFetch<{ servers: MCPServerItem[] }>('/api/mcps');
  if (!res.ok) throw new Error('Unable to load MCP servers.');
  return res.data.servers || [];
}

export async function saveMCPServers(servers: MCPServerItem[]): Promise<MCPServerItem[]> {
  if (detectedStandaloneMode) {
    writeLocalJson(STANDALONE_MCPS_KEY, servers);
    return servers;
  }
  const res = await apiFetch<{ servers: MCPServerItem[]; error?: string }>('/api/mcps', {
    method: 'PUT',
    body: JSON.stringify({ servers }),
  });
  if (!res.ok) throw new Error(res.data.error || 'Unable to save MCP servers.');
  return res.data.servers || [];
}
