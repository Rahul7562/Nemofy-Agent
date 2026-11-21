import fs from 'node:fs';
import path from 'node:path';

export interface RoleLLMConfig {
  provider: string;
  model: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
  reasoning_effort: string;
}

export interface PersistedLLMConfig {
  primary: RoleLLMConfig;
  worker: RoleLLMConfig;
  max_turns: number;
}

export interface WorkspaceNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  size?: number;
  modifiedAt?: string;
  children?: WorkspaceNode[];
}

export interface SkillDocument {
  name: string;
  description: string;
  filename: string;
  content: string;
}

export interface MCPServerItem {
  name: string;
  url: string;
  transport: 'streamable_http' | 'sse';
  enabled: boolean;
  timeout_seconds: number;
}

export interface ToolEventRecord {
  id: string;
  tool: string;
  args_summary?: string;
  status: 'running' | 'ok' | 'error' | 'rejected';
  output_summary?: string;
  started_at?: string;
  finished_at?: string | null;
}

export interface FileChangeRecord {
  kind: 'file' | 'dir';
  action: 'created' | 'modified' | 'deleted';
  path: string;
  tool?: string;
  timestamp?: string;
}

export interface ChatMessageRecord {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  plan?: Array<{
    step_id: string;
    description: string;
    status: string;
    tool_or_action: string;
    result: string;
  }>;
  activeSkills?: string[];
  subagentReports?: Array<Record<string, unknown>>;
  activeModels?: {
    primary?: { provider?: string; model?: string; reasoning_effort?: string | null };
    worker?: { provider?: string; model?: string; reasoning_effort?: string | null };
  };
  thinking?: string | null;
  toolEvents?: ToolEventRecord[];
  fileChanges?: FileChangeRecord[];
  pendingApproval?: Record<string, unknown> | null;
  status?: string;
}

export interface ChatThreadRecord {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessageRecord[];
  pendingApproval?: Record<string, unknown> | null;
}

const PROVIDER_DEFAULTS: Record<string, { baseUrl: string; envVars: string[]; requiresKey: boolean }> = {
  river_ai: { baseUrl: 'https://api.river.ai/v1', envVars: ['RIVER_AI_API'], requiresKey: true },
  openai: { baseUrl: 'https://api.openai.com/v1', envVars: ['OPENAI_API_KEY'], requiresKey: true },
  openrouter: { baseUrl: 'https://openrouter.ai/api/v1', envVars: ['OPENROUTER_API_KEY'], requiresKey: true },
  nvidia: { baseUrl: 'https://integrate.api.nvidia.com/v1', envVars: ['NVIDIA_API_KEY'], requiresKey: true },
  gemini: {
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
    envVars: ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
    requiresKey: true,
  },
  deepseek: { baseUrl: 'https://api.deepseek.com/v1', envVars: ['DEEPSEEK_API_KEY'], requiresKey: true },
  groq: { baseUrl: 'https://api.groq.com/openai/v1', envVars: ['GROQ_API_KEY'], requiresKey: true },
  anthropic: { baseUrl: 'https://api.anthropic.com/v1', envVars: ['ANTHROPIC_API_KEY'], requiresKey: true },
  together: { baseUrl: 'https://api.together.xyz/v1', envVars: ['TOGETHER_API_KEY'], requiresKey: true },
  fireworks: { baseUrl: 'https://api.fireworks.ai/inference/v1', envVars: ['FIREWORKS_API_KEY'], requiresKey: true },
  ollama: { baseUrl: 'http://127.0.0.1:11434/v1', envVars: ['OLLAMA_API_KEY'], requiresKey: false },
  vllm: { baseUrl: 'http://127.0.0.1:8000/v1', envVars: ['VLLM_API_KEY'], requiresKey: false },
  custom: { baseUrl: '', envVars: ['PRIMARY_API_KEY', 'WORKER_API_KEY'], requiresKey: true },
};

const PROVIDER_ALIASES: Record<string, string> = {
  river: 'river_ai',
  google: 'gemini',
  nvidia_nim: 'nvidia',
  nim: 'nvidia',
  local: 'ollama',
};

export function normalizeProviderId(provider?: string | null): string {
  const clean = (provider || '').trim().toLowerCase();
  if (!clean) return 'river_ai';
  return PROVIDER_ALIASES[clean] || clean;
}

function normalizeRoleConfig(raw: Partial<RoleLLMConfig> | undefined, fallback: RoleLLMConfig): RoleLLMConfig {
  const provider = normalizeProviderId(raw?.provider || fallback.provider);
  const presetBase = PROVIDER_DEFAULTS[provider]?.baseUrl || '';
  const rawBase = typeof raw?.base_url === 'string' ? raw.base_url.trim().replace(/\/+$/, '') : '';
  const base_url = rawBase || presetBase || fallback.base_url;
  const model = typeof raw?.model === 'string' && raw.model.trim() ? raw.model.trim() : fallback.model;
  const rawTemp = Number(raw?.temperature);
  const temperature = Number.isFinite(rawTemp) && rawTemp >= 0 && rawTemp <= 2 ? rawTemp : fallback.temperature;
  const rawTokens = Number(raw?.max_tokens);
  const max_tokens =
    Number.isFinite(rawTokens) && rawTokens >= 64 && rawTokens <= 128000
      ? Math.round(rawTokens)
      : fallback.max_tokens;
  const reasoning_effort =
    typeof raw?.reasoning_effort === 'string' && raw.reasoning_effort.trim()
      ? raw.reasoning_effort.trim().toLowerCase()
      : fallback.reasoning_effort;

  return {
    provider,
    model,
    base_url,
    temperature,
    max_tokens,
    reasoning_effort,
  };
}

export function maskEnvSecret(value?: string | null): string | null {
  const clean = (value || '').trim();
  if (!clean) return null;
  if (clean.length <= 8) return '****';
  return `${clean.slice(0, 4)}...${clean.slice(-4)}`;
}

function formatEnvValue(raw: string): string {
  const clean = raw.replace(/\r?\n/g, '').trim();
  if (!clean) return '';
  if (/[\s#'"`$\\]/.test(clean)) {
    return `"${clean.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
  }
  return clean;
}

export function upsertEnvContent(existingContent: string, updates: Record<string, string>): string {
  const remaining = new Map<string, string>();
  for (const [k, v] of Object.entries(updates)) {
    const cleanKey = k.trim();
    if (!/^[A-Z][A-Z0-9_]*$/.test(cleanKey)) continue;
    remaining.set(cleanKey, v);
  }

  const lines = existingContent ? existingContent.split(/\r?\n/) : [];
  const nextLines: string[] = [];

  for (const line of lines) {
    const match = line.match(/^\s*([A-Z][A-Z0-9_]*)\s*=/);
    if (match && remaining.has(match[1])) {
      const key = match[1];
      const nextVal = remaining.get(key) || '';
      remaining.delete(key);
      nextLines.push(`${key}=${formatEnvValue(nextVal)}`);
    } else {
      nextLines.push(line);
    }
  }

  // Remove trailing blank lines before appending new keys cleanly
  while (nextLines.length > 0 && nextLines[nextLines.length - 1] === '') {
    nextLines.pop();
  }

  for (const [key, val] of remaining.entries()) {
    nextLines.push(`${key}=${formatEnvValue(val)}`);
  }

  return nextLines.join('\n') + '\n';
}

export class WorkspaceAndConfigStore {
  readonly rootDir: string;
  readonly workspaceDir: string;
  readonly skillsDir: string;
  readonly mcpConfigPath: string;
  readonly llmConfigPath: string;
  readonly threadsPath: string;
  readonly envFilePath: string;

  constructor(rootDir?: string) {
    this.rootDir = path.resolve(rootDir || process.cwd());
    this.workspaceDir = path.resolve(process.env.WORKSPACE_DIR || path.join(this.rootDir, 'workspace'));
    this.skillsDir = path.resolve(process.env.SKILLS_DIR || path.join(this.rootDir, 'Skills'));
    this.mcpConfigPath = path.resolve(process.env.MCP_CONFIG_PATH || path.join(this.rootDir, 'MCPs', 'servers.json'));
    this.llmConfigPath = path.resolve(process.env.LLM_CONFIG_PATH || path.join(this.rootDir, 'data', 'llm_config.json'));
    this.threadsPath = path.resolve(path.join(this.rootDir, 'data', 'threads.json'));
    this.envFilePath = path.resolve(path.join(this.rootDir, '.env'));

    fs.mkdirSync(this.workspaceDir, { recursive: true });
    fs.mkdirSync(this.skillsDir, { recursive: true });
    fs.mkdirSync(path.dirname(this.llmConfigPath), { recursive: true });
    fs.mkdirSync(path.dirname(this.mcpConfigPath), { recursive: true });
  }

  // ---------------------------------------------------------------------------
  // Workspace Filesystem (confined strictly inside ./workspace)
  // ---------------------------------------------------------------------------
  resolveWorkspacePath(relPath: string, allowRoot = false): string {
    const clean = (relPath || '').trim().replace(/^\/workspace\/?/, '').replace(/^\/+/, '');
    if (!clean || clean === '.') {
      if (allowRoot) return this.workspaceDir;
      throw new Error('A file or folder path inside /workspace is required.');
    }
    if (clean.includes('\0') || clean.split('/').includes('..')) {
      throw new Error('Path traversal outside /workspace is not allowed.');
    }
    const target = path.resolve(this.workspaceDir, clean);
    if (target !== this.workspaceDir && !target.startsWith(this.workspaceDir + path.sep)) {
      throw new Error('Resolved path escapes /workspace.');
    }
    return target;
  }

  listWorkspaceTree(dir = this.workspaceDir, relPrefix = ''): WorkspaceNode[] {
    if (!fs.existsSync(dir)) return [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    const nodes: WorkspaceNode[] = [];
    for (const entry of entries) {
      if (entry.name === '.git' || entry.name === '__pycache__') continue;
      const fullPath = path.join(dir, entry.name);
      const relPath = relPrefix ? `${relPrefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        nodes.push({
          name: entry.name,
          path: relPath,
          type: 'folder',
          children: this.listWorkspaceTree(fullPath, relPath),
        });
      } else if (entry.isFile()) {
        const stat = fs.statSync(fullPath);
        nodes.push({
          name: entry.name,
          path: relPath,
          type: 'file',
          size: stat.size,
          modifiedAt: stat.mtime.toISOString(),
        });
      }
    }
    return nodes.sort((a, b) => {
      if (a.type !== b.type) return a.type === 'folder' ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
  }

  readWorkspaceFile(relPath: string): { path: string; content: string; size: number; modifiedAt: string } {
    const full = this.resolveWorkspacePath(relPath);
    if (!fs.existsSync(full)) throw new Error(`File not found: ${relPath}`);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) throw new Error(`Path is a folder, not a file: ${relPath}`);
    const content = fs.readFileSync(full, 'utf-8');
    return {
      path: path.relative(this.workspaceDir, full).replace(/\\/g, '/'),
      content,
      size: stat.size,
      modifiedAt: stat.mtime.toISOString(),
    };
  }

  writeWorkspaceFile(relPath: string, content: string): { path: string; size: number; modifiedAt: string } {
    const full = this.resolveWorkspacePath(relPath);
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, content, 'utf-8');
    const stat = fs.statSync(full);
    return {
      path: path.relative(this.workspaceDir, full).replace(/\\/g, '/'),
      size: stat.size,
      modifiedAt: stat.mtime.toISOString(),
    };
  }

  createWorkspaceFolder(relPath: string): { path: string } {
    const full = this.resolveWorkspacePath(relPath);
    fs.mkdirSync(full, { recursive: true });
    return { path: path.relative(this.workspaceDir, full).replace(/\\/g, '/') };
  }

  deleteWorkspaceEntry(relPath: string): void {
    const full = this.resolveWorkspacePath(relPath);
    if (!fs.existsSync(full)) throw new Error(`Entry not found: ${relPath}`);
    fs.rmSync(full, { recursive: true, force: true });
  }

  // ---------------------------------------------------------------------------
  // LLM Config (Main Agent & Sub-Agent)
  // ---------------------------------------------------------------------------
  syncEnvFromFile(): void {
    if (!fs.existsSync(this.envFilePath)) return;
    try {
      const content = fs.readFileSync(this.envFilePath, 'utf-8');
      for (const rawLine of content.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#')) continue;
        const eqIdx = line.indexOf('=');
        if (eqIdx <= 0) continue;
        const key = line.slice(0, eqIdx).trim();
        if (!/^[A-Z][A-Z0-9_]*$/.test(key)) continue;
        let val = line.slice(eqIdx + 1).trim();
        if (
          (val.startsWith('"') && val.endsWith('"')) ||
          (val.startsWith("'") && val.endsWith("'"))
        ) {
          val = val.slice(1, -1);
        }
        if (val) {
          process.env[key] = val;
        }
      }
    } catch {
      // ignore read errors
    }
  }

  getProviderAvailability(
    profileSecrets?: Record<string, string>,
  ): Record<
    string,
    {
      base_url: string;
      env_keys: string[];
      primary_env_var: string;
      api_key_configured: boolean;
      masked_key: string | null;
      requires_api_key: boolean;
      storage?: 'profile_vault' | 'env';
    }
  > {
    if (!profileSecrets) {
      this.syncEnvFromFile();
    }
    const sourceEnv: Record<string, string | undefined> = profileSecrets || process.env;
    const storage: 'profile_vault' | 'env' = profileSecrets ? 'profile_vault' : 'env';
    const out: Record<
      string,
      {
        base_url: string;
        env_keys: string[];
        primary_env_var: string;
        api_key_configured: boolean;
        masked_key: string | null;
        requires_api_key: boolean;
        storage?: 'profile_vault' | 'env';
      }
    > = {};
    for (const [key, info] of Object.entries(PROVIDER_DEFAULTS)) {
      let foundValue = '';
      for (const ev of info.envVars) {
        const val = (sourceEnv[ev] || '').trim();
        if (val) {
          foundValue = val;
          break;
        }
      }
      const configured = !info.requiresKey || Boolean(foundValue);
      out[key] = {
        base_url: info.baseUrl,
        env_keys: info.envVars,
        primary_env_var: info.envVars[0] || 'PRIMARY_API_KEY',
        api_key_configured: configured,
        masked_key: maskEnvSecret(foundValue),
        requires_api_key: info.requiresKey,
        storage,
      };
    }
    return out;
  }

  getEnvVarForProvider(provider: string, role: 'primary' | 'worker'): string {
    const normalized = normalizeProviderId(provider);
    if (normalized === 'custom') {
      return role === 'worker' ? 'WORKER_API_KEY' : 'PRIMARY_API_KEY';
    }
    return PROVIDER_DEFAULTS[normalized]?.envVars[0] || (role === 'worker' ? 'WORKER_API_KEY' : 'PRIMARY_API_KEY');
  }

  saveEnvUpdates(updates: Record<string, string>): string[] {
    const cleanUpdates: Record<string, string> = {};
    for (const [k, v] of Object.entries(updates)) {
      const cleanKey = k.trim();
      const cleanVal = (v || '').trim();
      if (!/^[A-Z][A-Z0-9_]*$/.test(cleanKey) || !cleanVal) continue;
      cleanUpdates[cleanKey] = cleanVal;
    }
    const keys = Object.keys(cleanUpdates);
    if (keys.length === 0) return [];

    const existing = fs.existsSync(this.envFilePath)
      ? fs.readFileSync(this.envFilePath, 'utf-8')
      : '';
    const nextContent = upsertEnvContent(existing, cleanUpdates);
    fs.writeFileSync(this.envFilePath, nextContent, { encoding: 'utf-8', mode: 0o600 });

    for (const [k, v] of Object.entries(cleanUpdates)) {
      process.env[k] = v;
    }
    return keys;
  }

  getLLMConfig(): PersistedLLMConfig {
    const defaultPrimaryProvider = normalizeProviderId(process.env.PRIMARY_PROVIDER || 'river_ai');
    const defaultWorkerProvider = normalizeProviderId(process.env.WORKER_PROVIDER || defaultPrimaryProvider);
    const defaults: PersistedLLMConfig = {
      primary: {
        provider: defaultPrimaryProvider,
        model: process.env.PRIMARY_MODEL || 'deepseek-ai/DeepSeek-V4.1-Flash',
        base_url:
          process.env.PRIMARY_BASE_URL ||
          PROVIDER_DEFAULTS[defaultPrimaryProvider]?.baseUrl ||
          'https://api.river.ai/v1',
        temperature: Number(process.env.PRIMARY_TEMPERATURE || '0.2'),
        max_tokens: Number(process.env.PRIMARY_MAX_TOKENS || '16384'),
        reasoning_effort: process.env.PRIMARY_REASONING_EFFORT || 'off',
      },
      worker: {
        provider: defaultWorkerProvider,
        model: process.env.WORKER_MODEL || 'deepseek-ai/DeepSeek-V4.1-Flash',
        base_url:
          process.env.WORKER_BASE_URL ||
          PROVIDER_DEFAULTS[defaultWorkerProvider]?.baseUrl ||
          'https://api.river.ai/v1',
        temperature: Number(process.env.WORKER_TEMPERATURE || '0.1'),
        max_tokens: Number(process.env.WORKER_MAX_TOKENS || '8192'),
        reasoning_effort: process.env.WORKER_REASONING_EFFORT || 'off',
      },
      max_turns: Number(process.env.MAX_TURNS || '30'),
    };

    if (!fs.existsSync(this.llmConfigPath)) {
      return defaults;
    }

    try {
      const parsed = JSON.parse(fs.readFileSync(this.llmConfigPath, 'utf-8')) as Partial<PersistedLLMConfig>;
      return {
        primary: normalizeRoleConfig(parsed.primary, defaults.primary),
        worker: normalizeRoleConfig(parsed.worker, defaults.worker),
        max_turns:
          typeof parsed.max_turns === 'number' && Number.isFinite(parsed.max_turns)
            ? Math.max(1, Math.min(100, Math.round(parsed.max_turns)))
            : defaults.max_turns,
      };
    } catch {
      return defaults;
    }
  }

  saveLLMConfig(next: Partial<PersistedLLMConfig>): PersistedLLMConfig {
    const current = this.getLLMConfig();
    const merged: PersistedLLMConfig = {
      primary: normalizeRoleConfig(next.primary, current.primary),
      worker: normalizeRoleConfig(next.worker, current.worker),
      max_turns:
        typeof next.max_turns === 'number' && Number.isFinite(next.max_turns)
          ? Math.max(1, Math.min(100, Math.round(next.max_turns)))
          : current.max_turns,
    };
    fs.writeFileSync(this.llmConfigPath, JSON.stringify(merged, null, 2) + '\n', 'utf-8');
    return merged;
  }

  // ---------------------------------------------------------------------------
  // Skills (./Skills/*.md)
  // ---------------------------------------------------------------------------
  listSkills(): SkillDocument[] {
    if (!fs.existsSync(this.skillsDir)) return [];
    return fs
      .readdirSync(this.skillsDir)
      .filter((f) => f.endsWith('.md'))
      .sort()
      .map((filename) => {
        const content = fs.readFileSync(path.join(this.skillsDir, filename), 'utf-8');
        const nameMatch = content.match(/^name:\s*(.+)$/m);
        const descMatch = content.match(/^description:\s*(.+)$/m);
        return {
          name: (nameMatch ? nameMatch[1].trim().replace(/^["']|["']$/g, '') : filename.replace(/\.md$/, '')),
          description: descMatch
            ? descMatch[1].trim().replace(/^["']|["']$/g, '')
            : 'Custom agent skill definition',
          filename,
          content,
        };
      });
  }

  saveSkill(name: string, content: string): SkillDocument {
    const safeName = name.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
    if (!safeName) throw new Error('Invalid skill name.');
    const filename = `${safeName}.md`;
    const full = path.join(this.skillsDir, filename);
    fs.writeFileSync(full, content, 'utf-8');
    const descMatch = content.match(/^description:\s*(.+)$/m);
    return {
      name: safeName,
      description: descMatch ? descMatch[1].trim() : 'Custom agent skill definition',
      filename,
      content,
    };
  }

  deleteSkill(name: string): void {
    const safeName = name.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
    const full = path.join(this.skillsDir, `${safeName}.md`);
    if (fs.existsSync(full)) fs.unlinkSync(full);
  }

  // ---------------------------------------------------------------------------
  // Pure-Python MCP Servers (./MCPs/servers.json)
  // ---------------------------------------------------------------------------
  listMCPServers(): MCPServerItem[] {
    if (!fs.existsSync(this.mcpConfigPath)) return [];
    try {
      const parsed = JSON.parse(fs.readFileSync(this.mcpConfigPath, 'utf-8')) as {
        servers?: MCPServerItem[];
      };
      return Array.isArray(parsed.servers) ? parsed.servers : [];
    } catch {
      return [];
    }
  }

  saveMCPServers(servers: MCPServerItem[]): MCPServerItem[] {
    const clean = servers
      .filter((s) => s && typeof s.name === 'string' && s.name.trim() && typeof s.url === 'string' && s.url.trim())
      .map((s) => ({
        name: s.name.trim(),
        url: s.url.trim(),
        transport: (s.transport === 'sse' ? 'sse' : 'streamable_http') as 'streamable_http' | 'sse',
        enabled: s.enabled !== false,
        timeout_seconds: Math.max(2, Math.min(180, Number(s.timeout_seconds) || 30)),
      }));
    fs.writeFileSync(this.mcpConfigPath, JSON.stringify({ servers: clean }, null, 2) + '\n', 'utf-8');
    return clean;
  }

  // ---------------------------------------------------------------------------
  // Chat Threads (./data/threads.json)
  // ---------------------------------------------------------------------------
  listThreads(): ChatThreadRecord[] {
    if (!fs.existsSync(this.threadsPath)) return [];
    try {
      const parsed = JSON.parse(fs.readFileSync(this.threadsPath, 'utf-8')) as ChatThreadRecord[];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  saveThreads(threads: ChatThreadRecord[]): void {
    fs.writeFileSync(this.threadsPath, JSON.stringify(threads, null, 2) + '\n', 'utf-8');
  }
}
