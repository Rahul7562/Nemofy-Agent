import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';

export type MediaModality = 'image' | 'video' | 'audio';
export type ExecutionAccessMode = 'default' | 'full_access';

export const MIN_MAX_TURNS = 1;
export const MAX_MAX_TURNS = 180;

export function normalizeAccessMode(raw: unknown): ExecutionAccessMode {
  return raw === 'full_access' ? 'full_access' : 'default';
}

export function validateMaxTurns(
  raw: unknown,
): { ok: true; value: number } | { ok: false; error: string } {
  if (typeof raw === 'boolean' || raw === null || raw === undefined) {
    return {
      ok: false,
      error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
    };
  }
  let parsed: number;
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!/^-?\d+$/.test(trimmed)) {
      return {
        ok: false,
        error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
      };
    }
    parsed = Number(trimmed);
  } else if (typeof raw === 'number') {
    parsed = raw;
  } else {
    return {
      ok: false,
      error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
    };
  }

  if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) {
    return {
      ok: false,
      error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
    };
  }
  if (parsed < MIN_MAX_TURNS || parsed > MAX_MAX_TURNS) {
    return {
      ok: false,
      error: `Max execution turns must be between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS} (received ${parsed}).`,
    };
  }
  return { ok: true, value: parsed };
}

export interface RoleLLMConfig {
  provider: string;
  model: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
  reasoning_effort: string;
  context_window?: number | null;
  max_output_tokens?: number | null;
}

export interface ContextCategorySnapshot {
  key: string;
  label: string;
  description: string;
  tokens: number;
  percent: number;
  share_of_used: number;
}

export interface ContextUsageSnapshot {
  model: string;
  provider: string;
  context_window: number | null;
  context_window_source: string;
  max_output_tokens?: number | null;
  max_output_tokens_source?: string;
  configured_max_tokens?: number | null;
  used_tokens: number;
  free_tokens: number | null;
  usage_percent: number | null;
  auto_compact_threshold_percent: number;
  classification: {
    level: 'optimal' | 'moderate' | 'high' | 'critical' | 'unknown';
    label: string;
    description: string;
  };
  categories: ContextCategorySnapshot[];
  compacted_count: number;
  last_compacted_at: string | null;
  tokens_saved_last_compaction: number;
  calibrated_from_provider?: boolean;
}

export interface AccessoryLLMConfig extends RoleLLMConfig {
  enabled: boolean;
}

export interface AccessoriesConfig {
  image: AccessoryLLMConfig;
  video: AccessoryLLMConfig;
  audio: AccessoryLLMConfig;
}

export interface PersistedLLMConfig {
  primary: RoleLLMConfig;
  worker: RoleLLMConfig;
  accessories: AccessoriesConfig;
  max_turns: number;
  access_mode: ExecutionAccessMode;
}

export interface ChatAttachmentRecord {
  id: string;
  name: string;
  kind: MediaModality;
  mimeType: string;
  size: number;
  url: string;
  storageKey: string;
}

export interface WorkspaceNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  size?: number;
  modifiedAt?: string;
  children?: WorkspaceNode[];
  hasChildren?: boolean;
  childCount?: number;
  truncated?: boolean;
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
  attachments?: ChatAttachmentRecord[];
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
  contextUsage?: ContextUsageSnapshot | null;
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

function normalizeRoleConfig(
  raw: Partial<RoleLLMConfig> | undefined,
  fallback: RoleLLMConfig,
  options: { allowEmptyModel?: boolean } = {},
): RoleLLMConfig {
  const provider = normalizeProviderId(raw?.provider || fallback.provider);
  const presetBase = PROVIDER_DEFAULTS[provider]?.baseUrl || '';
  const rawBase = typeof raw?.base_url === 'string' ? raw.base_url.trim().replace(/\/+$/, '') : '';
  const base_url = rawBase || presetBase || fallback.base_url;
  const hasExplicitModel = typeof raw?.model === 'string';
  const trimmedModel = hasExplicitModel ? raw!.model!.trim() : '';
  const model =
    trimmedModel ||
    (options.allowEmptyModel && hasExplicitModel ? '' : fallback.model);
  const rawTemp = Number(raw?.temperature);
  const temperature = Number.isFinite(rawTemp) && rawTemp >= 0 && rawTemp <= 2 ? rawTemp : fallback.temperature;
  const rawTokens = Number(raw?.max_tokens);
  const max_tokens =
    Number.isFinite(rawTokens) && rawTokens >= 64 && rawTokens <= 20_000_000
      ? Math.round(rawTokens)
      : fallback.max_tokens;
  const reasoning_effort =
    typeof raw?.reasoning_effort === 'string' && raw.reasoning_effort.trim()
      ? raw.reasoning_effort.trim().toLowerCase()
      : fallback.reasoning_effort;

  let context_window: number | null | undefined;
  if (raw && 'context_window' in raw) {
    const rawCtx = Number(raw.context_window);
    context_window = Number.isFinite(rawCtx) && rawCtx >= 512 ? Math.round(rawCtx) : null;
  } else if (
    typeof fallback.context_window === 'number' &&
    Number.isFinite(fallback.context_window) &&
    fallback.context_window >= 512 &&
    model === fallback.model &&
    provider === fallback.provider
  ) {
    context_window = Math.round(fallback.context_window);
  }

  let max_output_tokens: number | null | undefined;
  if (raw && 'max_output_tokens' in raw) {
    const rawOut = Number(raw.max_output_tokens);
    max_output_tokens = Number.isFinite(rawOut) && rawOut >= 256 ? Math.round(rawOut) : null;
  } else if (
    typeof fallback.max_output_tokens === 'number' &&
    Number.isFinite(fallback.max_output_tokens) &&
    fallback.max_output_tokens >= 256 &&
    model === fallback.model &&
    provider === fallback.provider
  ) {
    max_output_tokens = Math.round(fallback.max_output_tokens);
  }

  return {
    provider,
    model,
    base_url,
    temperature,
    max_tokens,
    reasoning_effort,
    ...(context_window ? { context_window } : {}),
    ...(max_output_tokens ? { max_output_tokens } : {}),
  };
}

export const DEFAULT_ACCESSORY_MODEL = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free';

export function getDefaultAccessoriesConfig(): AccessoriesConfig {
  const base: AccessoryLLMConfig = {
    enabled: true,
    provider: 'openrouter',
    model: DEFAULT_ACCESSORY_MODEL,
    base_url: 'https://openrouter.ai/api/v1',
    temperature: 0.2,
    max_tokens: 4096,
    reasoning_effort: 'low',
  };
  return {
    image: { ...base },
    video: { ...base },
    audio: { ...base },
  };
}

function normalizeAccessoryConfig(
  raw: Partial<AccessoryLLMConfig> | undefined,
  fallback: AccessoryLLMConfig,
): AccessoryLLMConfig {
  const roleNormalized = normalizeRoleConfig(raw, fallback);
  return {
    ...roleNormalized,
    enabled: typeof raw?.enabled === 'boolean' ? raw.enabled : fallback.enabled,
  };
}

const ALLOWED_MEDIA_MIME: Record<string, { kind: MediaModality; ext: string }> = {
  'image/png': { kind: 'image', ext: '.png' },
  'image/jpeg': { kind: 'image', ext: '.jpg' },
  'image/jpg': { kind: 'image', ext: '.jpg' },
  'image/webp': { kind: 'image', ext: '.webp' },
  'image/gif': { kind: 'image', ext: '.gif' },
  'image/bmp': { kind: 'image', ext: '.bmp' },
  'image/avif': { kind: 'image', ext: '.avif' },
  'audio/wav': { kind: 'audio', ext: '.wav' },
  'audio/x-wav': { kind: 'audio', ext: '.wav' },
  'audio/wave': { kind: 'audio', ext: '.wav' },
  'audio/mpeg': { kind: 'audio', ext: '.mp3' },
  'audio/mp3': { kind: 'audio', ext: '.mp3' },
  'audio/ogg': { kind: 'audio', ext: '.ogg' },
  'audio/opus': { kind: 'audio', ext: '.ogg' },
  'audio/flac': { kind: 'audio', ext: '.flac' },
  'audio/x-flac': { kind: 'audio', ext: '.flac' },
  'audio/mp4': { kind: 'audio', ext: '.m4a' },
  'audio/m4a': { kind: 'audio', ext: '.m4a' },
  'audio/x-m4a': { kind: 'audio', ext: '.m4a' },
  'audio/aac': { kind: 'audio', ext: '.aac' },
  'audio/webm': { kind: 'audio', ext: '.webm' },
  'video/mp4': { kind: 'video', ext: '.mp4' },
  'video/webm': { kind: 'video', ext: '.webm' },
  'video/quicktime': { kind: 'video', ext: '.mov' },
  'video/x-matroska': { kind: 'video', ext: '.mkv' },
  'video/ogg': { kind: 'video', ext: '.ogv' },
};

const EXT_TO_MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.avif': 'image/avif',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.ogg': 'audio/ogg',
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.aac': 'audio/aac',
  '.webm': 'video/webm',
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.ogv': 'video/ogg',
};

const MAX_UPLOAD_BYTES: Record<MediaModality, number> = {
  image: 15 * 1024 * 1024,
  audio: 25 * 1024 * 1024,
  video: 40 * 1024 * 1024,
};

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
  readonly uploadsDir: string;
  readonly envFilePath: string;

  constructor(rootDir?: string) {
    this.rootDir = path.resolve(rootDir || process.cwd());
    this.workspaceDir = path.resolve(process.env.WORKSPACE_DIR || path.join(this.rootDir, 'workspace'));
    this.skillsDir = path.resolve(process.env.SKILLS_DIR || path.join(this.rootDir, 'Skills'));
    this.mcpConfigPath = path.resolve(process.env.MCP_CONFIG_PATH || path.join(this.rootDir, 'MCPs', 'servers.json'));
    this.llmConfigPath = path.resolve(process.env.LLM_CONFIG_PATH || path.join(this.rootDir, 'data', 'llm_config.json'));
    this.threadsPath = path.resolve(path.join(this.rootDir, 'data', 'threads.json'));
    this.uploadsDir = path.resolve(process.env.NEMOFY_UPLOADS_DIR || path.join(this.rootDir, 'data', 'uploads'));
    this.envFilePath = path.resolve(path.join(this.rootDir, '.env'));

    fs.mkdirSync(this.workspaceDir, { recursive: true });
    fs.mkdirSync(this.skillsDir, { recursive: true });
    fs.mkdirSync(path.dirname(this.llmConfigPath), { recursive: true });
    fs.mkdirSync(path.dirname(this.mcpConfigPath), { recursive: true });
    fs.mkdirSync(this.uploadsDir, { recursive: true });
  }

  // ---------------------------------------------------------------------------
  // Multimodal Media Uploads (confined strictly inside ./data/uploads)
  // ---------------------------------------------------------------------------
  saveUploadedMedia(params: {
    threadId: string;
    filename: string;
    mimeType: string;
    buffer: Buffer;
  }): ChatAttachmentRecord & { localPath: string } {
    const cleanMime = (params.mimeType || '').toLowerCase().split(';')[0].trim();
    const extFromName = path.extname(params.filename || '').toLowerCase();
    const resolvedMime =
      ALLOWED_MEDIA_MIME[cleanMime]
        ? cleanMime
        : extFromName && EXT_TO_MIME[extFromName] && ALLOWED_MEDIA_MIME[EXT_TO_MIME[extFromName]]
          ? EXT_TO_MIME[extFromName]
          : '';
    const mediaSpec = resolvedMime ? ALLOWED_MEDIA_MIME[resolvedMime] : undefined;
    if (!mediaSpec) {
      throw new Error(
        `Unsupported media type '${cleanMime || extFromName || 'unknown'}'. Allowed formats: PNG, JPEG, WebP, GIF, MP3, WAV, OGG, FLAC, M4A, WebM, MP4, MOV.`,
      );
    }
    if (!params.buffer || params.buffer.length === 0) {
      throw new Error('Uploaded media file is empty.');
    }
    const maxBytes = MAX_UPLOAD_BYTES[mediaSpec.kind];
    if (params.buffer.length > maxBytes) {
      throw new Error(
        `${mediaSpec.kind.toUpperCase()} file exceeds the ${Math.round(maxBytes / (1024 * 1024))} MB limit.`,
      );
    }

    const safeThread = (params.threadId || 'draft')
      .replace(/[^a-zA-Z0-9_-]/g, '')
      .slice(0, 80) || 'draft';
    const safeDisplayName =
      path.basename(params.filename || `upload${mediaSpec.ext}`).replace(/[^\w.\- ()]/g, '_').slice(0, 120) ||
      `upload${mediaSpec.ext}`;
    const id = `att-${crypto.randomBytes(6).toString('hex')}`;
    const storedFilename = `${id}${mediaSpec.ext}`;
    const threadDir = path.resolve(this.uploadsDir, safeThread);
    if (!threadDir.startsWith(this.uploadsDir + path.sep)) {
      throw new Error('Invalid upload directory.');
    }
    fs.mkdirSync(threadDir, { recursive: true });
    const fullPath = path.resolve(threadDir, storedFilename);
    if (!fullPath.startsWith(threadDir + path.sep)) {
      throw new Error('Invalid upload target path.');
    }
    fs.writeFileSync(fullPath, params.buffer);

    const storageKey = `${safeThread}/${storedFilename}`;
    return {
      id,
      name: safeDisplayName,
      kind: mediaSpec.kind,
      mimeType: resolvedMime,
      size: params.buffer.length,
      url: `/api/media/${encodeURIComponent(safeThread)}/${encodeURIComponent(storedFilename)}`,
      storageKey,
      localPath: fullPath,
    };
  }

  resolveUploadedMediaPath(
    threadId: string,
    filename: string,
  ): { fullPath: string; mimeType: string } | null {
    const safeThread = (threadId || '').trim();
    const safeFile = (filename || '').trim();
    if (
      !/^[a-zA-Z0-9_-]{1,80}$/.test(safeThread) ||
      !/^[a-zA-Z0-9_.-]{1,120}$/.test(safeFile) ||
      safeFile.includes('..')
    ) {
      return null;
    }
    const fullPath = path.resolve(this.uploadsDir, safeThread, safeFile);
    if (!fullPath.startsWith(this.uploadsDir + path.sep)) {
      return null;
    }
    if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
      return null;
    }
    const ext = path.extname(safeFile).toLowerCase();
    const mimeType = EXT_TO_MIME[ext];
    if (!mimeType) return null;
    return { fullPath, mimeType };
  }

  resolveAttachmentLocalPath(storageKey: string): string | null {
    const parts = (storageKey || '').split('/');
    if (parts.length !== 2) return null;
    const resolved = this.resolveUploadedMediaPath(parts[0], parts[1]);
    return resolved ? resolved.fullPath : null;
  }

  // ---------------------------------------------------------------------------
  // Workspace Filesystem (confined strictly inside ./workspace)
  // ---------------------------------------------------------------------------
  private getWorkspaceRealDir(): string {
    fs.mkdirSync(this.workspaceDir, { recursive: true });
    return fs.realpathSync(this.workspaceDir);
  }

  resolveWorkspacePath(relPath: string, allowRoot = false): string {
    const wsReal = this.getWorkspaceRealDir();
    const clean = (relPath || '').trim().replace(/^\/workspace\/?/, '').replace(/^\/+/, '');
    if (!clean || clean === '.') {
      if (allowRoot) return wsReal;
      throw new Error('A file or folder path inside /workspace is required.');
    }
    const segments = clean.split(/[/\\]+/).filter(Boolean);
    if (clean.includes('\0') || segments.includes('..')) {
      throw new Error('Path traversal outside /workspace is not allowed.');
    }
    const target = path.resolve(wsReal, ...segments);
    if (target !== wsReal && !target.startsWith(wsReal + path.sep)) {
      throw new Error('Resolved path escapes /workspace.');
    }

    // Walk every path segment to reject symbolic links escaping /workspace
    let current = wsReal;
    for (const seg of segments) {
      current = path.join(current, seg);
      if (fs.existsSync(current)) {
        const lstat = fs.lstatSync(current);
        if (lstat.isSymbolicLink()) {
          const realTarget = fs.realpathSync(current);
          if (realTarget !== wsReal && !realTarget.startsWith(wsReal + path.sep)) {
            throw new Error('Symbolic link target escapes /workspace.');
          }
        }
      } else {
        // Also catch broken symlinks (existsSync is false, lstatSync succeeds)
        try {
          const lstat = fs.lstatSync(current);
          if (lstat.isSymbolicLink()) {
            throw new Error('Symbolic link target escapes /workspace.');
          }
        } catch (err) {
          if (err instanceof Error && err.message.includes('Symbolic link')) {
            throw err;
          }
        }
        break;
      }
    }

    if (fs.existsSync(target)) {
      const realResolved = fs.realpathSync(target);
      if (realResolved !== wsReal && !realResolved.startsWith(wsReal + path.sep)) {
        throw new Error('Resolved path escapes /workspace.');
      }
      return realResolved;
    }

    return target;
  }

  private static readonly IGNORED_TREE_DIRS = new Set([
    '.git',
    '__pycache__',
    '.cache',
    'node_modules',
    '.venv',
    'venv',
    '.pytest_cache',
    '.mypy_cache',
  ]);

  private static readonly MAX_READABLE_FILE_BYTES = 1_500_000;

  listWorkspaceTree(
    dir = this.workspaceDir,
    relPrefix = '',
    options: { maxDepth?: number; maxEntriesPerDir?: number; maxTotalNodes?: number } = {},
  ): WorkspaceNode[] {
    const maxDepth = options.maxDepth ?? 2;
    const maxEntriesPerDir = options.maxEntriesPerDir ?? 250;
    const budget = { remaining: options.maxTotalNodes ?? 400 };

    const walk = (currentDir: string, prefix: string, depth: number): WorkspaceNode[] => {
      if (!fs.existsSync(currentDir) || budget.remaining <= 0) return [];
      let rawEntries: fs.Dirent[];
      try {
        rawEntries = fs.readdirSync(currentDir, { withFileTypes: true });
      } catch {
        return [];
      }

      const filtered = rawEntries.filter(
        (e) => !WorkspaceAndConfigStore.IGNORED_TREE_DIRS.has(e.name),
      );
      filtered.sort((a, b) => {
        const aDir = a.isDirectory();
        const bDir = b.isDirectory();
        if (aDir !== bDir) return aDir ? -1 : 1;
        return a.name.localeCompare(b.name);
      });

      const nodes: WorkspaceNode[] = [];
      for (const entry of filtered.slice(0, maxEntriesPerDir)) {
        if (budget.remaining <= 0) break;
        const fullPath = path.join(currentDir, entry.name);
        const relPath = prefix ? `${prefix}/${entry.name}` : entry.name;

        if (entry.isDirectory()) {
          budget.remaining -= 1;
          let childCount = 0;
          try {
            childCount = fs
              .readdirSync(fullPath)
              .filter((n) => !WorkspaceAndConfigStore.IGNORED_TREE_DIRS.has(n)).length;
          } catch {
            childCount = 0;
          }

          if (depth < maxDepth && budget.remaining > 0 && childCount > 0) {
            const children = walk(fullPath, relPath, depth + 1);
            nodes.push({
              name: entry.name,
              path: relPath,
              type: 'folder',
              childCount,
              hasChildren: childCount > 0,
              truncated: childCount > children.length,
              children,
            });
          } else {
            nodes.push({
              name: entry.name,
              path: relPath,
              type: 'folder',
              childCount,
              hasChildren: childCount > 0,
              truncated: childCount > 0,
            });
          }
        } else if (entry.isFile()) {
          try {
            const stat = fs.statSync(fullPath);
            budget.remaining -= 1;
            nodes.push({
              name: entry.name,
              path: relPath,
              type: 'file',
              size: stat.size,
              modifiedAt: stat.mtime.toISOString(),
            });
          } catch {
            // Skip unreadable or vanished entries safely
          }
        }
      }
      return nodes;
    };

    return walk(dir, relPrefix, 0);
  }

  listWorkspaceDirectory(
    relDir = '',
    options: { maxEntries?: number } = {},
  ): { entries: WorkspaceNode[]; totalCount: number; truncated: boolean } {
    const clean = (relDir || '').trim().replace(/^\/workspace\/?/, '').replace(/^\/+/, '');
    const targetDir = clean ? this.resolveWorkspacePath(clean) : this.workspaceDir;
    if (!fs.existsSync(targetDir)) {
      throw new Error(`Directory not found: ${relDir}`);
    }
    const stat = fs.statSync(targetDir);
    if (!stat.isDirectory()) {
      throw new Error(`Path is not a directory: ${relDir}`);
    }
    const maxEntries = Math.max(10, Math.min(500, options.maxEntries ?? 250));
    const rawEntries = fs
      .readdirSync(targetDir, { withFileTypes: true })
      .filter((e) => !WorkspaceAndConfigStore.IGNORED_TREE_DIRS.has(e.name));

    rawEntries.sort((a, b) => {
      const aDir = a.isDirectory();
      const bDir = b.isDirectory();
      if (aDir !== bDir) return aDir ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    const entries: WorkspaceNode[] = [];
    for (const entry of rawEntries.slice(0, maxEntries)) {
      const fullPath = path.join(targetDir, entry.name);
      const relPath = clean ? `${clean}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        let childCount = 0;
        try {
          childCount = fs
            .readdirSync(fullPath)
            .filter((n) => !WorkspaceAndConfigStore.IGNORED_TREE_DIRS.has(n)).length;
        } catch {
          childCount = 0;
        }
        entries.push({
          name: entry.name,
          path: relPath,
          type: 'folder',
          childCount,
          hasChildren: childCount > 0,
          truncated: childCount > 0,
        });
      } else if (entry.isFile()) {
        try {
          const fstat = fs.statSync(fullPath);
          entries.push({
            name: entry.name,
            path: relPath,
            type: 'file',
            size: fstat.size,
            modifiedAt: fstat.mtime.toISOString(),
          });
        } catch {
          // ignore
        }
      }
    }

    return {
      entries,
      totalCount: rawEntries.length,
      truncated: rawEntries.length > entries.length,
    };
  }

  readWorkspaceFile(relPath: string): { path: string; content: string; size: number; modifiedAt: string } {
    const full = this.resolveWorkspacePath(relPath);
    if (!fs.existsSync(full)) throw new Error(`File not found: ${relPath}`);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) throw new Error(`Path is a folder, not a file: ${relPath}`);
    if (stat.size > WorkspaceAndConfigStore.MAX_READABLE_FILE_BYTES) {
      const mb = (stat.size / (1024 * 1024)).toFixed(2);
      throw new Error(
        `File is too large to open in the inline editor (${mb} MB, limit 1.5 MB). Ask Nemo to inspect or summarize it instead.`,
      );
    }
    const rawBuf = fs.readFileSync(full);
    const probeLen = Math.min(rawBuf.length, 4096);
    for (let i = 0; i < probeLen; i++) {
      if (rawBuf[i] === 0) {
        throw new Error(
          `Binary file cannot be edited as plain text (${path.basename(full)}).`,
        );
      }
    }
    const content = rawBuf.toString('utf-8');
    return {
      path: path.relative(this.workspaceDir, full).replace(/\\/g, '/'),
      content,
      size: stat.size,
      modifiedAt: stat.mtime.toISOString(),
    };
  }

  writeWorkspaceFile(
    relPath: string,
    content: string,
    options: { createOnly?: boolean } = {},
  ): { path: string; size: number; modifiedAt: string } {
    const full = this.resolveWorkspacePath(relPath);
    const wsReal = this.getWorkspaceRealDir();
    fs.mkdirSync(path.dirname(full), { recursive: true });
    // Re-verify parent realpath after mkdirSync before writing
    const parentReal = fs.realpathSync(path.dirname(full));
    if (parentReal !== wsReal && !parentReal.startsWith(wsReal + path.sep)) {
      throw new Error('Resolved parent path escapes /workspace.');
    }
    if (options.createOnly) {
      try {
        fs.writeFileSync(full, content, { encoding: 'utf-8', flag: 'wx' });
      } catch (err) {
        if ((err as NodeJS.ErrnoException)?.code === 'EEXIST') {
          const conflictErr = new Error(`File already exists: /workspace/${relPath}`) as NodeJS.ErrnoException;
          conflictErr.code = 'EEXIST';
          throw conflictErr;
        }
        throw err;
      }
    } else {
      fs.writeFileSync(full, content, 'utf-8');
    }
    const stat = fs.statSync(full);
    return {
      path: path.relative(wsReal, full).replace(/\\/g, '/'),
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
      accessories: getDefaultAccessoriesConfig(),
      max_turns: Math.max(
        MIN_MAX_TURNS,
        Math.min(MAX_MAX_TURNS, Math.round(Number(process.env.MAX_TURNS || '30') || 30)),
      ),
      access_mode: normalizeAccessMode(process.env.EXECUTION_ACCESS_MODE),
    };

    if (!fs.existsSync(this.llmConfigPath)) {
      return defaults;
    }

    try {
      const parsed = JSON.parse(fs.readFileSync(this.llmConfigPath, 'utf-8')) as Partial<PersistedLLMConfig>;
      return {
        primary: normalizeRoleConfig(parsed.primary, defaults.primary, { allowEmptyModel: true }),
        worker: normalizeRoleConfig(parsed.worker, defaults.worker),
        accessories: {
          image: normalizeAccessoryConfig(parsed.accessories?.image, defaults.accessories.image),
          video: normalizeAccessoryConfig(parsed.accessories?.video, defaults.accessories.video),
          audio: normalizeAccessoryConfig(parsed.accessories?.audio, defaults.accessories.audio),
        },
        max_turns:
          typeof parsed.max_turns === 'number' && Number.isFinite(parsed.max_turns)
            ? Math.max(MIN_MAX_TURNS, Math.min(MAX_MAX_TURNS, Math.round(parsed.max_turns)))
            : defaults.max_turns,
        access_mode:
          parsed.access_mode !== undefined
            ? normalizeAccessMode(parsed.access_mode)
            : defaults.access_mode,
      };
    } catch {
      return defaults;
    }
  }

  saveLLMConfig(
    next: Partial<Omit<PersistedLLMConfig, 'accessories'>> & {
      accessories?: Partial<{
        image: Partial<AccessoryLLMConfig>;
        video: Partial<AccessoryLLMConfig>;
        audio: Partial<AccessoryLLMConfig>;
      }>;
    },
  ): PersistedLLMConfig {
    const current = this.getLLMConfig();
    const merged: PersistedLLMConfig = {
      primary: normalizeRoleConfig(next.primary, current.primary, { allowEmptyModel: true }),
      worker: normalizeRoleConfig(next.worker, current.worker),
      accessories: {
        image: normalizeAccessoryConfig(next.accessories?.image, current.accessories.image),
        video: normalizeAccessoryConfig(next.accessories?.video, current.accessories.video),
        audio: normalizeAccessoryConfig(next.accessories?.audio, current.accessories.audio),
      },
      max_turns:
        typeof next.max_turns === 'number' && Number.isFinite(next.max_turns)
          ? Math.max(MIN_MAX_TURNS, Math.min(MAX_MAX_TURNS, Math.round(next.max_turns)))
          : current.max_turns,
      access_mode:
        next.access_mode !== undefined
          ? normalizeAccessMode(next.access_mode)
          : current.access_mode,
    };
    const payload = JSON.stringify(merged, null, 2) + '\n';
    const tmpPath = `${this.llmConfigPath}.${process.pid}.${crypto.randomBytes(4).toString('hex')}.tmp`;
    fs.writeFileSync(tmpPath, payload, 'utf-8');
    fs.renameSync(tmpPath, this.llmConfigPath);
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

  private static validateMcpServerUrl(rawUrl: string): string {
    let parsed: URL;
    try {
      parsed = new URL(rawUrl);
    } catch {
      throw new Error(`Invalid MCP server URL: ${rawUrl}`);
    }
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
      throw new Error(`MCP server URL must use http:// or https://: ${rawUrl}`);
    }
    if (parsed.username || parsed.password) {
      throw new Error('MCP server URL must not contain embedded credentials.');
    }
    const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '');
    if (
      !host ||
      host === 'localhost' ||
      host.endsWith('.localhost') ||
      host.endsWith('.local') ||
      host.endsWith('.internal') ||
      host === 'metadata.google.internal'
    ) {
      throw new Error(`MCP server URL targets a restricted or local host: ${host}`);
    }
    if (net.isIPv4(host)) {
      const [a, b] = host.split('.').map(Number);
      if (
        a === 0 ||
        a === 10 ||
        a === 127 ||
        (a === 169 && b === 254) ||
        (a === 172 && b >= 16 && b <= 31) ||
        (a === 192 && b === 168) ||
        (a === 100 && b >= 64 && b <= 127) ||
        a >= 224
      ) {
        throw new Error(`MCP server URL targets a private, loopback, or metadata IP: ${host}`);
      }
    } else if (net.isIPv6(host)) {
      const first = parseInt(host.split(':')[0] || '0', 16);
      if (
        host === '::' ||
        host === '::1' ||
        host.startsWith('::ffff:') ||
        (first & 0xfe00) === 0xfc00 ||
        (first & 0xffc0) === 0xfe80 ||
        (first & 0xff00) === 0xff00
      ) {
        throw new Error(`MCP server URL targets a private, loopback, or metadata IP: ${host}`);
      }
    }
    return rawUrl;
  }

  saveMCPServers(servers: MCPServerItem[]): MCPServerItem[] {
    const clean = servers
      .filter((s) => s && typeof s.name === 'string' && s.name.trim() && typeof s.url === 'string' && s.url.trim())
      .map((s) => ({
        name: s.name.trim(),
        url: WorkspaceAndConfigStore.validateMcpServerUrl(s.url.trim()),
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
    fs.mkdirSync(path.dirname(this.threadsPath), { recursive: true });
    const tmpPath = `${this.threadsPath}.${process.pid}.${crypto.randomBytes(6).toString('hex')}.tmp`;
    fs.writeFileSync(tmpPath, JSON.stringify(threads, null, 2) + '\n', 'utf-8');
    fs.renameSync(tmpPath, this.threadsPath);
  }

  updateThreads<T>(mutator: (threads: ChatThreadRecord[]) => T): T {
    const fresh = this.listThreads();
    const result = mutator(fresh);
    this.saveThreads(fresh);
    return result;
  }
}
