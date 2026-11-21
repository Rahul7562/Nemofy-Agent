export type NavSection = 'chat' | 'filesystem' | 'settings';
export type SettingsTab = 'llm' | 'password' | 'skills-mcps' | 'appearance';
export type ThemeMode = 'light' | 'dark';

export interface AuthSession {
  authenticated: boolean;
  username?: string;
  updatedAt?: string;
  mode?: 'server' | 'standalone';
  vaultKeysLoaded?: number;
}

export interface PlanStep {
  step_id: string;
  description: string;
  status: string;
  tool_or_action: string;
  result: string;
}

export interface PendingApproval {
  tool_name?: string;
  tool_call_id?: string;
  arguments?: Record<string, unknown>;
  risk_reason?: string;
  [key: string]: unknown;
}

export interface ToolEventItem {
  id: string;
  tool: string;
  args_summary?: string;
  status: 'running' | 'ok' | 'error' | 'rejected';
  output_summary?: string;
  started_at?: string;
  finished_at?: string | null;
}

export interface FileChangeItem {
  kind: 'file' | 'dir';
  action: 'created' | 'modified' | 'deleted';
  path: string;
  tool?: string;
  timestamp?: string;
}

export interface ActivityFeedItem {
  id: string;
  phase: string;
  text: string;
  timestamp?: string;
}

export interface LiveAgentActivity {
  jobId: string;
  status: string;
  currentPhase: string;
  lastEvent: string;
  elapsedSeconds: number;
  turnCount: number;
  thinking: string | null;
  activityFeed?: ActivityFeedItem[];
  toolEvents: ToolEventItem[];
  fileChanges: FileChangeItem[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  plan?: PlanStep[];
  activeSkills?: string[];
  subagentReports?: Array<Record<string, unknown>>;
  activeModels?: {
    primary?: { provider?: string; model?: string; reasoning_effort?: string | null };
    worker?: { provider?: string; model?: string; reasoning_effort?: string | null };
  };
  thinking?: string | null;
  toolEvents?: ToolEventItem[];
  fileChanges?: FileChangeItem[];
  pendingApproval?: PendingApproval | null;
  status?: string;
}

export interface ChatThread {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  pendingApproval?: PendingApproval | null;
}

export interface WorkspaceNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  size?: number;
  modifiedAt?: string;
  children?: WorkspaceNode[];
}

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

export interface ProviderAvailability {
  base_url: string;
  env_keys: string[];
  primary_env_var: string;
  api_key_configured: boolean;
  masked_key?: string | null;
  requires_api_key: boolean;
  storage?: 'profile_vault' | 'env';
}

export interface ProfileVaultEntry {
  env_var: string;
  provider: string;
  provider_label: string;
  masked_key: string;
  updated_at: string;
}

export interface ProfileVaultSummary {
  username: string;
  encrypted: boolean;
  algorithm: string;
  entries: ProfileVaultEntry[];
}

export interface ProviderModelOption {
  id: string;
  name?: string | null;
  owned_by?: string | null;
  context_length?: number | null;
  is_free?: boolean;
  source?: 'live' | 'preset' | 'custom';
}

export interface ProviderModelsResponse {
  ok: boolean;
  provider: string;
  base_url: string;
  api_key_configured?: boolean;
  source: 'live' | 'preset' | 'error';
  models: ProviderModelOption[];
  count: number;
  error?: string | null;
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

export interface AgentHealthResponse {
  connected: boolean;
  core: {
    status?: string;
    agent_ready?: boolean;
    primary_model?: string;
    worker_model?: string;
    sandbox_backend?: string;
    sandbox?: {
      backend?: string;
      workspace_dir?: string;
    };
    skills_loaded?: string[];
    mcp?: Record<string, unknown>;
  } | null;
  error: string | null;
}
