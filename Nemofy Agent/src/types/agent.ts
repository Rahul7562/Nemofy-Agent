export type NavSection = 'chat' | 'filesystem' | 'settings';
export type SettingsTab = 'llm' | 'password' | 'skills-mcps' | 'appearance';
export type ThemeMode = 'light' | 'dark';

export interface AuthSession {
  authenticated: boolean;
  username?: string;
  updatedAt?: string;
  mode?: 'server' | 'standalone';
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

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  plan?: PlanStep[];
  activeSkills?: string[];
  subagentReports?: Array<Record<string, unknown>>;
  activeModels?: {
    primary?: { provider?: string; model?: string };
    worker?: { provider?: string; model?: string };
  };
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
  api_key_configured: boolean;
  requires_api_key: boolean;
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
    skills_loaded?: string[];
    mcp?: Record<string, unknown>;
  } | null;
  error: string | null;
}
