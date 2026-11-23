export type NavSection = 'chat' | 'filesystem' | 'settings';
export type SettingsTab = 'llm' | 'accessories' | 'password' | 'skills-mcps' | 'appearance';
export type ThemeMode = 'light' | 'dark';
export type MediaModality = 'image' | 'video' | 'audio';
export type ModelRoleKey = 'primary' | 'worker' | MediaModality;

export interface ChatAttachment {
  id: string;
  name: string;
  kind: MediaModality;
  mimeType: string;
  size: number;
  url: string;
  storageKey?: string;
}

export interface AuthSession {
  authenticated: boolean;
  needsSetup?: boolean;
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

export type ContextCategoryKey =
  | 'system_prompt'
  | 'tools'
  | 'skills'
  | 'user_messages'
  | 'assistant_messages'
  | 'tool_outputs'
  | 'compacted_summary';

export type ContextClassificationLevel = 'optimal' | 'moderate' | 'high' | 'critical' | 'unknown';
export type ExecutionAccessMode = 'default' | 'full_access';

export interface ContextClassification {
  level: ContextClassificationLevel;
  label: string;
  description: string;
}

export interface ContextCategoryBreakdown {
  key: ContextCategoryKey;
  label: string;
  description: string;
  tokens: number;
  percent: number;
  share_of_used: number;
}

export type CustomProviderProtocol = 'openai_compatible' | 'anthropic_compatible';

export interface CustomProviderModelConfig {
  id: string;
  name?: string;
  context_window?: number | null;
  max_output_tokens?: number | null;
}

export interface CustomProviderConfig {
  id: string;
  name: string;
  base_url: string;
  protocol: CustomProviderProtocol;
  enabled: boolean;
  models: CustomProviderModelConfig[];
  timeout_seconds?: number | null;
  created_at?: string;
  updated_at?: string;
  api_key_configured?: boolean;
  masked_key?: string | null;
}

export interface PlanningTimeoutConfig {
  connect_timeout_seconds: number;
  read_timeout_seconds: number;
  max_retries: number;
  fallback_on_timeout: boolean;
  fallback_provider?: string | null;
  fallback_model?: string | null;
}

export interface ThreadSessionConfig {
  provider: string;
  model: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
  reasoning_effort: string;
  context_window?: number | null;
  context_window_source?: string | null;
  manual_context_window?: number | null;
  max_output_tokens?: number | null;
  access_mode: ExecutionAccessMode;
}

export interface ContextUsageSnapshot {
  model: string;
  provider: string;
  context_window: number | null;
  context_window_source: string;
  detection_mode?: 'auto' | 'manual' | 'unknown';
  is_manual_override?: boolean;
  manual_context_window?: number | null;
  max_output_tokens?: number | null;
  max_output_tokens_source?: string;
  configured_max_tokens?: number | null;
  reserved_output_tokens?: number;
  safety_buffer_tokens?: number;
  usable_input_window?: number | null;
  remaining_input_budget?: number | null;
  input_budget_percent?: number | null;
  used_tokens: number;
  free_tokens: number | null;
  usage_percent: number | null;
  auto_compact_threshold_percent: number;
  classification: ContextClassification;
  categories: ContextCategoryBreakdown[];
  compacted_count: number;
  last_compacted_at: string | null;
  tokens_saved_last_compaction: number;
  calibrated_from_provider: boolean;
}

export interface ActiveTaskItem {
  id: string;
  kind: string;
  label: string;
  elapsed_seconds: number;
  expected_seconds?: number;
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
  contextUsage?: ContextUsageSnapshot | null;
  activeTasks?: ActiveTaskItem[];
}

export interface ActiveBackgroundJob {
  jobId: string;
  threadId: string;
  status: string;
  activity: LiveAgentActivity;
  createdAt: number;
}

export type ContextWindowSourceType =
  | 'manual'
  | 'custom_provider_config'
  | 'provider_catalog'
  | 'provider_registry'
  | 'reference_catalog'
  | 'model_registry'
  | 'unknown';

export interface ModelResolutionResult {
  resolved: boolean;
  provider: string;
  model: string;
  matched_model: string | null;
  context_window: number | null;
  max_output_tokens: number | null;
  context_window_source: ContextWindowSourceType;
  max_output_tokens_source: ContextWindowSourceType;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  attachments?: ChatAttachment[];
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
  contextUsage?: ContextUsageSnapshot | null;
  sessionConfig?: ThreadSessionConfig;
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

export interface RoleLLMConfig {
  provider: string;
  model: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
  reasoning_effort: string;
  context_window?: number | null;
  context_window_source?: string | null;
  manual_context_window?: number | null;
  max_output_tokens?: number | null;
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
  accessories?: AccessoriesConfig;
  max_turns: number;
  access_mode?: ExecutionAccessMode;
  allow_unsafe_local_bash?: boolean;
  firecrawl_api_url?: string;
  custom_providers?: CustomProviderConfig[];
  planning_timeout?: PlanningTimeoutConfig;
}

export interface ProviderAvailability {
  base_url: string;
  env_keys: string[];
  primary_env_var: string;
  api_key_configured: boolean;
  masked_key?: string | null;
  requires_api_key: boolean;
  storage?: 'profile_vault' | 'env';
  is_custom?: boolean;
  name?: string;
  enabled?: boolean;
  protocol?: CustomProviderProtocol;
  models?: CustomProviderModelConfig[];
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
  max_output_tokens?: number | null;
  is_free?: boolean;
  input_modalities?: string[];
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

export interface FirecrawlHealthStatus {
  healthy?: boolean;
  reachable?: boolean;
  mode?: string;
  engine?: string | null;
  version?: string | null;
  api_url?: string;
  status?: string;
  detail?: string;
}

export interface AgentHealthResponse {
  connected: boolean;
  core: {
    status?: string;
    agent_ready?: boolean;
    primary_model?: string;
    worker_model?: string;
    sandbox_backend?: string;
    allow_unsafe_local_bash?: boolean;
    sandbox?: {
      backend?: string;
      workspace_dir?: string;
      allow_unsafe_local_bash?: boolean;
    };
    firecrawl?: FirecrawlHealthStatus;
    skills_loaded?: string[];
    mcp?: Record<string, unknown>;
  } | null;
  error: string | null;
}
