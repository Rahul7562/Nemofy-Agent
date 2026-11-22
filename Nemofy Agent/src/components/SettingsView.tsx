import {
  AudioLines,
  Blocks,
  CheckCircle2,
  Cpu,
  Eye,
  EyeOff,
  Image as ImageIcon,
  KeyRound,
  Layers,
  Lock,
  Moon,
  Palette,
  Plus,
  RotateCcw,
  Save,
  Server,
  ShieldCheck,
  Sparkles,
  Sun,
  Trash2,
  Video as VideoIcon,
  Wifi,
} from 'lucide-react';
import { type FormEvent, useEffect, useState } from 'react';
import {
  changeAdminPassword,
  deleteProfileVaultKey,
  deleteSkillDocument,
  fetchAgentHealth,
  fetchLLMSettings,
  fetchMCPServers,
  fetchSkills,
  getRemoteBackendUrl,
  isStandaloneMode,
  probeLLMModel,
  saveLLMSettings,
  saveMCPServers,
  saveProfileVaultKey,
  saveSkillDocument,
  setRemoteBackendUrl,
} from '../lib/apiClient.ts';
import { validatePasswordChange } from '../lib/authValidation.ts';
import {
  LLM_PROVIDER_PRESETS,
  applyProviderPreset,
  canonicalizeProviderId,
  detectModelInputModalities,
  getDefaultAccessoriesConfig,
  getProviderPreset,
  normalizeAccessoriesConfig,
  validateLLMConfig,
} from '../lib/llmPresets.ts';
import { formatTokenCount, resolveModelLimits } from '../lib/contextUtils.ts';
import { getThinkingProfile, normalizeThinkingLevel } from '../lib/thinkingModes.ts';
import { ModelSelectorCombobox } from './ModelSelectorCombobox.tsx';
import type {
  AccessoryLLMConfig,
  AgentHealthResponse,
  MCPServerItem,
  MediaModality,
  ModelRoleKey,
  PersistedLLMConfig,
  ProfileVaultSummary,
  ProviderAvailability,
  ProviderModelOption,
  RoleLLMConfig,
  SettingsTab,
  SkillDocument,
  ThemeMode,
} from '../types/agent.ts';

interface SettingsViewProps {
  theme: ThemeMode;
  onSetTheme: (mode: ThemeMode) => void;
  onLLMConfigSaved: (config: PersistedLLMConfig) => void;
  initialTab?: SettingsTab;
  embedded?: boolean;
}

const SETTINGS_TABS: Array<{ id: SettingsTab; label: string; icon: typeof Cpu }> = [
  { id: 'llm', label: 'LLM Config', icon: Cpu },
  { id: 'accessories', label: 'Accessories', icon: Blocks },
  { id: 'password', label: 'Change Password', icon: KeyRound },
  { id: 'skills-mcps', label: 'Skills & MCPs', icon: Layers },
  { id: 'appearance', label: 'Appearance & Runtime', icon: Palette },
];

const ACCESSORY_META: Array<{
  modality: MediaModality;
  title: string;
  subtitle: string;
  icon: typeof ImageIcon;
}> = [
  {
    modality: 'image',
    title: 'Image Accessory',
    subtitle: 'Runs as a specialist vision sub-agent when your Primary LLM does not natively accept image inputs.',
    icon: ImageIcon,
  },
  {
    modality: 'video',
    title: 'Video Accessory',
    subtitle: 'Runs as a specialist video sub-agent when your Primary LLM does not natively accept video inputs.',
    icon: VideoIcon,
  },
  {
    modality: 'audio',
    title: 'Audio Accessory',
    subtitle: 'Runs as a specialist audio sub-agent when your Primary LLM does not natively accept audio inputs.',
    icon: AudioLines,
  },
];

function RoleConfigCard({
  title,
  subtitle,
  roleKey,
  config,
  providers,
  apiKeyDraft,
  onApiKeyChange,
  onChange,
  onProbe,
  probeState,
  headerBadge,
  enabledToggle,
}: {
  title: string;
  subtitle: string;
  roleKey: ModelRoleKey;
  config: RoleLLMConfig;
  providers: Record<string, ProviderAvailability>;
  apiKeyDraft: string;
  onApiKeyChange: (val: string) => void;
  onChange: (next: RoleLLMConfig) => void;
  onProbe: () => void;
  probeState: { loading: boolean; message?: string; ok?: boolean };
  headerBadge?: { text: string; tone: 'native' | 'delegate' | 'muted' };
  enabledToggle?: { checked: boolean; onToggle: (checked: boolean) => void };
}) {
  const [showKey, setShowKey] = useState(false);
  const [tempDraft, setTempDraft] = useState(String(config.temperature));
  const [tokensDraft, setTokensDraft] = useState(String(config.max_tokens));

  useEffect(() => {
    setTempDraft(String(config.temperature));
  }, [config.temperature]);

  useEffect(() => {
    setTokensDraft(String(config.max_tokens));
  }, [config.max_tokens]);

  const canonicalProvider = canonicalizeProviderId(config.provider);
  const preset = getProviderPreset(canonicalProvider);
  const selectValue = preset ? preset.id : 'custom';
  const avail = providers[canonicalProvider] || providers[selectValue];
  const thinkingProfile = getThinkingProfile(canonicalProvider, config.model);
  const effectiveThinking = normalizeThinkingLevel(thinkingProfile, config.reasoning_effort);
  const envVarName =
    selectValue === 'custom'
      ? roleKey === 'worker'
        ? 'WORKER_API_KEY'
        : roleKey === 'primary'
          ? 'PRIMARY_API_KEY'
          : `ACCESSORY_${roleKey.toUpperCase()}_API_KEY`
      : avail?.primary_env_var || preset?.envVars[0] || 'PRIMARY_API_KEY';

  function handleModelChange(nextModel: string) {
    const nextProfile = getThinkingProfile(canonicalProvider, nextModel);
    const trimmed = nextModel.trim();
    const resolved = trimmed ? resolveModelLimits(canonicalProvider, trimmed) : null;
    const autoMaxTokens =
      resolved && resolved.maxOutputTokensSource === 'model_registry'
        ? resolved.maxOutputTokens
        : config.max_tokens;
    onChange({
      ...config,
      provider: canonicalProvider,
      model: nextModel,
      context_window: undefined,
      max_output_tokens: undefined,
      max_tokens: autoMaxTokens,
      reasoning_effort: normalizeThinkingLevel(nextProfile, config.reasoning_effort),
    });
  }

  function handleSelectModelOption(option: ProviderModelOption) {
    const nextProfile = getThinkingProfile(canonicalProvider, option.id);
    const explicitCtx =
      typeof option.context_length === 'number' && option.context_length > 0
        ? Math.round(option.context_length)
        : undefined;
    const explicitOut =
      typeof option.max_output_tokens === 'number' && option.max_output_tokens > 0
        ? Math.round(option.max_output_tokens)
        : undefined;
    const resolved = resolveModelLimits(canonicalProvider, option.id, explicitCtx, explicitOut);
    onChange({
      ...config,
      provider: canonicalProvider,
      model: option.id,
      context_window: explicitCtx,
      max_output_tokens: explicitOut,
      max_tokens: resolved.maxOutputTokens,
      reasoning_effort: normalizeThinkingLevel(nextProfile, config.reasoning_effort),
    });
  }

  function handleCatalogLoaded(models: ProviderModelOption[]) {
    const currentId = config.model.trim();
    if (!currentId || models.length === 0) return;
    const matched =
      models.find((m) => m.id === currentId) ||
      models.find((m) => m.id.toLowerCase() === currentId.toLowerCase());
    if (!matched) return;
    const nextCtx =
      typeof matched.context_length === 'number' && matched.context_length > 0
        ? Math.round(matched.context_length)
        : config.context_window;
    const nextOut =
      typeof matched.max_output_tokens === 'number' && matched.max_output_tokens > 0
        ? Math.round(matched.max_output_tokens)
        : config.max_output_tokens;
    if (nextCtx === config.context_window && nextOut === config.max_output_tokens) {
      return;
    }
    onChange({
      ...config,
      context_window: nextCtx,
      max_output_tokens: nextOut,
    });
  }

  const resolvedLimits = config.model.trim()
    ? resolveModelLimits(
        canonicalProvider,
        config.model,
        config.context_window,
        config.max_output_tokens,
      )
    : null;

  const isDisabled = enabledToggle ? !enabledToggle.checked : false;

  return (
    <section
      className={`rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-5 sm:p-6 space-y-4 transition-opacity ${
        isDisabled ? 'opacity-70' : ''
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="font-display text-base font-semibold text-[var(--ink)]">{title}</h2>
          {headerBadge && (
            <div
              className={`mt-0.5 inline-flex items-center gap-1.5 text-[11px] font-medium ${
                headerBadge.tone === 'native'
                  ? 'text-[var(--jade)]'
                  : headerBadge.tone === 'delegate'
                    ? 'text-[var(--signal)]'
                    : 'text-[var(--ink-faint)]'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                  headerBadge.tone === 'native'
                    ? 'bg-[var(--jade)]'
                    : headerBadge.tone === 'delegate'
                      ? 'bg-[var(--signal)]'
                      : 'bg-[var(--ink-faint)]'
                }`}
              />
              <span>{headerBadge.text}</span>
            </div>
          )}
          <p className="text-xs text-[var(--ink-soft)] mt-1">{subtitle}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {enabledToggle && (
            <label className="inline-flex items-center gap-1.5 text-xs font-medium text-[var(--ink-soft)] cursor-pointer select-none">
              <input
                type="checkbox"
                checked={enabledToggle.checked}
                onChange={(e) => enabledToggle.onToggle(e.target.checked)}
                className="accent-[var(--signal)] w-3.5 h-3.5 cursor-pointer"
              />
              <span>{enabledToggle.checked ? 'Enabled' : 'Disabled'}</span>
            </label>
          )}
          <button
            type="button"
            disabled={probeState.loading || isDisabled}
            onClick={onProbe}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] disabled:opacity-45 text-xs font-medium text-[var(--ink)] transition-colors cursor-pointer shrink-0"
          >
            <Wifi className="w-3.5 h-3.5 text-[var(--signal)]" />
            <span>{probeState.loading ? 'Testing...' : 'Test'}</span>
          </button>
        </div>
      </div>

      {/* Provider */}
      <div>
        <label
          htmlFor={`${roleKey}-provider`}
          className="block text-xs font-medium text-[var(--ink)] mb-1.5"
        >
          Model Provider
        </label>
        <select
          id={`${roleKey}-provider`}
          value={selectValue}
          onChange={(e) =>
            onChange(
              applyProviderPreset(config, e.target.value, {
                clearModelOnProviderSwitch: roleKey === 'primary',
              }),
            )
          }
          className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
        >
          {LLM_PROVIDER_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </div>

      {/* Model ID (Live Provider Model Dropdown + Search + Custom ID) */}
      <div>
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <label
            htmlFor={`${roleKey}-model`}
            className="block text-xs font-medium text-[var(--ink)]"
          >
            Model
          </label>
          {resolvedLimits && (
            <span className="font-mono text-[11px] tabular-nums text-[var(--ink-faint)]">
              {formatTokenCount(resolvedLimits.contextWindow)} total · {formatTokenCount(resolvedLimits.maxOutputTokens)} out
              {roleKey === 'primary' ? ' · 80% auto-compact' : ''}
            </span>
          )}
        </div>
        <ModelSelectorCombobox
          id={`${roleKey}-model`}
          role={roleKey}
          provider={selectValue}
          baseUrl={config.base_url}
          apiKeyDraft={apiKeyDraft}
          value={config.model}
          onChange={handleModelChange}
          onSelectOption={handleSelectModelOption}
          onCatalogLoaded={handleCatalogLoaded}
        />
      </div>

      {/* Provider API Key (encrypted inside signed-in user's profile vault) */}
      <div>
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <label
            htmlFor={`${roleKey}-apikey`}
            className="block text-xs font-medium text-[var(--ink)]"
          >
            API Key <span className="font-mono text-[11px] text-[var(--ink-faint)]">(Vault: {envVarName})</span>
          </label>
          {avail && (
            <span
              className={`text-[11px] font-medium ${
                avail.api_key_configured ? 'text-[var(--jade)]' : 'text-[var(--honey)]'
              }`}
            >
              {avail.requires_api_key
                ? avail.api_key_configured
                  ? avail.masked_key
                    ? `In vault (${avail.masked_key})`
                    : 'Saved in vault'
                  : 'Not saved yet'
                : 'Optional (local)'}
            </span>
          )}
        </div>
        <div className="relative">
          <input
            id={`${roleKey}-apikey`}
            type={showKey ? 'text' : 'password'}
            autoComplete="off"
            value={apiKeyDraft}
            onChange={(e) => onApiKeyChange(e.target.value)}
            placeholder={
              avail?.api_key_configured && avail.masked_key
                ? `Leave blank to keep ${avail.masked_key}, or paste new key`
                : `Paste ${envVarName} to encrypt in your profile vault`
            }
            className="w-full h-10 pl-3 pr-10 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none focus:border-[var(--signal)]"
          />
          <button
            type="button"
            onClick={() => setShowKey((v) => !v)}
            aria-label={showKey ? 'Hide API key' : 'Show API key'}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-[var(--ink-faint)] hover:text-[var(--ink)] cursor-pointer"
          >
            {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Base URL */}
      <div>
        <label
          htmlFor={`${roleKey}-baseurl`}
          className="block text-xs font-medium text-[var(--ink)] mb-1.5"
        >
          OpenAI-Compatible Base URL
        </label>
        <input
          id={`${roleKey}-baseurl`}
          type="url"
          value={config.base_url}
          onChange={(e) => onChange({ ...config, base_url: e.target.value })}
          placeholder={preset?.baseUrl || 'https://api.openai.com/v1'}
          className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
        />
      </div>

      {/* Numeric & Reasoning Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div>
          <label
            htmlFor={`${roleKey}-temp`}
            className="block text-xs font-medium text-[var(--ink)] mb-1.5"
          >
            Temperature
          </label>
          <input
            id={`${roleKey}-temp`}
            type="number"
            step="0.01"
            min={0}
            max={2}
            value={tempDraft}
            onChange={(e) => {
              const raw = e.target.value;
              setTempDraft(raw);
              if (raw.trim() === '') return;
              const num = Number(raw);
              if (Number.isFinite(num)) {
                onChange({ ...config, temperature: num });
              }
            }}
            onBlur={() => {
              const num = Number(tempDraft);
              if (!tempDraft.trim() || !Number.isFinite(num)) {
                setTempDraft(String(config.temperature));
              }
            }}
            className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
          />
        </div>
        <div>
          <label
            htmlFor={`${roleKey}-tokens`}
            className="block text-xs font-medium text-[var(--ink)] mb-1.5"
          >
            Max Output Tokens
          </label>
          <input
            id={`${roleKey}-tokens`}
            type="number"
            step="1"
            min={128}
            max={20000000}
            value={tokensDraft}
            onChange={(e) => {
              const raw = e.target.value;
              setTokensDraft(raw);
              if (raw.trim() === '') return;
              const num = Number(raw);
              if (Number.isFinite(num)) {
                onChange({ ...config, max_tokens: Math.round(num) });
              }
            }}
            onBlur={() => {
              const num = Number(tokensDraft);
              if (!tokensDraft.trim() || !Number.isFinite(num)) {
                setTokensDraft(String(config.max_tokens));
              }
            }}
            className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
          />
          {resolvedLimits && (
            <div className="mt-1 flex items-center justify-between gap-1.5 text-[10px] text-[var(--ink-faint)]">
              <span className="font-mono tabular-nums truncate">
                Max: {formatTokenCount(resolvedLimits.maxOutputTokens)}
              </span>
              {config.max_tokens !== resolvedLimits.maxOutputTokens && (
                <button
                  type="button"
                  onClick={() => onChange({ ...config, max_tokens: resolvedLimits.maxOutputTokens })}
                  className="text-[var(--signal)] hover:underline cursor-pointer shrink-0 font-medium"
                >
                  Use full
                </button>
              )}
            </div>
          )}
        </div>
        <div>
          <label
            htmlFor={`${roleKey}-reasoning`}
            className="block text-xs font-medium text-[var(--ink)] mb-1.5"
          >
            Thinking Mode
          </label>
          <select
            id={`${roleKey}-reasoning`}
            value={effectiveThinking}
            disabled={!thinkingProfile.supported}
            onChange={(e) => onChange({ ...config, reasoning_effort: e.target.value })}
            className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-xs text-[var(--ink)] disabled:opacity-60 focus:outline-none focus:border-[var(--signal)]"
          >
            {thinkingProfile.options.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="mt-1 text-[10px] text-[var(--ink-faint)] truncate" title={thinkingProfile.familyLabel}>
            {thinkingProfile.familyLabel}
          </p>
        </div>
      </div>

      {probeState.message && (
        <div
          className={`rounded-lg px-3 py-2 text-xs border ${
            probeState.ok
              ? 'border-[var(--jade)]/35 bg-[var(--jade)]/10 text-[var(--jade)]'
              : 'border-[var(--flare)]/35 bg-[var(--flare)]/10 text-[var(--flare)]'
          }`}
        >
          {probeState.message}
        </div>
      )}
    </section>
  );
}

export function SettingsView({
  theme,
  onSetTheme,
  onLLMConfigSaved,
  initialTab = 'llm',
  embedded = false,
}: SettingsViewProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>(initialTab);

  // LLM Config & Profile API Key Vault State
  const [llmConfig, setLlmConfig] = useState<PersistedLLMConfig | null>(null);
  const [llmLoadError, setLlmLoadError] = useState<string | null>(null);
  const [providers, setProviders] = useState<Record<string, ProviderAvailability>>({});
  const [vault, setVault] = useState<ProfileVaultSummary | null>(null);
  const [vaultProvider, setVaultProvider] = useState('nvidia');
  const [vaultKeyInput, setVaultKeyInput] = useState('');
  const [showVaultKeyInput, setShowVaultKeyInput] = useState(false);
  const [vaultBusy, setVaultBusy] = useState(false);
  const [vaultMessage, setVaultMessage] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);
  const [primaryApiKey, setPrimaryApiKey] = useState('');
  const [workerApiKey, setWorkerApiKey] = useState('');
  const [accessoryApiKeys, setAccessoryApiKeys] = useState<Record<MediaModality, string>>({
    image: '',
    video: '',
    audio: '',
  });
  const [llmSaving, setLlmSaving] = useState(false);
  const [llmMessage, setLlmMessage] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);
  const [accessoriesMessage, setAccessoriesMessage] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);
  const [primaryProbe, setPrimaryProbe] = useState<{ loading: boolean; message?: string; ok?: boolean }>({
    loading: false,
  });
  const [workerProbe, setWorkerProbe] = useState<{ loading: boolean; message?: string; ok?: boolean }>({
    loading: false,
  });
  const [accessoryProbes, setAccessoryProbes] = useState<
    Record<MediaModality, { loading: boolean; message?: string; ok?: boolean }>
  >({
    image: { loading: false },
    video: { loading: false },
    audio: { loading: false },
  });

  // Change Password State
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPasswords, setShowPasswords] = useState(false);
  const [pwSaving, setPwSaving] = useState(false);
  const [pwFeedback, setPwFeedback] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);

  // Skills & MCPs State
  const [skills, setSkills] = useState<SkillDocument[]>([]);
  const [selectedSkillName, setSelectedSkillName] = useState<string | null>(null);
  const [skillDraftName, setSkillDraftName] = useState('');
  const [skillDraftContent, setSkillDraftContent] = useState('');
  const [skillFeedback, setSkillFeedback] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);

  const [mcpServers, setMcpServers] = useState<MCPServerItem[]>([]);
  const [newMcpName, setNewMcpName] = useState('');
  const [newMcpUrl, setNewMcpUrl] = useState('');
  const [newMcpTransport, setNewMcpTransport] = useState<'streamable_http' | 'sse'>('streamable_http');
  const [mcpFeedback, setMcpFeedback] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);

  // Runtime & Remote URL State
  const [health, setHealth] = useState<AgentHealthResponse | null>(null);
  const [remoteUrl, setRemoteUrl] = useState(() => getRemoteBackendUrl());
  const [remoteSaved, setRemoteSaved] = useState(false);

  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  async function loadAllSettings() {
    setLlmLoadError(null);
    const [llmResult, skillsResult, mcpsResult, healthResult] = await Promise.allSettled([
      fetchLLMSettings(),
      fetchSkills(),
      fetchMCPServers(),
      fetchAgentHealth(),
    ]);

    if (llmResult.status === 'fulfilled') {
      setLlmConfig(llmResult.value.config);
      setProviders(llmResult.value.providers);
      if (llmResult.value.vault) {
        setVault(llmResult.value.vault);
      }
    } else {
      setLlmLoadError(
        llmResult.reason instanceof Error
          ? llmResult.reason.message
          : 'Unable to load LLM settings from the server.',
      );
    }

    if (skillsResult.status === 'fulfilled') {
      setSkills(skillsResult.value);
      if (skillsResult.value.length > 0) {
        setSelectedSkillName((prev) => prev || skillsResult.value[0].name);
        setSkillDraftName((prev) => prev || skillsResult.value[0].name);
        setSkillDraftContent((prev) => prev || skillsResult.value[0].content);
      }
    }

    if (mcpsResult.status === 'fulfilled') {
      setMcpServers(mcpsResult.value);
    }

    if (healthResult.status === 'fulfilled') {
      setHealth(healthResult.value);
    }
  }

  useEffect(() => {
    void loadAllSettings();
  }, []);

  async function handleSaveLLM(e: FormEvent, source: 'llm' | 'accessories' = 'llm') {
    e.preventDefault();
    if (!llmConfig) return;
    const setStatus = source === 'accessories' ? setAccessoriesMessage : setLlmMessage;
    setStatus(null);
    const normalizedConfig: PersistedLLMConfig = {
      ...llmConfig,
      accessories: normalizeAccessoriesConfig(llmConfig.accessories),
    };
    const check = validateLLMConfig(normalizedConfig, source);
    if (!check.valid) {
      setStatus({ type: 'error', text: check.error || 'Invalid configuration.' });
      return;
    }

    setLlmSaving(true);
    try {
      const apiKeysPayload: Partial<Record<ModelRoleKey, string>> = {};
      if (primaryApiKey.trim()) apiKeysPayload.primary = primaryApiKey.trim();
      if (workerApiKey.trim()) apiKeysPayload.worker = workerApiKey.trim();
      if (accessoryApiKeys.image.trim()) apiKeysPayload.image = accessoryApiKeys.image.trim();
      if (accessoryApiKeys.video.trim()) apiKeysPayload.video = accessoryApiKeys.video.trim();
      if (accessoryApiKeys.audio.trim()) apiKeysPayload.audio = accessoryApiKeys.audio.trim();

      const saved = await saveLLMSettings(
        normalizedConfig,
        Object.keys(apiKeysPayload).length > 0 ? apiKeysPayload : undefined,
      );
      setLlmConfig(saved.config);
      setProviders(saved.providers);
      if (saved.vault) {
        setVault(saved.vault);
      }
      setPrimaryApiKey('');
      setWorkerApiKey('');
      setAccessoryApiKeys({ image: '', video: '', audio: '' });
      onLLMConfigSaved(saved.config);
      const envSummary =
        saved.savedEnvKeys && saved.savedEnvKeys.length > 0
          ? ` Encrypted & stored in profile vault: ${saved.savedEnvKeys.join(', ')}.`
          : '';
      const syncNote = saved.syncWarning ? ` Note: ${saved.syncWarning}` : '';
      setStatus({
        type: 'ok',
        text:
          source === 'accessories'
            ? `Multimodal Accessory models (Image, Video, Audio) saved.${envSummary}${syncNote}`
            : `Main Agent (Nemo) and Sub-Agent LLM configuration saved.${envSummary}${syncNote}`,
      });
    } catch (err) {
      setStatus({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to save LLM configuration.',
      });
    } finally {
      setLlmSaving(false);
    }
  }

  function handleUpdateAccessory(modality: MediaModality, next: AccessoryLLMConfig) {
    if (!llmConfig) return;
    const currentAcc = normalizeAccessoriesConfig(llmConfig.accessories);
    setLlmConfig({
      ...llmConfig,
      accessories: {
        ...currentAcc,
        [modality]: next,
      },
    });
    setAccessoryProbes((prev) => ({ ...prev, [modality]: { loading: false } }));
  }

  function handleResetAccessoriesToDefault() {
    if (!llmConfig) return;
    setLlmConfig({
      ...llmConfig,
      accessories: getDefaultAccessoriesConfig(),
    });
    setAccessoryProbes({
      image: { loading: false },
      video: { loading: false },
      audio: { loading: false },
    });
    setAccessoriesMessage({
      type: 'ok',
      text: 'Loaded OpenRouter Nemotron Omni default for Image, Video, and Audio. Click Save Accessories to apply.',
    });
  }

  async function handleSaveVaultKey() {
    const cleanKey = vaultKeyInput.trim();
    if (!cleanKey) {
      setVaultMessage({ type: 'error', text: 'Paste an API key before saving.' });
      return;
    }
    setVaultBusy(true);
    setVaultMessage(null);
    try {
      const res = await saveProfileVaultKey({
        provider: vaultProvider,
        api_key: cleanKey,
      });
      setVault(res.vault);
      setProviders(res.providers);
      setVaultKeyInput('');
      setVaultMessage({
        type: 'ok',
        text: `Encrypted and saved ${res.savedKey} to ${res.vault.username}'s profile vault.`,
      });
    } catch (err) {
      setVaultMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to store API key.',
      });
    } finally {
      setVaultBusy(false);
    }
  }

  async function handleDeleteVaultKey(envVar: string) {
    setVaultBusy(true);
    setVaultMessage(null);
    try {
      const res = await deleteProfileVaultKey(envVar);
      setVault(res.vault);
      setProviders(res.providers);
      setVaultMessage({
        type: 'ok',
        text: `Removed ${envVar} from ${res.vault.username}'s profile vault.`,
      });
    } catch (err) {
      setVaultMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to remove API key.',
      });
    } finally {
      setVaultBusy(false);
    }
  }

  async function handleProbe(role: ModelRoleKey) {
    if (!llmConfig) return;
    const isAccessory = role === 'image' || role === 'video' || role === 'audio';
    const roleCfg: RoleLLMConfig = isAccessory
      ? normalizeAccessoriesConfig(llmConfig.accessories)[role]
      : llmConfig[role];
    const keyDraft = isAccessory
      ? accessoryApiKeys[role].trim()
      : role === 'primary'
        ? primaryApiKey.trim()
        : workerApiKey.trim();

    const setProbe = (state: { loading: boolean; message?: string; ok?: boolean }) => {
      if (isAccessory) {
        setAccessoryProbes((prev) => ({ ...prev, [role]: state }));
      } else if (role === 'primary') {
        setPrimaryProbe(state);
      } else {
        setWorkerProbe(state);
      }
    };

    setProbe({ loading: true });
    try {
      const res = await probeLLMModel(role, roleCfg, keyDraft || undefined);
      const ok = Boolean(
        res.ok !== undefined
          ? res.ok
          : res.authenticated === true && !res.error,
      );
      const modelListedNote =
        ok && res.model_listed === false
          ? ' (Endpoint authenticated, though this exact model ID was not found in /models — verify spelling.)'
          : '';
      setProbe({
        loading: false,
        ok,
        message: ok
          ? `Connected to ${String(res.provider || roleCfg.provider)} (${String(res.model || roleCfg.model)}).${modelListedNote}`
          : String(res.error || res.detail || 'Probe failed. Check provider key or endpoint.'),
      });
    } catch (err) {
      setProbe({
        loading: false,
        ok: false,
        message: err instanceof Error ? err.message : 'Probe request failed.',
      });
    }
  }

  async function handlePasswordSubmit(e: FormEvent) {
    e.preventDefault();
    setPwFeedback(null);
    const check = validatePasswordChange(currentPassword, newPassword, confirmPassword);
    if (!check.valid) {
      setPwFeedback({ type: 'error', text: check.error || 'Invalid password input.' });
      return;
    }

    setPwSaving(true);
    try {
      const res = await changeAdminPassword(currentPassword, newPassword);
      if (!res.ok) {
        setPwFeedback({ type: 'error', text: res.error || 'Password update failed.' });
        return;
      }
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setPwFeedback({
        type: 'ok',
        text: res.message || 'Password updated. All previous sessions have been rotated.',
      });
    } catch (err) {
      setPwFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to update password.',
      });
    } finally {
      setPwSaving(false);
    }
  }

  async function handleSaveSkill(e: FormEvent) {
    e.preventDefault();
    if (!skillDraftName.trim()) return;
    setSkillFeedback(null);
    try {
      const saved = await saveSkillDocument(skillDraftName.trim(), skillDraftContent);
      const refreshed = await fetchSkills();
      setSkills(refreshed);
      setSelectedSkillName(saved.name);
      setSkillDraftName(saved.name);
      setSkillFeedback({ type: 'ok', text: `Saved /Skills/${saved.filename}` });
    } catch (err) {
      setSkillFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to save skill.',
      });
    }
  }

  async function handleDeleteSkill(name: string) {
    setSkillFeedback(null);
    try {
      await deleteSkillDocument(name);
      const refreshed = await fetchSkills();
      setSkills(refreshed);
      if (selectedSkillName === name) {
        setSelectedSkillName(refreshed[0]?.name || null);
        setSkillDraftName(refreshed[0]?.name || '');
        setSkillDraftContent(refreshed[0]?.content || '');
      }
      setSkillFeedback({ type: 'ok', text: `Deleted skill "${name}"` });
    } catch (err) {
      setSkillFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to delete skill.',
      });
    }
  }

  async function handleAddMcpServer(e: FormEvent) {
    e.preventDefault();
    if (!newMcpName.trim() || !newMcpUrl.trim()) return;
    setMcpFeedback(null);
    const next: MCPServerItem[] = [
      ...mcpServers,
      {
        name: newMcpName.trim(),
        url: newMcpUrl.trim(),
        transport: newMcpTransport,
        enabled: true,
        timeout_seconds: 30,
      },
    ];
    try {
      const saved = await saveMCPServers(next);
      setMcpServers(saved);
      setNewMcpName('');
      setNewMcpUrl('');
      setMcpFeedback({ type: 'ok', text: 'MCP server registry saved to /MCPs/servers.json.' });
    } catch (err) {
      setMcpFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to save MCP server.',
      });
    }
  }

  async function handleToggleMcp(index: number) {
    setMcpFeedback(null);
    const next = mcpServers.map((s, idx) => (idx === index ? { ...s, enabled: !s.enabled } : s));
    try {
      const saved = await saveMCPServers(next);
      setMcpServers(saved);
    } catch (err) {
      setMcpFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to update MCP server.',
      });
    }
  }

  async function handleDeleteMcp(index: number) {
    setMcpFeedback(null);
    const next = mcpServers.filter((_, idx) => idx !== index);
    try {
      const saved = await saveMCPServers(next);
      setMcpServers(saved);
    } catch (err) {
      setMcpFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to remove MCP server.',
      });
    }
  }

  function handleSaveRemoteBackend(e: FormEvent) {
    e.preventDefault();
    setRemoteBackendUrl(remoteUrl);
    setRemoteSaved(true);
    setTimeout(() => setRemoteSaved(false), 3000);
  }

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden scrollbar-none bg-[var(--paper)]">
      <div className={`max-w-5xl mx-auto ${embedded ? 'px-5 sm:px-6 py-5' : 'px-5 sm:px-8 py-6 sm:py-8'} space-y-6`}>
        {/* Header */}
        {!embedded && (
          <div>
            <h1 className="font-display text-2xl font-semibold tracking-tight text-[var(--ink)]">
              Settings
            </h1>
            <p className="mt-1 text-sm text-[var(--ink-soft)]">
              Configure Main Agent &amp; Sub-Agent LLMs, administrator security, dynamic skills, and MCP servers.
            </p>
          </div>
        )}

        {/* Sub-navigation Tabs */}
        <div
          role="tablist"
          aria-label="Settings sections"
          className="flex flex-wrap gap-1.5 border-b border-[var(--rule)] pb-3"
        >
          {SETTINGS_TABS.map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                type="button"
                aria-selected={active}
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs sm:text-sm font-medium transition-colors cursor-pointer ${
                  active
                    ? 'bg-[var(--blush)] text-[var(--signal)] font-semibold'
                    : 'text-[var(--ink-soft)] hover:bg-[var(--paper-sunk)] hover:text-[var(--ink)]'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* TAB 1: LLM CONFIG (Main Agent & Sub-Agent) */}
        {activeTab === 'llm' && (
          <form onSubmit={(e) => void handleSaveLLM(e, 'llm')} noValidate className="space-y-5">
            {llmConfig ? (
              <>
                {llmMessage && (
                  <div
                    role={llmMessage.type === 'error' ? 'alert' : 'status'}
                    className={`rounded-lg px-4 py-3 text-sm border ${
                      llmMessage.type === 'ok'
                        ? 'border-[var(--jade)]/35 bg-[var(--jade)]/10 text-[var(--jade)]'
                        : 'border-[var(--flare)]/35 bg-[var(--flare)]/10 text-[var(--flare)]'
                    }`}
                  >
                    {llmMessage.text}
                  </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                  <RoleConfigCard
                    title="Main Agent (Nemo)"
                    subtitle="Handles conversation, planning, reasoning, and final responses."
                    roleKey="primary"
                    config={llmConfig.primary}
                    providers={providers}
                    apiKeyDraft={primaryApiKey}
                    onApiKeyChange={(val) => {
                      setPrimaryApiKey(val);
                      setPrimaryProbe({ loading: false });
                    }}
                    onChange={(next) => {
                      setLlmConfig({ ...llmConfig, primary: next });
                      setPrimaryProbe({ loading: false });
                    }}
                    onProbe={() => void handleProbe('primary')}
                    probeState={primaryProbe}
                  />
                  <RoleConfigCard
                    title="Sub-Agent (Worker)"
                    subtitle="Executes delegated research, coding, and parallel sub-tasks."
                    roleKey="worker"
                    config={llmConfig.worker}
                    providers={providers}
                    apiKeyDraft={workerApiKey}
                    onApiKeyChange={(val) => {
                      setWorkerApiKey(val);
                      setWorkerProbe({ loading: false });
                    }}
                    onChange={(next) => {
                      setLlmConfig({ ...llmConfig, worker: next });
                      setWorkerProbe({ loading: false });
                    }}
                    onProbe={() => void handleProbe('worker')}
                    probeState={workerProbe}
                  />
                </div>

                <div className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div>
                      <label
                        htmlFor="max-turns-input"
                        className="block text-xs font-medium text-[var(--ink)]"
                      >
                        Max Execution Turns
                      </label>
                      <p className="text-[11px] text-[var(--ink-faint)]">
                        Upper bound on autonomous tool loops per request (1–100).
                      </p>
                    </div>
                    <input
                      id="max-turns-input"
                      type="number"
                      min={1}
                      max={100}
                      value={llmConfig.max_turns || ''}
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (!raw.trim()) return;
                        const num = Number(raw);
                        if (Number.isFinite(num)) {
                          setLlmConfig({ ...llmConfig, max_turns: Math.round(num) });
                        }
                      }}
                      className="w-24 h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={llmSaving}
                    className="h-10 px-5 inline-flex items-center justify-center gap-2 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-50 text-white text-sm font-semibold transition-colors cursor-pointer"
                  >
                    <Save className="w-4 h-4" />
                    <span>{llmSaving ? 'Saving...' : 'Save LLM Config'}</span>
                  </button>
                </div>

                {/* Per-Profile Encrypted API Key Vault */}
                <section
                  aria-label="Profile API Key Vault"
                  className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-5 sm:p-6 space-y-4"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h2 className="font-display text-base font-semibold text-[var(--ink)] flex items-center gap-2">
                        <ShieldCheck className="w-4 h-4 text-[var(--jade)]" />
                        <span>
                          Profile API Key Vault ({vault?.username || 'admin'})
                        </span>
                      </h2>
                      <p className="text-xs text-[var(--ink-soft)] mt-0.5">
                        API keys are encrypted at rest on the server and automatically loaded whenever you sign in as{' '}
                        <span className="font-mono text-[var(--ink)]">{vault?.username || 'admin'}</span>.
                      </p>
                    </div>
                    <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-[var(--jade)]">
                      <Lock className="w-3 h-3" />
                      <span>AES-256-GCM · Auto-loaded</span>
                    </span>
                  </div>

                  {vaultMessage && (
                    <div
                      role={vaultMessage.type === 'error' ? 'alert' : 'status'}
                      className={`rounded-lg px-3.5 py-2 text-xs border ${
                        vaultMessage.type === 'ok'
                          ? 'border-[var(--jade)]/35 bg-[var(--jade)]/10 text-[var(--jade)]'
                          : 'border-[var(--flare)]/35 bg-[var(--flare)]/10 text-[var(--flare)]'
                      }`}
                    >
                      {vaultMessage.text}
                    </div>
                  )}

                  {vault && vault.entries.length > 0 ? (
                    <div className="divide-y divide-[var(--rule)] border border-[var(--rule)] rounded-lg overflow-hidden">
                      {vault.entries.map((entry) => (
                        <div
                          key={entry.env_var}
                          className="px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 bg-[var(--paper)]"
                        >
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-semibold text-[var(--ink)]">
                                {entry.provider_label}
                              </span>
                              <span className="font-mono text-[11px] text-[var(--ink-faint)]">
                                {entry.env_var}
                              </span>
                            </div>
                            <div className="font-mono text-xs text-[var(--jade)] mt-0.5">
                              {entry.masked_key}
                            </div>
                          </div>
                          <button
                            type="button"
                            disabled={vaultBusy}
                            onClick={() => void handleDeleteVaultKey(entry.env_var)}
                            aria-label={`Remove ${entry.env_var} from profile vault`}
                            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md border border-[var(--rule)] text-xs text-[var(--ink-soft)] hover:text-[var(--flare)] hover:border-[var(--flare)]/40 disabled:opacity-50 transition-colors cursor-pointer"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                            <span>Remove</span>
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-[var(--ink-faint)]">
                      No API keys stored in this profile vault yet. Add a provider key below or save one from either agent card above.
                    </p>
                  )}

                  {/* Quick-Add Provider Key to Profile Vault */}
                  <div className="grid grid-cols-1 sm:grid-cols-12 gap-2.5 pt-1">
                    <div className="sm:col-span-4">
                      <label htmlFor="vault-provider-select" className="sr-only">
                        Provider for new vault API key
                      </label>
                      <select
                        id="vault-provider-select"
                        value={vaultProvider}
                        onChange={(e) => setVaultProvider(e.target.value)}
                        className="w-full h-9 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                      >
                        {LLM_PROVIDER_PRESETS.filter((p) => p.requiresApiKey).map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.label} ({p.envVars[0] || 'PRIMARY_API_KEY'})
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="sm:col-span-6 relative">
                      <label htmlFor="vault-apikey-input" className="sr-only">
                        API Key value
                      </label>
                      <input
                        id="vault-apikey-input"
                        type={showVaultKeyInput ? 'text' : 'password'}
                        autoComplete="off"
                        value={vaultKeyInput}
                        onChange={(e) => setVaultKeyInput(e.target.value)}
                        placeholder="Paste API key to store securely in your profile"
                        className="w-full h-9 pl-3 pr-9 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none focus:border-[var(--signal)]"
                      />
                      <button
                        type="button"
                        onClick={() => setShowVaultKeyInput((v) => !v)}
                        aria-label={showVaultKeyInput ? 'Hide vault API key' : 'Show vault API key'}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-[var(--ink-faint)] hover:text-[var(--ink)] cursor-pointer"
                      >
                        {showVaultKeyInput ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                    <div className="sm:col-span-2">
                      <button
                        type="button"
                        disabled={vaultBusy || !vaultKeyInput.trim()}
                        onClick={() => void handleSaveVaultKey()}
                        className="w-full h-9 px-3 inline-flex items-center justify-center gap-1.5 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] hover:text-[var(--signal)] disabled:opacity-45 text-xs font-semibold text-[var(--ink)] transition-colors cursor-pointer"
                      >
                        <KeyRound className="w-3.5 h-3.5" />
                        <span>Store Key</span>
                      </button>
                    </div>
                  </div>
                </section>
              </>
            ) : llmLoadError ? (
              <div
                role="alert"
                className="rounded-xl border border-[var(--flare)]/35 bg-[var(--flare)]/10 p-4 flex items-center justify-between gap-3 text-sm text-[var(--flare)]"
              >
                <span>{llmLoadError}</span>
                <button
                  type="button"
                  onClick={() => void loadAllSettings()}
                  className="px-3 py-1.5 rounded-lg border border-[var(--flare)]/40 bg-[var(--paper)] text-xs font-medium text-[var(--ink)] hover:border-[var(--signal)] cursor-pointer shrink-0"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="text-sm text-[var(--ink-faint)]">Loading LLM configuration...</div>
            )}
          </form>
        )}

        {/* TAB 2: ACCESSORIES (Multimodal Image / Video / Audio Sub-Agents) */}
        {activeTab === 'accessories' && (
          <form onSubmit={(e) => void handleSaveLLM(e, 'accessories')} noValidate className="space-y-5">
            {llmConfig ? (
              (() => {
                const accessories = normalizeAccessoriesConfig(llmConfig.accessories);
                const primaryModalities = detectModelInputModalities(
                  llmConfig.primary.provider,
                  llmConfig.primary.model,
                );
                return (
                  <>
                    {accessoriesMessage && (
                      <div
                        role={accessoriesMessage.type === 'error' ? 'alert' : 'status'}
                        className={`rounded-lg px-4 py-3 text-sm border ${
                          accessoriesMessage.type === 'ok'
                            ? 'border-[var(--jade)]/35 bg-[var(--jade)]/10 text-[var(--jade)]'
                            : 'border-[var(--flare)]/35 bg-[var(--flare)]/10 text-[var(--flare)]'
                        }`}
                      >
                        {accessoriesMessage.text}
                      </div>
                    )}

                    {/* Primary Model Multimodal Routing Overview */}
                    <section className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-5 space-y-3.5">
                      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <h2 className="font-display text-base font-semibold text-[var(--ink)]">
                            Multimodal Input Routing
                          </h2>
                          <p className="text-xs text-[var(--ink-soft)] mt-0.5 leading-relaxed">
                            When you attach images, video, or audio in chat, Nemo sends supported modalities directly to your active Main Agent
                            {llmConfig.primary.model.trim() ? (
                              <>
                                {' '}(<span className="font-mono text-[var(--ink)]">{llmConfig.primary.model}</span>)
                              </>
                            ) : (
                              ' (not selected yet)'
                            )}
                            {' '}and automatically spawns the matching Accessory sub-agent below for any unsupported modality.
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={handleResetAccessoriesToDefault}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] text-xs font-medium text-[var(--ink)] transition-colors cursor-pointer shrink-0 self-start"
                        >
                          <RotateCcw className="w-3.5 h-3.5 text-[var(--signal)]" />
                          <span>Use Nemotron Omni for all 3</span>
                        </button>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
                        {ACCESSORY_META.map(({ modality, icon: ModIcon }) => {
                          const native = primaryModalities.has(modality);
                          const accEntry = accessories[modality];
                          return (
                            <div
                              key={modality}
                              className="rounded-lg border border-[var(--rule)] bg-[var(--paper)] px-3.5 py-2.5 flex items-center justify-between gap-2"
                            >
                              <div className="flex items-center gap-2 min-w-0">
                                <ModIcon className="w-4 h-4 text-[var(--signal)] shrink-0" />
                                <span className="text-xs font-semibold capitalize text-[var(--ink)]">
                                  {modality}
                                </span>
                              </div>
                              <span
                                className={`text-[11px] font-mono truncate ${
                                  native
                                    ? 'text-[var(--jade)]'
                                    : accEntry.enabled
                                      ? 'text-[var(--signal)]'
                                      : 'text-[var(--ink-faint)]'
                                }`}
                              >
                                {native
                                  ? 'Native (Primary)'
                                  : accEntry.enabled
                                    ? 'Accessory sub-agent'
                                    : 'Disabled'}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </section>

                    {/* Per-Modality Accessory Configuration Cards */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                      {ACCESSORY_META.map(({ modality, title, subtitle }) => {
                        const accCfg = accessories[modality];
                        const nativeSupported = primaryModalities.has(modality);
                        const headerBadge = nativeSupported
                          ? { text: 'Standby (Primary supports natively)', tone: 'native' as const }
                          : accCfg.enabled
                            ? { text: 'Active for current Primary', tone: 'delegate' as const }
                            : { text: 'Disabled', tone: 'muted' as const };

                        return (
                          <RoleConfigCard
                            key={modality}
                            title={title}
                            subtitle={subtitle}
                            roleKey={modality}
                            config={accCfg}
                            providers={providers}
                            apiKeyDraft={accessoryApiKeys[modality]}
                            onApiKeyChange={(val) => {
                              setAccessoryApiKeys((prev) => ({ ...prev, [modality]: val }));
                              setAccessoryProbes((prev) => ({ ...prev, [modality]: { loading: false } }));
                            }}
                            onChange={(nextRole) =>
                              handleUpdateAccessory(modality, {
                                ...nextRole,
                                enabled: accCfg.enabled,
                              })
                            }
                            onProbe={() => void handleProbe(modality)}
                            probeState={accessoryProbes[modality]}
                            headerBadge={headerBadge}
                            enabledToggle={{
                              checked: accCfg.enabled,
                              onToggle: (checked) =>
                                handleUpdateAccessory(modality, { ...accCfg, enabled: checked }),
                            }}
                          />
                        );
                      })}
                    </div>

                    <div className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      <p className="text-xs text-[var(--ink-soft)] leading-relaxed">
                        Accessory API keys are encrypted in your signed-in profile vault ({vault?.username || 'admin'}) and reused automatically whenever a specialist sub-agent runs.
                      </p>
                      <button
                        type="submit"
                        disabled={llmSaving}
                        className="h-10 px-5 inline-flex items-center justify-center gap-2 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-50 text-white text-sm font-semibold transition-colors cursor-pointer shrink-0"
                      >
                        <Save className="w-4 h-4" />
                        <span>{llmSaving ? 'Saving...' : 'Save Accessories'}</span>
                      </button>
                    </div>
                  </>
                );
              })()
            ) : llmLoadError ? (
              <div
                role="alert"
                className="rounded-xl border border-[var(--flare)]/35 bg-[var(--flare)]/10 p-4 flex items-center justify-between gap-3 text-sm text-[var(--flare)]"
              >
                <span>{llmLoadError}</span>
                <button
                  type="button"
                  onClick={() => void loadAllSettings()}
                  className="px-3 py-1.5 rounded-lg border border-[var(--flare)]/40 bg-[var(--paper)] text-xs font-medium text-[var(--ink)] hover:border-[var(--signal)] cursor-pointer shrink-0"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="text-sm text-[var(--ink-faint)]">Loading Accessories configuration...</div>
            )}
          </form>
        )}

        {/* TAB 3: CHANGE PASSWORD */}
        {activeTab === 'password' && (
          <section className="max-w-lg rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-6 space-y-5">
            <div>
              <h2 className="font-display text-base font-semibold text-[var(--ink)]">
                Change Administrator Password
              </h2>
              <p className="mt-1 text-xs text-[var(--ink-soft)] leading-relaxed">
                Update the credentials protecting this public Nemofy Agent portal. Changing the password rotates active session tokens immediately.
              </p>
            </div>

            <form onSubmit={handlePasswordSubmit} className="space-y-4">
              <div>
                <label
                  htmlFor="current-password"
                  className="block text-xs font-medium text-[var(--ink)] mb-1.5"
                >
                  Current Password
                </label>
                <div className="relative">
                  <input
                    id="current-password"
                    type={showPasswords ? 'text' : 'password'}
                    autoComplete="current-password"
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    placeholder="Enter current password"
                    className="w-full h-10 pl-3 pr-10 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPasswords((v) => !v)}
                    aria-label={showPasswords ? 'Hide passwords' : 'Show passwords'}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-[var(--ink-faint)] hover:text-[var(--ink)] cursor-pointer"
                  >
                    {showPasswords ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <div>
                <label
                  htmlFor="new-password"
                  className="block text-xs font-medium text-[var(--ink)] mb-1.5"
                >
                  New Password
                </label>
                <input
                  id="new-password"
                  type={showPasswords ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="At least 8 characters"
                  className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                />
              </div>

              <div>
                <label
                  htmlFor="confirm-password"
                  className="block text-xs font-medium text-[var(--ink)] mb-1.5"
                >
                  Confirm New Password
                </label>
                <input
                  id="confirm-password"
                  type={showPasswords ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter new password"
                  className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                />
              </div>

              {pwFeedback && (
                <div
                  role={pwFeedback.type === 'error' ? 'alert' : 'status'}
                  className={`rounded-lg px-3.5 py-2.5 text-xs border ${
                    pwFeedback.type === 'ok'
                      ? 'border-[var(--jade)]/35 bg-[var(--jade)]/10 text-[var(--jade)]'
                      : 'border-[var(--flare)]/35 bg-[var(--flare)]/10 text-[var(--flare)]'
                  }`}
                >
                  {pwFeedback.text}
                </div>
              )}

              <button
                type="submit"
                disabled={pwSaving}
                className="h-10 px-5 inline-flex items-center gap-2 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-50 text-white text-sm font-semibold transition-colors cursor-pointer"
              >
                <KeyRound className="w-4 h-4" />
                <span>{pwSaving ? 'Updating...' : 'Update Password'}</span>
              </button>
            </form>
          </section>
        )}

        {/* TAB 3: SKILLS & MCPS */}
        {activeTab === 'skills-mcps' && (
          <div className="space-y-6">
            {/* Dynamic Skills */}
            <section className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-5 sm:p-6 space-y-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="font-display text-base font-semibold text-[var(--ink)] flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-[var(--signal)]" />
                    <span>Agent Skills (/Skills)</span>
                  </h2>
                  <p className="text-xs text-[var(--ink-soft)] mt-0.5">
                    Markdown playbooks loaded dynamically by the Python agent runtime.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setSelectedSkillName(null);
                    setSkillDraftName('custom_skill');
                    setSkillDraftContent(
                      '---\nname: custom_skill\ndescription: Describe when the agent should load this skill.\n---\n\n# Custom Skill Instructions\n\n- Step 1\n- Step 2\n',
                    );
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--rule-strong)] text-xs font-medium text-[var(--ink)] hover:border-[var(--signal)] cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>New Skill</span>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-1.5">
                  {skills.map((s) => (
                    <div
                      key={s.name}
                      className={`flex items-center justify-between gap-2 px-3 py-2 rounded-lg border text-xs ${
                        selectedSkillName === s.name
                          ? 'border-[var(--signal)] bg-[var(--blush)] text-[var(--signal)] font-medium'
                          : 'border-[var(--rule)] bg-[var(--paper)] text-[var(--ink-soft)]'
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedSkillName(s.name);
                          setSkillDraftName(s.name);
                          setSkillDraftContent(s.content);
                        }}
                        className="flex-1 text-left truncate cursor-pointer"
                      >
                        {s.name}
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleDeleteSkill(s.name)}
                        aria-label={`Delete skill ${s.name}`}
                        className="p-1 hover:text-[var(--flare)] cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                  {skills.length === 0 && (
                    <p className="text-xs text-[var(--ink-faint)]">No skills found in /Skills.</p>
                  )}
                </div>

                <form onSubmit={handleSaveSkill} className="md:col-span-2 space-y-3">
                  <div>
                    <label htmlFor="skill-name" className="block text-xs font-medium text-[var(--ink)] mb-1">
                      Skill Name
                    </label>
                    <input
                      id="skill-name"
                      type="text"
                      value={skillDraftName}
                      onChange={(e) => setSkillDraftName(e.target.value)}
                      className="w-full h-9 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                    />
                  </div>
                  <div>
                    <label htmlFor="skill-body" className="block text-xs font-medium text-[var(--ink)] mb-1">
                      Markdown Content
                    </label>
                    <textarea
                      id="skill-body"
                      rows={8}
                      value={skillDraftContent}
                      onChange={(e) => setSkillDraftContent(e.target.value)}
                      className="w-full p-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)] scrollbar-none"
                    />
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span
                      role={skillFeedback?.type === 'error' ? 'alert' : 'status'}
                      className={`text-xs ${skillFeedback?.type === 'error' ? 'text-[var(--flare)] font-medium' : 'text-[var(--jade)]'}`}
                    >
                      {skillFeedback?.text || ''}
                    </span>
                    <button
                      type="submit"
                      className="h-9 px-4 inline-flex items-center gap-1.5 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] text-white text-xs font-semibold cursor-pointer"
                    >
                      <Save className="w-3.5 h-3.5" />
                      <span>Save Skill</span>
                    </button>
                  </div>
                </form>
              </div>
            </section>

            {/* Pure-Python MCP Servers */}
            <section className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-5 sm:p-6 space-y-4">
              <div>
                <h2 className="font-display text-base font-semibold text-[var(--ink)] flex items-center gap-2">
                  <Server className="w-4 h-4 text-[var(--signal)]" />
                  <span>MCP Servers (/MCPs/servers.json)</span>
                </h2>
                <p className="text-xs text-[var(--ink-soft)] mt-0.5">
                  Attach external Model Context Protocol (MCP) tool servers directly to the pure-Python Agent-Core.
                </p>
              </div>

              {mcpServers.length > 0 ? (
                <div className="divide-y divide-[var(--rule)] border border-[var(--rule)] rounded-lg overflow-hidden">
                  {mcpServers.map((srv, idx) => (
                    <div
                      key={`${srv.name}-${idx}`}
                      className="px-4 py-3 flex flex-wrap items-center justify-between gap-3 bg-[var(--paper)]"
                    >
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-[var(--ink)]">{srv.name}</div>
                        <div className="font-mono text-xs text-[var(--ink-faint)] truncate">
                          {srv.url} ({srv.transport})
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => void handleToggleMcp(idx)}
                          className={`px-2.5 py-1 rounded-md text-xs font-medium border cursor-pointer ${
                            srv.enabled
                              ? 'border-[var(--jade)]/40 bg-[var(--jade)]/10 text-[var(--jade)]'
                              : 'border-[var(--rule)] text-[var(--ink-faint)]'
                          }`}
                        >
                          {srv.enabled ? 'Enabled' : 'Disabled'}
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleDeleteMcp(idx)}
                          aria-label={`Remove ${srv.name}`}
                          className="p-1.5 rounded-md text-[var(--ink-faint)] hover:text-[var(--flare)] cursor-pointer"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-[var(--ink-faint)]">
                  No external MCP servers configured yet. Add one below if you want the agent to discover remote MCP tools.
                </p>
              )}

              <form onSubmit={handleAddMcpServer} className="grid grid-cols-1 sm:grid-cols-4 gap-3 pt-2">
                <input
                  type="text"
                  value={newMcpName}
                  onChange={(e) => setNewMcpName(e.target.value)}
                  placeholder="Server name (e.g. github)"
                  aria-label="MCP server name"
                  className="h-9 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                />
                <input
                  type="url"
                  value={newMcpUrl}
                  onChange={(e) => setNewMcpUrl(e.target.value)}
                  placeholder="https://mcp.example.com/mcp"
                  aria-label="MCP server URL"
                  className="sm:col-span-2 h-9 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                />
                <div className="flex gap-2">
                  <select
                    value={newMcpTransport}
                    onChange={(e) => setNewMcpTransport(e.target.value as 'streamable_http' | 'sse')}
                    aria-label="MCP transport"
                    className="flex-1 h-9 px-2 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-xs text-[var(--ink)]"
                  >
                    <option value="streamable_http">HTTP</option>
                    <option value="sse">SSE</option>
                  </select>
                  <button
                    type="submit"
                    className="h-9 px-3 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] text-white text-xs font-semibold cursor-pointer"
                  >
                    Add
                  </button>
                </div>
              </form>
              {mcpFeedback && (
                <p
                  role={mcpFeedback.type === 'error' ? 'alert' : 'status'}
                  className={`text-xs ${mcpFeedback.type === 'error' ? 'text-[var(--flare)] font-medium' : 'text-[var(--jade)]'}`}
                >
                  {mcpFeedback.text}
                </p>
              )}
            </section>
          </div>
        )}

        {/* TAB 4: APPEARANCE & RUNTIME */}
        {activeTab === 'appearance' && (
          <div className="space-y-6 max-w-2xl">
            <section className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-6 space-y-4">
              <div>
                <h2 className="font-display text-base font-semibold text-[var(--ink)]">
                  Theme Appearance
                </h2>
                <p className="text-xs text-[var(--ink-soft)] mt-0.5">
                  Synced with the Nemofy brand light and dark palettes.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => onSetTheme('light')}
                  className={`p-4 rounded-xl border text-left transition-colors cursor-pointer ${
                    theme === 'light'
                      ? 'border-[var(--signal)] bg-[var(--blush)]'
                      : 'border-[var(--rule)] bg-[var(--paper)] hover:border-[var(--rule-strong)]'
                  }`}
                >
                  <Sun className="w-5 h-5 text-[var(--honey)] mb-2" />
                  <div className="text-sm font-semibold text-[var(--ink)]">Light Mode</div>
                  <div className="text-xs text-[var(--ink-soft)] mt-0.5">
                    Crisp paper canvas &amp; indigo signal
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => onSetTheme('dark')}
                  className={`p-4 rounded-xl border text-left transition-colors cursor-pointer ${
                    theme === 'dark'
                      ? 'border-[var(--signal)] bg-[var(--blush)]'
                      : 'border-[var(--rule)] bg-[var(--paper)] hover:border-[var(--rule-strong)]'
                  }`}
                >
                  <Moon className="w-5 h-5 text-[var(--signal)] mb-2" />
                  <div className="text-sm font-semibold text-[var(--ink)]">Dark Mode</div>
                  <div className="text-xs text-[var(--ink-soft)] mt-0.5">
                    Deep navy surface &amp; bright slate ink
                  </div>
                </button>
              </div>
            </section>

            <section className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-6 space-y-4">
              <div>
                <h2 className="font-display text-base font-semibold text-[var(--ink)]">
                  Python Agent-Core Backend
                </h2>
                <p className="text-xs text-[var(--ink-soft)] mt-0.5">
                  When deploying this Nemofy Agent web folder on Vercel, set your hosted Agent-Core server URL below (or leave blank when served from the unified server).
                </p>
              </div>

              <form onSubmit={handleSaveRemoteBackend} className="flex flex-col sm:flex-row gap-2.5">
                <label htmlFor="remote-backend-url" className="sr-only">
                  Remote Agent-Core Backend URL
                </label>
                <input
                  id="remote-backend-url"
                  type="url"
                  value={remoteUrl}
                  onChange={(e) => setRemoteUrl(e.target.value)}
                  placeholder="https://agent-api.yourdomain.com (leave empty for same-origin)"
                  className="flex-1 h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                />
                <button
                  type="submit"
                  className="h-10 px-4 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] text-white text-xs font-semibold cursor-pointer"
                >
                  Save Endpoint
                </button>
              </form>

              {remoteSaved && (
                <p className="text-xs text-[var(--jade)] inline-flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Backend endpoint preference updated.</span>
                </p>
              )}

              <div className="pt-3 border-t border-[var(--rule)] grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div>
                  <div className="text-[var(--ink-faint)]">Deployment Mode</div>
                  <div className="font-mono text-[var(--ink)] mt-0.5">
                    {isStandaloneMode() ? 'Standalone Frontend (Vercel)' : 'Full-Stack Server + Agent-Core'}
                  </div>
                </div>
                <div>
                  <div className="text-[var(--ink-faint)]">Python Sandbox Backend</div>
                  <div className="font-mono text-[var(--ink)] mt-0.5">
                    {health?.core?.sandbox?.backend ||
                      health?.core?.sandbox_backend ||
                      (health?.connected ? 'Connected' : 'Standby')}
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
