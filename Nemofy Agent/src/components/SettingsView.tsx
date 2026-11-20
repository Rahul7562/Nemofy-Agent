import {
  CheckCircle2,
  Cpu,
  Eye,
  EyeOff,
  KeyRound,
  Layers,
  Moon,
  Palette,
  Plus,
  Save,
  Server,
  Sparkles,
  Sun,
  Trash2,
  Wifi,
} from 'lucide-react';
import { type FormEvent, useEffect, useState } from 'react';
import {
  changeAdminPassword,
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
  saveSkillDocument,
  setRemoteBackendUrl,
} from '../lib/apiClient.ts';
import { validatePasswordChange } from '../lib/authValidation.ts';
import {
  LLM_PROVIDER_PRESETS,
  REASONING_EFFORT_OPTIONS,
  applyProviderPreset,
  getProviderPreset,
  validateLLMConfig,
} from '../lib/llmPresets.ts';
import type {
  AgentHealthResponse,
  MCPServerItem,
  PersistedLLMConfig,
  ProviderAvailability,
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
}

const SETTINGS_TABS: Array<{ id: SettingsTab; label: string; icon: typeof Cpu }> = [
  { id: 'llm', label: 'LLM Config', icon: Cpu },
  { id: 'password', label: 'Change Password', icon: KeyRound },
  { id: 'skills-mcps', label: 'Skills & MCPs', icon: Layers },
  { id: 'appearance', label: 'Appearance & Runtime', icon: Palette },
];

function RoleConfigCard({
  title,
  subtitle,
  roleKey,
  config,
  providers,
  onChange,
  onProbe,
  probeState,
}: {
  title: string;
  subtitle: string;
  roleKey: 'primary' | 'worker';
  config: RoleLLMConfig;
  providers: Record<string, ProviderAvailability>;
  onChange: (next: RoleLLMConfig) => void;
  onProbe: () => void;
  probeState: { loading: boolean; message?: string; ok?: boolean };
}) {
  const preset = getProviderPreset(config.provider);
  const avail = providers[config.provider];

  return (
    <section className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-5 sm:p-6 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="font-display text-base font-semibold text-[var(--ink)]">{title}</h2>
          <p className="text-xs text-[var(--ink-soft)] mt-0.5">{subtitle}</p>
        </div>
        <button
          type="button"
          disabled={probeState.loading}
          onClick={onProbe}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] text-xs font-medium text-[var(--ink)] transition-colors cursor-pointer shrink-0"
        >
          <Wifi className="w-3.5 h-3.5 text-[var(--signal)]" />
          <span>{probeState.loading ? 'Testing...' : 'Test'}</span>
        </button>
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
          value={config.provider}
          onChange={(e) => onChange(applyProviderPreset(config, e.target.value))}
          className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-sm text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
        >
          {LLM_PROVIDER_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        {avail && (
          <p className="mt-1.5 text-[11px] text-[var(--ink-faint)]">
            Server credentials ({avail.env_keys.join(' / ')}):{' '}
            <span className={avail.api_key_configured ? 'text-[var(--jade)] font-medium' : 'text-[var(--honey)] font-medium'}>
              {avail.requires_api_key
                ? avail.api_key_configured
                  ? 'Configured in server environment'
                  : 'Not set in server .env'
                : 'No API key required'}
            </span>
          </p>
        )}
      </div>

      {/* Model ID */}
      <div>
        <label
          htmlFor={`${roleKey}-model`}
          className="block text-xs font-medium text-[var(--ink)] mb-1.5"
        >
          Model Identifier
        </label>
        <input
          id={`${roleKey}-model`}
          type="text"
          value={config.model}
          onChange={(e) => onChange({ ...config, model: e.target.value })}
          placeholder="e.g. deepseek-ai/DeepSeek-V4.1-Flash"
          className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
        />
        {preset && preset.defaultModels.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {preset.defaultModels.map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => onChange({ ...config, model: m })}
                className={`px-2 py-1 rounded-md border text-[11px] font-mono transition-colors cursor-pointer ${
                  config.model === m
                    ? 'border-[var(--signal)] bg-[var(--blush)] text-[var(--signal)]'
                    : 'border-[var(--rule)] bg-[var(--paper-sunk)] text-[var(--ink-soft)] hover:text-[var(--ink)]'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        )}
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
          placeholder="https://api.openai.com/v1"
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
            value={config.temperature}
            onChange={(e) => onChange({ ...config, temperature: Number(e.target.value) })}
            className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
          />
        </div>
        <div>
          <label
            htmlFor={`${roleKey}-tokens`}
            className="block text-xs font-medium text-[var(--ink)] mb-1.5"
          >
            Max Tokens
          </label>
          <input
            id={`${roleKey}-tokens`}
            type="number"
            step="1"
            min={128}
            max={128000}
            value={config.max_tokens}
            onChange={(e) => onChange({ ...config, max_tokens: Number(e.target.value) })}
            className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
          />
        </div>
        <div>
          <label
            htmlFor={`${roleKey}-reasoning`}
            className="block text-xs font-medium text-[var(--ink)] mb-1.5"
          >
            Reasoning Effort
          </label>
          <select
            id={`${roleKey}-reasoning`}
            value={config.reasoning_effort}
            onChange={(e) => onChange({ ...config, reasoning_effort: e.target.value })}
            className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
          >
            {REASONING_EFFORT_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
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
}: SettingsViewProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>(initialTab);

  // LLM Config State
  const [llmConfig, setLlmConfig] = useState<PersistedLLMConfig | null>(null);
  const [providers, setProviders] = useState<Record<string, ProviderAvailability>>({});
  const [llmSaving, setLlmSaving] = useState(false);
  const [llmMessage, setLlmMessage] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);
  const [primaryProbe, setPrimaryProbe] = useState<{ loading: boolean; message?: string; ok?: boolean }>({
    loading: false,
  });
  const [workerProbe, setWorkerProbe] = useState<{ loading: boolean; message?: string; ok?: boolean }>({
    loading: false,
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
  const [skillFeedback, setSkillFeedback] = useState<string | null>(null);

  const [mcpServers, setMcpServers] = useState<MCPServerItem[]>([]);
  const [newMcpName, setNewMcpName] = useState('');
  const [newMcpUrl, setNewMcpUrl] = useState('');
  const [newMcpTransport, setNewMcpTransport] = useState<'streamable_http' | 'sse'>('streamable_http');
  const [mcpFeedback, setMcpFeedback] = useState<string | null>(null);

  // Runtime & Remote URL State
  const [health, setHealth] = useState<AgentHealthResponse | null>(null);
  const [remoteUrl, setRemoteUrl] = useState(() => getRemoteBackendUrl());
  const [remoteSaved, setRemoteSaved] = useState(false);

  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const [llmRes, skillsRes, mcpsRes, healthRes] = await Promise.all([
          fetchLLMSettings(),
          fetchSkills(),
          fetchMCPServers(),
          fetchAgentHealth(),
        ]);
        if (!mounted) return;
        setLlmConfig(llmRes.config);
        setProviders(llmRes.providers);
        setSkills(skillsRes);
        if (skillsRes.length > 0) {
          setSelectedSkillName(skillsRes[0].name);
          setSkillDraftName(skillsRes[0].name);
          setSkillDraftContent(skillsRes[0].content);
        }
        setMcpServers(mcpsRes);
        setHealth(healthRes);
      } catch {
        // handled per section
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  async function handleSaveLLM(e: FormEvent) {
    e.preventDefault();
    if (!llmConfig) return;
    setLlmMessage(null);
    const check = validateLLMConfig(llmConfig);
    if (!check.valid) {
      setLlmMessage({ type: 'error', text: check.error || 'Invalid configuration.' });
      return;
    }

    setLlmSaving(true);
    try {
      const saved = await saveLLMSettings(llmConfig);
      setLlmConfig(saved.config);
      setProviders(saved.providers);
      onLLMConfigSaved(saved.config);
      setLlmMessage({
        type: 'ok',
        text: 'Main Agent and Sub-Agent LLM configuration saved and synced to Agent-Core.',
      });
    } catch (err) {
      setLlmMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to save LLM configuration.',
      });
    } finally {
      setLlmSaving(false);
    }
  }

  async function handleProbe(role: 'primary' | 'worker') {
    if (!llmConfig) return;
    const setter = role === 'primary' ? setPrimaryProbe : setWorkerProbe;
    setter({ loading: true });
    try {
      const res = await probeLLMModel(role, llmConfig[role]);
      const ok = Boolean(res.ok);
      setter({
        loading: false,
        ok,
        message: ok
          ? `Connected to ${String(res.provider || llmConfig[role].provider)} (${String(res.model || llmConfig[role].model)}).`
          : String(res.error || res.detail || 'Probe failed. Check provider key or endpoint.'),
      });
    } catch (err) {
      setter({
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
      setSkillFeedback(`Saved /Skills/${saved.filename}`);
    } catch (err) {
      setSkillFeedback(err instanceof Error ? err.message : 'Unable to save skill.');
    }
  }

  async function handleDeleteSkill(name: string) {
    await deleteSkillDocument(name);
    const refreshed = await fetchSkills();
    setSkills(refreshed);
    if (selectedSkillName === name) {
      setSelectedSkillName(refreshed[0]?.name || null);
      setSkillDraftName(refreshed[0]?.name || '');
      setSkillDraftContent(refreshed[0]?.content || '');
    }
    setSkillFeedback(`Deleted skill "${name}"`);
  }

  async function handleAddMcpServer(e: FormEvent) {
    e.preventDefault();
    if (!newMcpName.trim() || !newMcpUrl.trim()) return;
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
    const saved = await saveMCPServers(next);
    setMcpServers(saved);
    setNewMcpName('');
    setNewMcpUrl('');
    setMcpFeedback('MCP server registry saved to /MCPs/servers.json.');
  }

  async function handleToggleMcp(index: number) {
    const next = mcpServers.map((s, idx) => (idx === index ? { ...s, enabled: !s.enabled } : s));
    const saved = await saveMCPServers(next);
    setMcpServers(saved);
  }

  async function handleDeleteMcp(index: number) {
    const next = mcpServers.filter((_, idx) => idx !== index);
    const saved = await saveMCPServers(next);
    setMcpServers(saved);
  }

  function handleSaveRemoteBackend(e: FormEvent) {
    e.preventDefault();
    setRemoteBackendUrl(remoteUrl);
    setRemoteSaved(true);
    setTimeout(() => setRemoteSaved(false), 3000);
  }

  return (
    <div className="h-full overflow-y-auto bg-[var(--paper)]">
      <div className="max-w-5xl mx-auto px-5 sm:px-8 py-6 sm:py-8 space-y-6">
        {/* Header */}
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight text-[var(--ink)]">
            Settings
          </h1>
          <p className="mt-1 text-sm text-[var(--ink-soft)]">
            Configure Main Agent &amp; Sub-Agent LLMs, administrator security, dynamic skills, and MCP servers.
          </p>
        </div>

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
          <form onSubmit={handleSaveLLM} noValidate className="space-y-5">
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
                    title="Main Agent (Brain)"
                    subtitle="Handles planning, reasoning, tool selection, and final responses."
                    roleKey="primary"
                    config={llmConfig.primary}
                    providers={providers}
                    onChange={(next) => setLlmConfig({ ...llmConfig, primary: next })}
                    onProbe={() => void handleProbe('primary')}
                    probeState={primaryProbe}
                  />
                  <RoleConfigCard
                    title="Sub-Agent (Worker)"
                    subtitle="Executes delegated research, coding, and isolated sub-tasks."
                    roleKey="worker"
                    config={llmConfig.worker}
                    providers={providers}
                    onChange={(next) => setLlmConfig({ ...llmConfig, worker: next })}
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
                      value={llmConfig.max_turns}
                      onChange={(e) =>
                        setLlmConfig({ ...llmConfig, max_turns: Number(e.target.value) })
                      }
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
              </>
            ) : (
              <div className="text-sm text-[var(--ink-faint)]">Loading LLM configuration...</div>
            )}
          </form>
        )}

        {/* TAB 2: CHANGE PASSWORD */}
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
                      className="w-full p-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                    />
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs text-[var(--jade)]">{skillFeedback || ''}</span>
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
              {mcpFeedback && <p className="text-xs text-[var(--jade)]">{mcpFeedback}</p>}
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
                    {health?.core?.sandbox_backend || (health?.connected ? 'Connected' : 'Standby')}
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
