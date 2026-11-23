import {
  AlertTriangle,
  Check,
  ChevronDown,
  FolderTree,
  Menu,
  PanelLeftOpen,
  Server,
  SlidersHorizontal,
  SquarePen,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import {
  LLM_PROVIDER_PRESETS,
  canonicalizeProviderId,
  getProviderDisplayName,
} from '../lib/llmPresets.ts';
import type {
  ContextUsageSnapshot,
  PersistedLLMConfig,
  ProviderAvailability,
  ProviderModelOption,
  ThreadSessionConfig,
} from '../types/agent.ts';
import { ContextUsageRing } from './ContextUsageRing.tsx';
import { ModelSelectorCombobox } from './ModelSelectorCombobox.tsx';

interface ChatHeaderProps {
  sidebarCollapsed: boolean;
  onExpandSidebar: () => void;
  onOpenMobileSidebar: () => void;
  onNewThread: () => void;
  llmConfig: PersistedLLMConfig | null;
  sessionConfig: ThreadSessionConfig | null;
  providers: Record<string, ProviderAvailability>;
  onQuickSwitchSessionProvider: (nextProvider: string) => void;
  onQuickSwitchPrimaryModel: (
    nextModel: string,
    option?: ProviderModelOption,
    options?: { persist?: boolean },
  ) => void;
  onSetManualContextWindow?: (manualWindow: number | null) => Promise<void> | void;
  onRefreshContextMetadata?: () => Promise<void> | void;
  onOpenLLMSettings: () => void;
  contextUsage: ContextUsageSnapshot;
  workspaceOpen: boolean;
  onToggleWorkspace: () => void;
  workspaceChangesCount: number;
}

export function ChatHeader({
  sidebarCollapsed,
  onExpandSidebar,
  onOpenMobileSidebar,
  onNewThread,
  llmConfig,
  sessionConfig,
  providers,
  onQuickSwitchSessionProvider,
  onQuickSwitchPrimaryModel,
  onSetManualContextWindow,
  onRefreshContextMetadata,
  onOpenLLMSettings,
  contextUsage,
  workspaceOpen,
  onToggleWorkspace,
  workspaceChangesCount,
}: ChatHeaderProps) {
  const [providerPopoverOpen, setProviderPopoverOpen] = useState(false);
  const [modelPopoverOpen, setModelPopoverOpen] = useState(false);
  const providerPopoverRef = useRef<HTMLDivElement>(null);
  const modelPopoverRef = useRef<HTMLDivElement>(null);

  const customProviders = llmConfig?.custom_providers || [];
  const activeProvider = canonicalizeProviderId(
    sessionConfig?.provider || llmConfig?.primary.provider || 'river_ai',
  );
  const activeModel = (sessionConfig?.model ?? llmConfig?.primary.model ?? '').trim();
  const activeBaseUrl = sessionConfig?.base_url ?? llmConfig?.primary.base_url ?? '';
  const activeManualWindow =
    sessionConfig !== null
      ? sessionConfig.manual_context_window ?? null
      : llmConfig?.primary.manual_context_window ?? null;

  const hasPrimaryModel = activeModel.length > 0;
  const providerLabel = getProviderDisplayName(activeProvider, customProviders, providers);
  const activeCustomProvider = customProviders.find((p) => p.id === activeProvider) || null;
  const activeAvail = providers[activeProvider];
  const providerNeedsKey = Boolean(
    activeAvail && activeAvail.requires_api_key && !activeAvail.api_key_configured,
  );
  const providerDisabled = Boolean(activeCustomProvider && !activeCustomProvider.enabled);

  useEffect(() => {
    if (!providerPopoverOpen && !modelPopoverOpen) return;
    function handleOutside(e: MouseEvent) {
      const target = e.target as Node;
      if (
        providerPopoverOpen &&
        providerPopoverRef.current &&
        !providerPopoverRef.current.contains(target)
      ) {
        setProviderPopoverOpen(false);
      }
      if (
        modelPopoverOpen &&
        modelPopoverRef.current &&
        !modelPopoverRef.current.contains(target)
      ) {
        setModelPopoverOpen(false);
      }
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setProviderPopoverOpen(false);
        setModelPopoverOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('keydown', handleKey);
    };
  }, [providerPopoverOpen, modelPopoverOpen]);

  const builtinPresets = LLM_PROVIDER_PRESETS.filter(
    (p) => p.id !== 'custom' || activeProvider === 'custom',
  );

  function renderProviderStatus(providerId: string, enabled = true) {
    if (!enabled) {
      return <span className="text-[11px] text-[var(--ink-faint)]">Disabled</span>;
    }
    const avail = providers[providerId];
    if (!avail) return null;
    if (!avail.requires_api_key) {
      return <span className="text-[11px] text-[var(--ink-faint)]">Local</span>;
    }
    if (avail.api_key_configured) {
      return <span className="text-[11px] text-[var(--jade)]">Ready</span>;
    }
    return <span className="text-[11px] text-[var(--honey)]">Key needed</span>;
  }

  return (
    <header className="h-14 px-3 sm:px-5 flex items-center justify-between gap-2.5 bg-[var(--paper)]/85 backdrop-blur-sm border-b border-[var(--rule)]/70 shrink-0 z-20">
      {/* Left: Sidebar trigger + Inline Session Provider & Model Switchers */}
      <div className="flex flex-1 items-center gap-1.5 min-w-0">
        {/* Mobile sidebar drawer trigger */}
        <button
          type="button"
          onClick={onOpenMobileSidebar}
          aria-label="Open sidebar"
          className="lg:hidden h-9 w-9 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer shrink-0"
        >
          <Menu className="w-4 h-4" />
        </button>

        {/* Desktop sidebar expand & quick new-chat buttons when collapsed */}
        {sidebarCollapsed && (
          <div className="hidden lg:flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={onExpandSidebar}
              aria-label="Expand sidebar"
              title="Expand sidebar"
              className="h-9 w-9 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer"
            >
              <PanelLeftOpen className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={onNewThread}
              aria-label="New chat"
              title="New chat"
              className="h-9 w-9 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer"
            >
              <SquarePen className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Inline Provider Selector (Session-Scoped) */}
        <div ref={providerPopoverRef} className="relative shrink-0 max-w-[44%] sm:max-w-[200px]">
          <button
            type="button"
            data-testid="active-provider-selector"
            onClick={() => {
              setModelPopoverOpen(false);
              setProviderPopoverOpen((v) => !v);
            }}
            aria-haspopup="listbox"
            aria-expanded={providerPopoverOpen}
            title={`Session provider: ${providerLabel}${providerNeedsKey ? ' (API key not configured)' : ''}`}
            className={`h-9 px-2.5 inline-flex items-center gap-1.5 rounded-lg border text-xs font-medium transition-colors cursor-pointer max-w-full ${
              providerPopoverOpen
                ? 'border-[var(--signal)] bg-[var(--paper-sunk)] text-[var(--ink)]'
                : providerNeedsKey || providerDisabled
                  ? 'border-[var(--honey)]/55 bg-[var(--honey)]/8 text-[var(--ink)] hover:border-[var(--honey)]'
                  : 'border-[var(--rule)] bg-[var(--paper-sunk)]/75 text-[var(--ink-soft)] hover:text-[var(--ink)] hover:border-[var(--rule-strong)]'
            }`}
          >
            {providerNeedsKey || providerDisabled ? (
              <AlertTriangle className="w-3.5 h-3.5 text-[var(--honey)] shrink-0" />
            ) : (
              <Server className="w-3.5 h-3.5 text-[var(--ink-faint)] shrink-0" />
            )}
            <span className="truncate">{providerLabel}</span>
            <ChevronDown
              className={`w-3.5 h-3.5 text-[var(--ink-faint)] shrink-0 transition-transform ${
                providerPopoverOpen ? 'rotate-180' : ''
              }`}
            />
          </button>

          {providerPopoverOpen && (
            <div
              role="dialog"
              aria-label="Select session provider"
              data-testid="provider-selector-popover"
              className="absolute left-0 top-full mt-2 w-[300px] sm:w-[320px] max-w-[calc(100vw-1.5rem)] rounded-2xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-xl overflow-hidden z-40"
            >
              <div className="px-3.5 py-2.5 border-b border-[var(--rule)] bg-[var(--paper-sunk)] flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="font-display text-xs font-semibold text-[var(--ink)]">
                    Session Provider
                  </div>
                  <div className="text-[11px] text-[var(--ink-faint)] truncate">
                    Applies only to this conversation
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setProviderPopoverOpen(false);
                    onOpenLLMSettings();
                  }}
                  className="text-[11px] font-medium text-[var(--signal)] hover:underline cursor-pointer shrink-0"
                >
                  Manage
                </button>
              </div>

              <div className="max-h-72 overflow-y-auto scrollbar-none py-1.5">
                {customProviders.length > 0 && (
                  <div className="pb-1.5 mb-1.5 border-b border-[var(--rule)]">
                    <div className="px-3.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--ink-faint)]">
                      Custom Providers
                    </div>
                    {customProviders.map((cp) => {
                      const selected = cp.id === activeProvider;
                      return (
                        <button
                          key={cp.id}
                          type="button"
                          disabled={!cp.enabled}
                          data-testid={`provider-option-${cp.id}`}
                          onClick={() => {
                            onQuickSwitchSessionProvider(cp.id);
                            setProviderPopoverOpen(false);
                          }}
                          className={`w-full px-3.5 py-2 text-left flex items-center justify-between gap-2 text-xs transition-colors ${
                            !cp.enabled
                              ? 'opacity-50 cursor-not-allowed text-[var(--ink-faint)]'
                              : selected
                                ? 'bg-[var(--blush)] text-[var(--signal)] font-medium cursor-pointer'
                                : 'text-[var(--ink)] hover:bg-[var(--paper-sunk)] cursor-pointer'
                          }`}
                        >
                          <div className="min-w-0">
                            <div className="truncate">{cp.name}</div>
                            <div className="font-mono text-[10px] text-[var(--ink-faint)] truncate">
                              {cp.protocol === 'anthropic_compatible' ? 'Anthropic' : 'OpenAI'} · {cp.models.length}{' '}
                              {cp.models.length === 1 ? 'model' : 'models'}
                            </div>
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            {renderProviderStatus(cp.id, cp.enabled)}
                            {selected && <Check className="w-3.5 h-3.5 text-[var(--signal)]" />}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}

                <div className="px-3.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--ink-faint)]">
                  Built-in Providers
                </div>
                {builtinPresets.map((preset) => {
                  const selected = preset.id === activeProvider;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      data-testid={`provider-option-${preset.id}`}
                      onClick={() => {
                        onQuickSwitchSessionProvider(preset.id);
                        setProviderPopoverOpen(false);
                      }}
                      className={`w-full px-3.5 py-2 text-left flex items-center justify-between gap-2 text-xs transition-colors cursor-pointer ${
                        selected
                          ? 'bg-[var(--blush)] text-[var(--signal)] font-medium'
                          : 'text-[var(--ink)] hover:bg-[var(--paper-sunk)]'
                      }`}
                    >
                      <span className="truncate">{preset.label}</span>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {renderProviderStatus(preset.id, true)}
                        {selected && <Check className="w-3.5 h-3.5 text-[var(--signal)]" />}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <span className="text-[var(--ink-faint)]/60 text-xs select-none hidden sm:inline">/</span>

        {/* Inline Primary Model Picker (Session-Scoped) */}
        <div ref={modelPopoverRef} className="relative min-w-0 max-w-full">
          <button
            type="button"
            data-testid="active-primary-model-button"
            onClick={() => {
              setProviderPopoverOpen(false);
              setModelPopoverOpen((v) => !v);
            }}
            aria-haspopup="dialog"
            aria-expanded={modelPopoverOpen}
            title={
              hasPrimaryModel
                ? `Session Model: ${providerLabel} / ${activeModel}`
                : 'Select model for this session'
            }
            className={`h-9 px-2 sm:px-2.5 inline-flex items-center gap-1.5 sm:gap-2 rounded-xl text-left transition-colors cursor-pointer max-w-full sm:max-w-[280px] ${
              modelPopoverOpen
                ? 'bg-[var(--paper-sunk)] text-[var(--ink)]'
                : hasPrimaryModel
                  ? 'text-[var(--ink)] hover:bg-[var(--paper-sunk)]'
                  : 'bg-[var(--blush)]/70 text-[var(--signal)] hover:bg-[var(--blush)]'
            }`}
          >
            <span className="font-display font-semibold text-sm tracking-tight truncate min-w-0">
              {hasPrimaryModel ? activeModel : 'Select model'}
            </span>
            <ChevronDown
              className={`w-3.5 h-3.5 text-[var(--ink-faint)] shrink-0 transition-transform ${
                modelPopoverOpen ? 'rotate-180' : ''
              }`}
            />
          </button>

          {modelPopoverOpen && (
            <div
              role="dialog"
              aria-label="Quick switch session model"
              className="absolute left-0 top-full mt-2 w-[320px] sm:w-[376px] max-w-[calc(100vw-1.5rem)] rounded-2xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-xl p-3.5 space-y-3 z-40"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="font-display text-xs font-semibold text-[var(--ink)]">
                    Session Model
                  </div>
                  <div className="text-[11px] text-[var(--ink-faint)] truncate">
                    {providerLabel}
                    {providerNeedsKey ? ' · API key missing' : ''}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setModelPopoverOpen(false);
                    onOpenLLMSettings();
                  }}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-[var(--signal)] hover:bg-[var(--blush)] transition-colors cursor-pointer shrink-0 whitespace-nowrap"
                >
                  <SlidersHorizontal className="w-3.5 h-3.5" />
                  <span>Full LLM Settings</span>
                </button>
              </div>

              {providerNeedsKey && (
                <div className="rounded-lg border border-[var(--honey)]/40 bg-[var(--honey)]/10 px-3 py-2 text-[11px] text-[var(--ink-soft)] flex items-center justify-between gap-2">
                  <span>API key for {providerLabel} is not configured yet.</span>
                  <button
                    type="button"
                    onClick={() => {
                      setModelPopoverOpen(false);
                      onOpenLLMSettings();
                    }}
                    className="text-[var(--signal)] font-medium hover:underline cursor-pointer shrink-0"
                  >
                    Add key
                  </button>
                </div>
              )}

              {llmConfig ? (
                <ModelSelectorCombobox
                  id="header-quick-primary-model"
                  role="primary"
                  provider={activeProvider}
                  providerLabelOverride={providerLabel}
                  customModels={activeCustomProvider?.models}
                  manualContextWindow={activeManualWindow}
                  baseUrl={activeBaseUrl}
                  apiKeyDraft=""
                  value={activeModel}
                  onChange={(nextModel) => {
                    onQuickSwitchPrimaryModel(nextModel, undefined, { persist: false });
                  }}
                  onSelectOption={(option) => {
                    onQuickSwitchPrimaryModel(option.id, option, { persist: true });
                    setModelPopoverOpen(false);
                  }}
                  onModelResolved={(resolution) => {
                    onQuickSwitchPrimaryModel(
                      resolution.model,
                      {
                        id: resolution.matched_model || resolution.model,
                        name: resolution.matched_model || resolution.model,
                        context_length: resolution.context_window,
                        max_output_tokens: resolution.max_output_tokens,
                      },
                      { persist: true },
                    );
                  }}
                />
              ) : (
                <div className="py-3 text-xs text-[var(--ink-faint)]">
                  Loading model configuration...
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right: Context Usage Ring + Workspace Drawer Toggle */}
      <div className="flex items-center gap-2 shrink-0">
        <ContextUsageRing
          usage={contextUsage}
          modelConfigured={hasPrimaryModel}
          onOpenLLMSettings={onOpenLLMSettings}
          onSetManualContextWindow={onSetManualContextWindow}
          onRefreshMetadata={onRefreshContextMetadata}
        />

        <button
          type="button"
          data-testid="workspace-drawer-toggle"
          onClick={onToggleWorkspace}
          aria-pressed={workspaceOpen}
          aria-label={workspaceOpen ? 'Close workspace files drawer' : 'Open workspace files drawer'}
          title={workspaceOpen ? 'Close /workspace drawer' : 'Inspect /workspace files'}
          className={`relative h-9 px-2.5 inline-flex items-center gap-1.5 rounded-lg border text-xs font-medium transition-colors cursor-pointer shrink-0 whitespace-nowrap ${
            workspaceOpen
              ? 'border-[var(--signal)] bg-[var(--blush)]/75 text-[var(--signal)]'
              : 'border-[var(--rule)] bg-[var(--paper-sunk)] hover:border-[var(--rule-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)]'
          }`}
        >
          <FolderTree className="w-4 h-4 shrink-0" />
          <span className="hidden sm:inline">Workspace</span>
          {workspaceChangesCount > 0 && (
            <span
              className="ml-0.5 inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-[var(--signal)] text-white font-mono text-[10px] font-semibold tabular-nums"
              title={`${workspaceChangesCount} workspace file ${workspaceChangesCount === 1 ? 'change' : 'changes'} in this chat`}
            >
              {workspaceChangesCount}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
