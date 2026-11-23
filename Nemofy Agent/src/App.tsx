import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChatView } from './components/ChatView.tsx';
import { LoginGate } from './components/LoginGate.tsx';
import { NemofyLogo } from './components/NemofyLogo.tsx';
import { SettingsModal } from './components/SettingsModal.tsx';
import { Sidebar } from './components/Sidebar.tsx';
import { WorkspaceDrawer } from './components/WorkspaceDrawer.tsx';
import { useAgentExecution } from './hooks/useAgentExecution.ts';
import { useChatThreads } from './hooks/useChatThreads.ts';
import {
  checkAuthSession,
  fetchLLMSettings,
  logoutSession,
  saveLLMSettings,
  updateAccessMode,
  updateThreadSessionConfig,
} from './lib/apiClient.ts';
import { resolveModelLimits } from './lib/contextUtils.ts';
import {
  applyProviderPreset,
  canonicalizeProviderId,
} from './lib/llmPresets.ts';
import { getThinkingProfile, normalizeThinkingLevel } from './lib/thinkingModes.ts';
import type {
  ExecutionAccessMode,
  PersistedLLMConfig,
  ProviderAvailability,
  ProviderModelOption,
  SettingsTab,
  ThemeMode,
  ThreadSessionConfig,
} from './types/agent.ts';

function getInitialTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'light';
  const saved = window.localStorage.getItem('nemofy_theme');
  if (saved === 'dark' || saved === 'light') return saved;
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

function getInitialSidebarCollapsed(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage.getItem('nemofy_sidebar_collapsed') === 'true';
  } catch {
    return false;
  }
}

function normalizeAccessMode(raw: unknown): ExecutionAccessMode {
  return raw === 'full_access' ? 'full_access' : 'default';
}

function buildDefaultSessionConfig(
  llmConfig: PersistedLLMConfig | null,
  fallbackAccessMode: ExecutionAccessMode,
): ThreadSessionConfig | null {
  if (!llmConfig) return null;
  return {
    provider: canonicalizeProviderId(llmConfig.primary.provider),
    model: llmConfig.primary.model,
    base_url: llmConfig.primary.base_url,
    temperature: llmConfig.primary.temperature,
    max_tokens: llmConfig.primary.max_tokens,
    reasoning_effort: llmConfig.primary.reasoning_effort,
    context_window: llmConfig.primary.context_window ?? null,
    context_window_source: llmConfig.primary.context_window_source ?? null,
    manual_context_window: llmConfig.primary.manual_context_window ?? null,
    max_output_tokens: llmConfig.primary.max_output_tokens ?? null,
    access_mode: normalizeAccessMode(llmConfig.access_mode ?? fallbackAccessMode),
  };
}

export function App() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [authChecked, setAuthChecked] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const [needsSetup, setNeedsSetup] = useState(false);
  const [username, setUsername] = useState('admin');

  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(getInitialSidebarCollapsed);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const [workspaceDrawerOpen, setWorkspaceDrawerOpen] = useState(false);
  const [workspaceTargetPath, setWorkspaceTargetPath] = useState<string | null>(null);
  const [workspaceRefreshKey, setWorkspaceRefreshKey] = useState(0);

  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('llm');

  const [llmConfig, setLlmConfig] = useState<PersistedLLMConfig | null>(null);
  const [providers, setProviders] = useState<Record<string, ProviderAvailability>>({});
  const [accessMode, setAccessMode] = useState<ExecutionAccessMode>('default');
  // Draft session config used only when no thread is selected yet
  const [draftSessionConfig, setDraftSessionConfig] = useState<Partial<ThreadSessionConfig> | null>(null);

  const {
    threads,
    activeThread,
    activeThreadId,
    setActiveThreadId,
    loadingThreads,
    threadsError,
    createNewThread,
    removeThread,
    upsertThread,
    updateThreadById,
  } = useChatThreads(authenticated);

  const activeSessionConfig = useMemo<ThreadSessionConfig | null>(() => {
    if (activeThread?.sessionConfig) {
      return activeThread.sessionConfig;
    }
    const base = buildDefaultSessionConfig(llmConfig, accessMode);
    if (!base) return null;
    return draftSessionConfig ? { ...base, ...draftSessionConfig } : base;
  }, [activeThread, llmConfig, accessMode, draftSessionConfig]);

  const effectiveAccessMode = activeSessionConfig?.access_mode ?? accessMode;

  const handleWorkspaceFilesChanged = useCallback(() => {
    setWorkspaceRefreshKey((k) => k + 1);
  }, []);

  const handleCreateThreadWithDraft = useCallback(
    async (title?: string) => {
      const seed = draftSessionConfig ?? undefined;
      const created = await createNewThread(title, seed);
      if (created) {
        setDraftSessionConfig(null);
      }
      return created;
    },
    [createNewThread, draftSessionConfig],
  );

  const {
    runningThreadIds,
    getThreadRun,
    getThreadDraft,
    setThreadPrompt,
    attachFilesToThread,
    removeAttachmentFromThread,
    sendMessage,
    submitApprovalDecision,
    stopThread,
    cancelAndClearThread,
  } = useAgentExecution({
    authenticated,
    onCreateThread: handleCreateThreadWithDraft,
    onUpsertThread: upsertThread,
    onUpdateThreadById: updateThreadById,
    onWorkspaceFilesChanged: handleWorkspaceFilesChanged,
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    try {
      window.localStorage.setItem('nemofy_theme', theme);
    } catch {
      // ignore storage errors
    }
  }, [theme]);

  useEffect(() => {
    try {
      window.localStorage.setItem('nemofy_sidebar_collapsed', String(sidebarCollapsed));
    } catch {
      // ignore storage errors
    }
  }, [sidebarCollapsed]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      const session = await checkAuthSession();
      if (!mounted) return;
      setAuthenticated(Boolean(session.authenticated));
      setNeedsSetup(Boolean(session.needsSetup));
      if (session.username) setUsername(session.username);
      setAuthChecked(true);
    })();
    return () => {
      mounted = false;
    };
  }, []);

  const reloadLLMSettings = useCallback(async () => {
    try {
      const res = await fetchLLMSettings();
      setLlmConfig(res.config);
      setProviders(res.providers || {});
      setAccessMode(normalizeAccessMode(res.config.access_mode));
    } catch {
      // non-blocking
    }
  }, []);

  useEffect(() => {
    if (!authenticated) return;
    void reloadLLMSettings();
  }, [authenticated, reloadLLMSettings]);

  function toggleTheme() {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }

  function openSettings(tab: SettingsTab = 'llm') {
    setSettingsTab(tab);
    setSettingsModalOpen(true);
  }

  const handleOpenWorkspaceFile = useCallback((path: string) => {
    setWorkspaceTargetPath(path);
    setWorkspaceDrawerOpen(true);
    setWorkspaceRefreshKey((k) => k + 1);
  }, []);

  async function handleLogout() {
    await logoutSession();
    setAuthenticated(false);
    setLlmConfig(null);
    setProviders({});
    setAccessMode('default');
    setDraftSessionConfig(null);
    setSettingsModalOpen(false);
    setWorkspaceDrawerOpen(false);
    setMobileSidebarOpen(false);
  }

  async function handleDeleteThread(id: string) {
    await cancelAndClearThread(id);
    await removeThread(id);
  }

  // Per-thread sequence guards prevent out-of-order concurrent PATCH responses from clobbering newer edits
  const sessionSwitchSeqRef = useRef<Map<string, number>>(new Map());

  function nextSessionSeq(targetKey: string): number {
    const next = (sessionSwitchSeqRef.current.get(targetKey) || 0) + 1;
    sessionSwitchSeqRef.current.set(targetKey, next);
    return next;
  }

  async function patchActiveSession(
    patch: Partial<ThreadSessionConfig> & { force_refresh?: boolean },
    options?: { persist?: boolean; optimistic?: Partial<ThreadSessionConfig> },
  ) {
    const shouldPersist = options?.persist !== false;
    const optimistic = options?.optimistic ?? patch;

    if (!activeThreadId) {
      setDraftSessionConfig((prev) => ({ ...(prev || {}), ...optimistic }));
      return;
    }

    const targetThreadId = activeThreadId;
    const seq = nextSessionSeq(targetThreadId);

    // Optimistically update ONLY this specific thread; never touch other threads or global llmConfig
    updateThreadById(targetThreadId, (thread) => {
      const base = thread.sessionConfig || buildDefaultSessionConfig(llmConfig, accessMode);
      if (!base) return thread;
      return {
        ...thread,
        sessionConfig: {
          ...base,
          ...optimistic,
        },
      };
    });

    if (!shouldPersist) return;

    try {
      const updated = await updateThreadSessionConfig(targetThreadId, patch);
      if (sessionSwitchSeqRef.current.get(targetThreadId) === seq) {
        upsertThread(updated, false);
      }
    } catch {
      // Keep optimistic state; user can inspect or retry
    }
  }

  async function handleQuickSwitchSessionProvider(nextProviderId: string) {
    if (!llmConfig || !activeSessionConfig) return;
    const customProviders = llmConfig.custom_providers || [];
    const applied = applyProviderPreset(
      {
        provider: activeSessionConfig.provider,
        model: activeSessionConfig.model,
        base_url: activeSessionConfig.base_url,
        temperature: activeSessionConfig.temperature,
        max_tokens: activeSessionConfig.max_tokens,
        reasoning_effort: activeSessionConfig.reasoning_effort,
        context_window: activeSessionConfig.context_window,
        context_window_source: activeSessionConfig.context_window_source,
        manual_context_window: activeSessionConfig.manual_context_window,
        max_output_tokens: activeSessionConfig.max_output_tokens,
      },
      nextProviderId,
      {
        clearModelOnProviderSwitch: false,
        customProviders,
      },
    );

    const customProv = customProviders.find((p) => p.id === applied.provider);
    const matchedCustomModel = customProv?.models.find(
      (m) => m.id.toLowerCase() === applied.model.trim().toLowerCase(),
    );
    const resolved = applied.model.trim()
      ? resolveModelLimits(
          applied.provider,
          applied.model.trim(),
          matchedCustomModel?.context_window ?? applied.context_window ?? null,
          matchedCustomModel?.max_output_tokens ?? applied.max_output_tokens ?? null,
          {
            explicitSource: matchedCustomModel?.context_window ? 'custom_provider_config' : null,
          },
        )
      : null;

    const patch: Partial<ThreadSessionConfig> = {
      provider: applied.provider,
      base_url: applied.base_url,
      model: applied.model,
      context_window: resolved?.contextWindow ?? null,
      context_window_source: resolved?.contextWindowSource ?? 'unknown',
      manual_context_window: null,
      max_output_tokens: resolved?.maxOutputTokens ?? null,
      max_tokens: resolved?.maxOutputTokens ?? applied.max_tokens,
      reasoning_effort: applied.reasoning_effort,
    };

    await patchActiveSession(patch, { persist: true });
  }

  async function handleQuickSwitchPrimaryModel(
    nextModel: string,
    option?: ProviderModelOption,
    options?: { persist?: boolean },
  ) {
    if (!llmConfig || !activeSessionConfig) return;
    const shouldPersist = options?.persist !== false;
    const canonicalProvider = canonicalizeProviderId(activeSessionConfig.provider);
    const trimmed = nextModel.trim();
    const isSameModel = trimmed.toLowerCase() === activeSessionConfig.model.trim().toLowerCase();
    const nextProfile = getThinkingProfile(canonicalProvider, nextModel);

    const customProv = (llmConfig.custom_providers || []).find((p) => p.id === canonicalProvider);
    const customModelEntry = customProv?.models.find(
      (m) => m.id.toLowerCase() === trimmed.toLowerCase(),
    );

    const explicitCtx =
      typeof option?.context_length === 'number' && option.context_length > 0
        ? Math.round(option.context_length)
        : customModelEntry?.context_window ?? null;
    const explicitOut =
      typeof option?.max_output_tokens === 'number' && option.max_output_tokens > 0
        ? Math.round(option.max_output_tokens)
        : customModelEntry?.max_output_tokens ?? null;

    const resolved = trimmed
      ? resolveModelLimits(canonicalProvider, trimmed, explicitCtx, explicitOut, {
          manualContextWindow: isSameModel ? activeSessionConfig.manual_context_window ?? null : null,
          explicitSource:
            typeof option?.context_length === 'number' && option.context_length > 0
              ? 'provider_catalog'
              : customModelEntry?.context_window
                ? 'custom_provider_config'
                : null,
        })
      : null;

    let nextMaxTokens = activeSessionConfig.max_tokens;
    if (resolved && typeof resolved.maxOutputTokens === 'number') {
      if (!isSameModel || nextMaxTokens <= 0) {
        nextMaxTokens = resolved.maxOutputTokens;
      } else if (nextMaxTokens > resolved.maxOutputTokens) {
        nextMaxTokens = resolved.maxOutputTokens;
      }
    }

    const patch: Partial<ThreadSessionConfig> = {
      provider: canonicalProvider,
      model: nextModel,
      context_window: resolved ? resolved.contextWindow : null,
      context_window_source: resolved ? resolved.contextWindowSource : 'unknown',
      manual_context_window: isSameModel ? activeSessionConfig.manual_context_window ?? null : null,
      max_output_tokens: resolved ? resolved.maxOutputTokens : null,
      max_tokens: nextMaxTokens,
      reasoning_effort: normalizeThinkingLevel(nextProfile, activeSessionConfig.reasoning_effort),
    };

    await patchActiveSession(patch, { persist: shouldPersist && trimmed.length > 0 });
  }

  async function handlePrimaryThinkingChange(nextEffort: string) {
    if (activeThreadId) {
      await patchActiveSession({ reasoning_effort: nextEffort }, { persist: true });
      return;
    }
    setDraftSessionConfig((prev) => ({ ...(prev || {}), reasoning_effort: nextEffort }));
    if (!llmConfig) return;
    const updated: PersistedLLMConfig = {
      ...llmConfig,
      primary: {
        ...llmConfig.primary,
        reasoning_effort: nextEffort,
      },
      access_mode: accessMode,
    };
    setLlmConfig(updated);
    try {
      const saved = await saveLLMSettings(updated);
      setLlmConfig(saved.config);
      setProviders(saved.providers || {});
    } catch {
      // Keep local optimistic selection
    }
  }

  async function handleChangeAccessMode(nextMode: ExecutionAccessMode) {
    const normalized = normalizeAccessMode(nextMode);
    if (activeThreadId) {
      await patchActiveSession({ access_mode: normalized }, { persist: true });
      return;
    }
    setAccessMode(normalized);
    setDraftSessionConfig((prev) => ({ ...(prev || {}), access_mode: normalized }));
    setLlmConfig((prev) => (prev ? { ...prev, access_mode: normalized } : prev));
    try {
      const res = await updateAccessMode(normalized);
      const confirmed = normalizeAccessMode(res.access_mode);
      setAccessMode(confirmed);
      setLlmConfig((prev) => (prev ? { ...prev, access_mode: confirmed } : prev));
    } catch {
      // Keep optimistic selection
    }
  }

  async function handleSetSessionManualContextWindow(manualWindow: number | null) {
    if (!activeSessionConfig) return;
    if (!activeThreadId) {
      setDraftSessionConfig((prev) => ({
        ...(prev || {}),
        manual_context_window: manualWindow,
        context_window: manualWindow ?? activeSessionConfig.context_window ?? null,
        context_window_source: manualWindow ? 'manual' : null,
      }));
      return;
    }
    await patchActiveSession(
      { manual_context_window: manualWindow },
      {
        persist: true,
        optimistic: {
          manual_context_window: manualWindow,
          ...(manualWindow ? { context_window: manualWindow, context_window_source: 'manual' } : {}),
        },
      },
    );
  }

  async function handleRefreshSessionContextMetadata() {
    if (!activeThreadId) return;
    const seq = nextSessionSeq(activeThreadId);
    try {
      const updated = await updateThreadSessionConfig(activeThreadId, { force_refresh: true });
      if (sessionSwitchSeqRef.current.get(activeThreadId) === seq) {
        upsertThread(updated, false);
      }
    } catch {
      // Non-blocking
    }
  }

  if (!authChecked) {
    return (
      <div className="h-screen flex items-center justify-center bg-[var(--paper)] text-[var(--ink-soft)]">
        <div className="flex items-center gap-3">
          <NemofyLogo size={26} />
          <span className="font-display text-sm font-medium">Loading Nemofy Agent...</span>
        </div>
      </div>
    );
  }

  if (!authenticated) {
    return (
      <LoginGate
        theme={theme}
        needsSetup={needsSetup}
        onToggleTheme={toggleTheme}
        onAuthenticated={(user) => {
          setUsername(user);
          setNeedsSetup(false);
          setAuthenticated(true);
        }}
      />
    );
  }

  const activeThreadRun = getThreadRun(activeThreadId);
  const activeThreadDraft = getThreadDraft(activeThreadId);

  return (
    <div className="h-screen flex bg-[var(--paper)] text-[var(--ink)] overflow-hidden">
      {/* Unified Single Collapsible Left Sidebar */}
      <Sidebar
        threads={threads}
        activeThreadId={activeThreadId}
        runningThreadIds={runningThreadIds}
        loadingThreads={loadingThreads}
        onNewThread={() => void createNewThread()}
        onSelectThread={(id) => setActiveThreadId(id)}
        onDeleteThread={(id) => void handleDeleteThread(id)}
        collapsed={sidebarCollapsed}
        onCollapse={() => setSidebarCollapsed(true)}
        onExpand={() => setSidebarCollapsed(false)}
        mobileOpen={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
        theme={theme}
        onToggleTheme={toggleTheme}
        username={username}
        onOpenSettings={() => openSettings('llm')}
        onLogout={() => void handleLogout()}
      />

      {/* Main Chat Canvas */}
      <main className="flex-1 flex min-w-0 min-h-0 overflow-hidden">
        <ChatView
          activeThread={activeThread}
          activeThreadId={activeThreadId}
          threadsError={threadsError}
          threadRun={activeThreadRun}
          threadDraft={activeThreadDraft}
          onPromptChange={(value) => setThreadPrompt(activeThreadId, value)}
          onAttachFiles={(files) => void attachFilesToThread(activeThreadId, files)}
          onRemoveAttachment={(id) => removeAttachmentFromThread(activeThreadId, id)}
          onSendMessage={({ prompt, reasoningEffort, hasPrimaryModel }) =>
            void sendMessage({
              threadId: activeThreadId,
              prompt,
              reasoningEffort,
              accessMode: effectiveAccessMode,
              hasPrimaryModel,
            })
          }
          onStopGeneration={() => {
            if (activeThreadId) {
              void stopThread(activeThreadId);
            }
          }}
          onApprovalDecision={({ decision, toolName, editedArgs }) => {
            if (!activeThreadId) return;
            void submitApprovalDecision({
              threadId: activeThreadId,
              decision,
              toolName,
              editedArgs,
              accessMode: effectiveAccessMode,
            });
          }}
          onCreateThread={() => createNewThread()}
          llmConfig={llmConfig}
          sessionConfig={activeSessionConfig}
          providers={providers}
          onQuickSwitchSessionProvider={(prov) => void handleQuickSwitchSessionProvider(prov)}
          onQuickSwitchPrimaryModel={(model, opt, opts) =>
            void handleQuickSwitchPrimaryModel(model, opt, opts)
          }
          onSetManualContextWindow={handleSetSessionManualContextWindow}
          onRefreshContextMetadata={handleRefreshSessionContextMetadata}
          onPrimaryThinkingChange={(lvl) => void handlePrimaryThinkingChange(lvl)}
          accessMode={effectiveAccessMode}
          onChangeAccessMode={(mode) => void handleChangeAccessMode(mode)}
          onOpenLLMSettings={() => openSettings('llm')}
          onOpenAccessoriesSettings={() => openSettings('accessories')}
          sidebarCollapsed={sidebarCollapsed}
          onExpandSidebar={() => setSidebarCollapsed(false)}
          onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
          workspaceOpen={workspaceDrawerOpen}
          onToggleWorkspace={() => {
            setWorkspaceDrawerOpen((v) => !v);
            setWorkspaceRefreshKey((k) => k + 1);
          }}
          onOpenWorkspaceFile={handleOpenWorkspaceFile}
        />

        {/* Right-Hand Split Workspace Drawer */}
        <WorkspaceDrawer
          open={workspaceDrawerOpen}
          onClose={() => setWorkspaceDrawerOpen(false)}
          targetPath={workspaceTargetPath}
          refreshKey={workspaceRefreshKey}
        />
      </main>

      {/* Centered Settings Modal Dialog */}
      <SettingsModal
        open={settingsModalOpen}
        onClose={() => {
          setSettingsModalOpen(false);
          void reloadLLMSettings();
        }}
        theme={theme}
        onSetTheme={setTheme}
        initialTab={settingsTab}
        activePrimaryReasoningEffort={activeSessionConfig?.reasoning_effort ?? llmConfig?.primary.reasoning_effort}
        activeAccessMode={effectiveAccessMode}
        onLLMConfigSaved={(next, nextProviders) => {
          setLlmConfig(next);
          if (nextProviders) {
            setProviders(nextProviders);
          }
          if (next.access_mode) {
            setAccessMode(normalizeAccessMode(next.access_mode));
          }
        }}
      />
    </div>
  );
}

export default App;
