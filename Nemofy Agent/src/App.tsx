import { useCallback, useEffect, useRef, useState } from 'react';
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
} from './lib/apiClient.ts';
import { resolveModelLimits } from './lib/contextUtils.ts';
import { canonicalizeProviderId } from './lib/llmPresets.ts';
import { getThinkingProfile, normalizeThinkingLevel } from './lib/thinkingModes.ts';
import type {
  ExecutionAccessMode,
  PersistedLLMConfig,
  ProviderModelOption,
  SettingsTab,
  ThemeMode,
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
  const [accessMode, setAccessMode] = useState<ExecutionAccessMode>('default');

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

  const handleWorkspaceFilesChanged = useCallback(() => {
    setWorkspaceRefreshKey((k) => k + 1);
  }, []);

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
    onCreateThread: createNewThread,
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

  useEffect(() => {
    if (!authenticated) return;
    let mounted = true;
    (async () => {
      try {
        const res = await fetchLLMSettings();
        if (mounted) {
          setLlmConfig(res.config);
          setAccessMode(normalizeAccessMode(res.config.access_mode));
        }
      } catch {
        // non-blocking
      }
    })();
    return () => {
      mounted = false;
    };
  }, [authenticated]);

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
    setAccessMode('default');
    setSettingsModalOpen(false);
    setWorkspaceDrawerOpen(false);
    setMobileSidebarOpen(false);
  }

  async function handleDeleteThread(id: string) {
    await cancelAndClearThread(id);
    await removeThread(id);
  }

  async function handlePrimaryThinkingChange(nextEffort: string) {
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
    } catch {
      // Keep local optimistic selection; sendChatMessage still sends explicit reasoningEffort
    }
  }

  async function handleChangeAccessMode(nextMode: ExecutionAccessMode) {
    const normalized = normalizeAccessMode(nextMode);
    setAccessMode(normalized);
    setLlmConfig((prev) => (prev ? { ...prev, access_mode: normalized } : prev));
    try {
      const res = await updateAccessMode(normalized);
      const confirmed = normalizeAccessMode(res.access_mode);
      setAccessMode(confirmed);
      setLlmConfig((prev) => (prev ? { ...prev, access_mode: confirmed } : prev));
    } catch {
      // Keep optimistic selection; also passed explicitly on every run/resume request
    }
  }

  const quickSwitchSeqRef = useRef(0);

  async function handleQuickSwitchPrimaryModel(
    nextModel: string,
    option?: ProviderModelOption,
    options?: { persist?: boolean },
  ) {
    if (!llmConfig) return;
    const shouldPersist = options?.persist !== false;
    const seq = ++quickSwitchSeqRef.current;
    const canonicalProvider = canonicalizeProviderId(llmConfig.primary.provider);
    const trimmed = nextModel.trim();
    const isSameModel = trimmed.toLowerCase() === llmConfig.primary.model.trim().toLowerCase();
    const nextProfile = getThinkingProfile(canonicalProvider, nextModel);

    const explicitCtx =
      typeof option?.context_length === 'number' && option.context_length > 0
        ? Math.round(option.context_length)
        : null;
    const explicitOut =
      typeof option?.max_output_tokens === 'number' && option.max_output_tokens > 0
        ? Math.round(option.max_output_tokens)
        : null;

    const resolved = trimmed
      ? resolveModelLimits(canonicalProvider, trimmed, explicitCtx, explicitOut)
      : null;

    let nextMaxTokens = llmConfig.primary.max_tokens;
    if (resolved && typeof resolved.maxOutputTokens === 'number') {
      if (!isSameModel || nextMaxTokens <= 0) {
        nextMaxTokens = resolved.maxOutputTokens;
      } else if (nextMaxTokens > resolved.maxOutputTokens) {
        nextMaxTokens = resolved.maxOutputTokens;
      }
    }

    const updated: PersistedLLMConfig = {
      ...llmConfig,
      access_mode: accessMode,
      primary: {
        ...llmConfig.primary,
        provider: canonicalProvider,
        model: nextModel,
        context_window: resolved ? resolved.contextWindow : null,
        max_output_tokens: resolved ? resolved.maxOutputTokens : null,
        max_tokens: nextMaxTokens,
        reasoning_effort: normalizeThinkingLevel(nextProfile, llmConfig.primary.reasoning_effort),
      },
    };

    setLlmConfig(updated);

    // Only persist to server when explicitly committing a non-empty model ID (not mid-keystroke)
    if (!shouldPersist || !trimmed) return;
    try {
      const saved = await saveLLMSettings(updated);
      if (quickSwitchSeqRef.current === seq) {
        setLlmConfig(saved.config);
      }
    } catch {
      // Non-blocking; user can still save or inspect errors in Settings modal
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
              accessMode,
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
              accessMode,
            });
          }}
          onCreateThread={createNewThread}
          llmConfig={llmConfig}
          onQuickSwitchPrimaryModel={(model, opt) => void handleQuickSwitchPrimaryModel(model, opt)}
          onPrimaryThinkingChange={(lvl) => void handlePrimaryThinkingChange(lvl)}
          accessMode={accessMode}
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
        onClose={() => setSettingsModalOpen(false)}
        theme={theme}
        onSetTheme={setTheme}
        initialTab={settingsTab}
        activePrimaryReasoningEffort={llmConfig?.primary.reasoning_effort}
        activeAccessMode={accessMode}
        onLLMConfigSaved={(next) => {
          setLlmConfig(next);
          if (next.access_mode) {
            setAccessMode(normalizeAccessMode(next.access_mode));
          }
        }}
      />
    </div>
  );
}

export default App;
