import {
  Check,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Layers,
  MessageSquarePlus,
  Send,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  X,
  Brain,
} from 'lucide-react';
import {
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  type ActivityUserPrefs,
  readActivityPrefs,
  writeActivityPrefs,
} from '../lib/activityUtils.ts';
import {
  cancelChatJob,
  createThread,
  deleteThread,
  listThreads,
  pollChatJob,
  resumeApproval,
  startChatJob,
} from '../lib/apiClient.ts';
import { computeAutoGrowHeight, resolveScrollPinState } from '../lib/scrollUtils.ts';
import {
  type ThinkingLevel,
  formatThinkingBadge,
  getThinkingProfile,
  normalizeThinkingLevel,
} from '../lib/thinkingModes.ts';
import type {
  ChatMessage,
  ChatThread,
  LiveAgentActivity,
  PersistedLLMConfig,
} from '../types/agent.ts';
import { AgentActivityPanel } from './AgentActivityPanel.tsx';
import { MarkdownMessage } from './MarkdownMessage.tsx';

interface ChatViewProps {
  llmConfig: PersistedLLMConfig | null;
  onPrimaryThinkingChange?: (level: ThinkingLevel) => void;
  onOpenLLMSettings: () => void;
}

function cleanThreadTitle(raw: string): string {
  return raw
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[*_~`#>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 56);
}

const STARTER_PROMPTS = [
  'Compare Python, TypeScript, and Rust in a Markdown data table and include an ASCII bar chart of developer adoption.',
  'Inspect /workspace files and summarize what is currently in my project.',
  'Draft a clear, actionable 5-step launch checklist for a new SaaS product.',
];

function MessageMetadata({ msg }: { msg: ChatMessage }) {
  const [planOpen, setPlanOpen] = useState(false);
  const [reportsOpen, setReportsOpen] = useState(false);
  const hasPlan = Boolean(msg.plan && msg.plan.length > 0);
  const hasReports = Boolean(msg.subagentReports && msg.subagentReports.length > 0);
  const hasSkills = Boolean(msg.activeSkills && msg.activeSkills.length > 0);

  if (!hasPlan && !hasReports && !hasSkills) return null;

  return (
    <div className="mt-3 pt-3 border-t border-[var(--rule)] space-y-2 text-xs">
      <div className="flex flex-wrap items-center gap-3 text-[var(--ink-soft)]">
        {hasPlan && (
          <button
            type="button"
            onClick={() => setPlanOpen((v) => !v)}
            className="inline-flex items-center gap-1 font-medium hover:text-[var(--ink)] cursor-pointer"
          >
            {planOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            <span>
              Execution plan ({msg.plan!.length} {msg.plan!.length === 1 ? 'step' : 'steps'})
            </span>
          </button>
        )}
        {hasReports && (
          <button
            type="button"
            onClick={() => setReportsOpen((v) => !v)}
            className="inline-flex items-center gap-1 font-medium hover:text-[var(--ink)] cursor-pointer"
          >
            {reportsOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            <Layers className="w-3.5 h-3.5" />
            <span>
              Sub-agent {msg.subagentReports!.length === 1 ? 'report' : 'reports'} (
              {msg.subagentReports!.length})
            </span>
          </button>
        )}
        {hasSkills && (
          <span className="font-mono text-[11px] text-[var(--ink-faint)]">
            Skills: {msg.activeSkills!.join(', ')}
          </span>
        )}
      </div>

      {planOpen && hasPlan && (
        <ul className="space-y-1.5 pl-1">
          {msg.plan!.map((step) => (
            <li key={step.step_id} className="flex items-start gap-2 text-[var(--ink-soft)]">
              <span
                className={`mt-0.5 font-mono text-[10px] uppercase shrink-0 ${
                  step.status === 'completed'
                    ? 'text-[var(--jade)]'
                    : step.status === 'failed'
                      ? 'text-[var(--flare)]'
                      : 'text-[var(--honey)]'
                }`}
              >
                [{step.status}]
              </span>
              <span className="text-[var(--ink)]">{step.description}</span>
            </li>
          ))}
        </ul>
      )}

      {reportsOpen && hasReports && (
        <div className="space-y-2">
          {msg.subagentReports!.map((rep, idx) => (
            <pre
              key={idx}
              className="p-2.5 rounded-md bg-[var(--paper-sunk)] border border-[var(--rule)] font-mono text-[11px] text-[var(--ink-soft)] overflow-x-auto"
            >
              {JSON.stringify(rep, null, 2)}
            </pre>
          ))}
        </div>
      )}
    </div>
  );
}

export function ChatView({
  llmConfig,
  onPrimaryThinkingChange,
  onOpenLLMSettings,
}: ChatViewProps) {
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [loadingThreads, setLoadingThreads] = useState(true);
  const [prompt, setPrompt] = useState('');
  const [sending, setSending] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [liveActivity, setLiveActivity] = useState<LiveAgentActivity | null>(null);
  const [activityPrefs, setActivityPrefs] = useState<ActivityUserPrefs>(() => readActivityPrefs());
  const [error, setError] = useState<string | null>(null);

  const pollCancelledRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const activeJobIdRef = useRef<string | null>(null);
  const activeAbortRef = useRef<AbortController | null>(null);

  // Adaptive Thinking Mode state (auto-adapts to active primary model capabilities)
  const primaryProvider = llmConfig?.primary.provider || 'river_ai';
  const primaryModel = llmConfig?.primary.model || 'deepseek-ai/DeepSeek-V4.1-Flash';
  const thinkingProfile = useMemo(
    () => getThinkingProfile(primaryProvider, primaryModel),
    [primaryProvider, primaryModel],
  );
  const [selectedThinking, setSelectedThinking] = useState<ThinkingLevel>(() =>
    normalizeThinkingLevel(thinkingProfile, llmConfig?.primary.reasoning_effort),
  );
  const [thinkingMenuOpen, setThinkingMenuOpen] = useState(false);
  const thinkingMenuRef = useRef<HTMLDivElement>(null);

  // HITL approval state
  const [editingApproval, setEditingApproval] = useState(false);
  const [editedArgsJson, setEditedArgsJson] = useState('');
  const [resuming, setResuming] = useState(false);

  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const isPinnedToBottomRef = useRef(true);
  const lastScrollTopRef = useRef(0);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  const activeThread = threads.find((t) => t.id === activeThreadId) || null;

  function handleUpdateActivityPrefs(patch: Partial<ActivityUserPrefs>) {
    setActivityPrefs((prev) => {
      const next = { ...prev, ...patch };
      writeActivityPrefs(next);
      return next;
    });
  }

  useEffect(() => {
    pollCancelledRef.current = false;
    return () => {
      pollCancelledRef.current = true;
      activeAbortRef.current?.abort();
    };
  }, []);

  // Smooth elapsed timer ticker while Nemo is working
  useEffect(() => {
    if (!sending) return;
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setLiveActivity((prev) => {
        if (!prev) return prev;
        const localElapsed = (Date.now() - startedAt) / 1000;
        return localElapsed > prev.elapsedSeconds
          ? { ...prev, elapsedSeconds: localElapsed }
          : prev;
      });
    }, 300);
    return () => window.clearInterval(timer);
  }, [sending]);

  // Keep selected thinking mode synchronized & valid whenever active model or llmConfig changes
  useEffect(() => {
    const nextLevel = normalizeThinkingLevel(
      thinkingProfile,
      llmConfig?.primary.reasoning_effort ?? selectedThinking,
    );
    setSelectedThinking(nextLevel);
  }, [thinkingProfile, llmConfig?.primary.reasoning_effort]);

  // Close thinking mode popover on outside click or Escape
  useEffect(() => {
    if (!thinkingMenuOpen) return;
    function handleOutside(e: MouseEvent) {
      if (thinkingMenuRef.current && !thinkingMenuRef.current.contains(e.target as Node)) {
        setThinkingMenuOpen(false);
      }
    }
    function handleEscape(e: globalThis.KeyboardEvent) {
      if (e.key === 'Escape') setThinkingMenuOpen(false);
    }
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [thinkingMenuOpen]);

  function handleSelectThinking(level: ThinkingLevel) {
    setSelectedThinking(level);
    setThinkingMenuOpen(false);
    onPrimaryThinkingChange?.(level);
  }

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const loaded = await listThreads();
        if (!mounted) return;
        setThreads(loaded);
        if (loaded.length > 0) {
          setActiveThreadId(loaded[0].id);
        }
      } catch (err) {
        if (mounted) setError(err instanceof Error ? err.message : 'Unable to load sessions.');
      } finally {
        if (mounted) setLoadingThreads(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // Auto-expand chat composer vertically as the user types multi-sentence prompts
  useLayoutEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const { heightPx, isCapped } = computeAutoGrowHeight(el.scrollHeight, 36, 220);
    el.style.height = `${heightPx}px`;
    el.style.overflowY = isCapped ? 'auto' : 'hidden';
  }, [prompt]);

  function scrollMessagesToBottom() {
    const el = messagesScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    lastScrollTopRef.current = el.scrollTop;
  }

  function handleMessagesScroll() {
    const el = messagesScrollRef.current;
    if (!el) return;
    isPinnedToBottomRef.current = resolveScrollPinState(
      {
        scrollTop: el.scrollTop,
        scrollHeight: el.scrollHeight,
        clientHeight: el.clientHeight,
      },
      lastScrollTopRef.current,
      isPinnedToBottomRef.current,
      48,
    );
    lastScrollTopRef.current = el.scrollTop;
  }

  // Re-pin cleanly when switching threads
  useEffect(() => {
    isPinnedToBottomRef.current = true;
    requestAnimationFrame(() => scrollMessagesToBottom());
  }, [activeThreadId]);

  // Follow new messages or activity ONLY when the user has not scrolled up
  useEffect(() => {
    if (!isPinnedToBottomRef.current) return;
    scrollMessagesToBottom();
  }, [
    activeThread?.messages.length,
    sending,
    liveActivity?.toolEvents.length,
    liveActivity?.fileChanges.length,
  ]);

  async function handleNewThread() {
    setError(null);
    try {
      const created = await createThread('New Agent Session');
      setThreads((prev) => [created, ...prev]);
      setActiveThreadId(created.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to create session.');
    }
  }

  async function handleDeleteThread(id: string) {
    try {
      await deleteThread(id);
      setThreads((prev) => {
        const remaining = prev.filter((t) => t.id !== id);
        if (activeThreadId === id) {
          setActiveThreadId(remaining[0]?.id || null);
        }
        return remaining;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to delete session.');
    }
  }

  async function handleStopGeneration() {
    if (!sending || stopping) return;
    stopRequestedRef.current = true;
    setStopping(true);
    activeAbortRef.current?.abort();

    const jobId = activeJobIdRef.current;
    if (jobId) {
      try {
        const res = await cancelChatJob(jobId);
        if (res.thread) {
          const stoppedThread = res.thread;
          setThreads((prev) => {
            const exists = prev.some((t) => t.id === stoppedThread.id);
            if (exists) {
              return prev.map((t) => (t.id === stoppedThread.id ? stoppedThread : t));
            }
            return [stoppedThread, ...prev];
          });
          setActiveThreadId(stoppedThread.id);
        }
      } catch {
        // Ignore network errors while cancelling; UI still stops cleanly
      }
    }
  }

  async function submitPrompt(textToSend: string) {
    const clean = textToSend.trim();
    if (!clean || sending) return;
    setError(null);
    setSending(true);
    setStopping(false);
    setThinkingMenuOpen(false);
    setPrompt('');

    stopRequestedRef.current = false;
    activeJobIdRef.current = null;
    const controller = new AbortController();
    activeAbortRef.current = controller;

    // User explicitly sent a new message: scroll once to show it
    isPinnedToBottomRef.current = true;

    const optimisticId = `temp-${Date.now()}`;
    let targetId = activeThreadId;
    let committedOnServer = false;

    try {
      const derivedTitle = cleanThreadTitle(clean) || 'New Agent Session';
      if (!targetId) {
        const created = await createThread(derivedTitle);
        setThreads((prev) => [created, ...prev]);
        setActiveThreadId(created.id);
        targetId = created.id;
      }

      // Optimistic user message display
      setThreads((prev) =>
        prev.map((t) =>
          t.id === targetId
            ? {
                ...t,
                title: t.messages.length === 0 ? derivedTitle : t.title,
                messages: [
                  ...t.messages,
                  {
                    id: optimisticId,
                    role: 'user',
                    content: clean,
                    timestamp: new Date().toISOString(),
                  },
                ],
              }
            : t,
        ),
      );

      const effectiveThinking = thinkingProfile.supported ? selectedThinking : 'off';
      setLiveActivity({
        jobId: 'starting',
        status: 'running',
        currentPhase: 'thinking',
        lastEvent: 'Reviewing your request and choosing the best approach',
        elapsedSeconds: 0,
        turnCount: 1,
        thinking: null,
        activityFeed: [
          {
            id: 'init-1',
            phase: 'thinking',
            text: 'Reviewing your request and choosing the best approach',
          },
        ],
        toolEvents: [],
        fileChanges: [],
      });

      const started = await startChatJob(targetId, clean, effectiveThinking, controller.signal);
      committedOnServer = true;

      // Commit the persisted user message from server right away
      setThreads((prev) => {
        const exists = prev.some((t) => t.id === started.thread.id);
        if (exists) {
          return prev.map((t) => (t.id === started.thread.id ? started.thread : t));
        }
        return [started.thread, ...prev];
      });
      setActiveThreadId(started.thread.id);

      if (!started.jobId) {
        // Standalone mode or immediate error already persisted on started.thread
        setLiveActivity(null);
        return;
      }

      activeJobIdRef.current = started.jobId;

      if (stopRequestedRef.current) {
        const stopped = await cancelChatJob(started.jobId);
        if (stopped.thread) {
          const stoppedThread = stopped.thread;
          setThreads((prev) => prev.map((t) => (t.id === stoppedThread.id ? stoppedThread : t)));
        }
        return;
      }

      if (started.activity) {
        setLiveActivity(started.activity);
      }

      // Live poll loop until terminal state (fast 300ms long-poll for live thinking deltas)
      while (!pollCancelledRef.current && !stopRequestedRef.current) {
        const polled = await pollChatJob(started.jobId, 300, controller.signal);
        if (pollCancelledRef.current || stopRequestedRef.current) break;

        if (polled.activity) {
          setLiveActivity(polled.activity);
        }

        if (polled.done) {
          if (polled.thread) {
            const finalThread = polled.thread;
            setThreads((prev) => {
              const exists = prev.some((t) => t.id === finalThread.id);
              if (exists) {
                return prev.map((t) => (t.id === finalThread.id ? finalThread : t));
              }
              return [finalThread, ...prev];
            });
            setActiveThreadId(finalThread.id);
          }
          break;
        }
      }
    } catch (err) {
      const isAbort =
        stopRequestedRef.current ||
        (err instanceof DOMException && err.name === 'AbortError') ||
        (err instanceof Error && err.name === 'AbortError');

      if (isAbort) {
        // User intentionally stopped the run — do not display an error banner
        if (!committedOnServer && targetId) {
          setThreads((prev) =>
            prev.map((t) =>
              t.id === targetId
                ? { ...t, messages: t.messages.filter((m) => m.id !== optimisticId) }
                : t,
            ),
          );
          setPrompt(clean);
        }
        return;
      }

      // Roll back optimistic message and restore draft so the user never loses input
      if (!committedOnServer && targetId) {
        setThreads((prev) =>
          prev.map((t) =>
            t.id === targetId
              ? { ...t, messages: t.messages.filter((m) => m.id !== optimisticId) }
              : t,
          ),
        );
        setPrompt(clean);
      }
      setError(err instanceof Error ? err.message : 'Failed to execute task.');
    } finally {
      activeJobIdRef.current = null;
      activeAbortRef.current = null;
      stopRequestedRef.current = false;
      setSending(false);
      setStopping(false);
      setLiveActivity(null);
    }
  }

  function handleFormSubmit(e: FormEvent) {
    e.preventDefault();
    void submitPrompt(prompt);
  }

  function handleTextareaKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void submitPrompt(prompt);
    }
  }

  async function handleApprovalDecision(decision: 'approve' | 'reject' | 'edit') {
    if (!activeThread || resuming) return;
    setError(null);
    let parsedArgs: Record<string, unknown> | undefined;
    if (decision === 'edit') {
      try {
        parsedArgs = JSON.parse(editedArgsJson) as Record<string, unknown>;
      } catch {
        setError('Edited tool arguments must be valid JSON.');
        return;
      }
    }
    setResuming(true);
    try {
      const res = await resumeApproval(activeThread.id, decision, parsedArgs);
      if (res.thread) {
        setThreads((prev) => prev.map((t) => (t.id === res.thread!.id ? res.thread! : t)));
      }
      setEditingApproval(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to submit approval decision.');
    } finally {
      setResuming(false);
    }
  }

  return (
    <div className="flex h-full min-h-0 bg-[var(--paper)]">
      {/* Left Sessions Rail */}
      <div className="hidden md:flex flex-col w-64 border-r border-[var(--rule)] bg-[var(--paper-raised)] shrink-0">
        <div className="h-16 px-3.5 border-b border-[var(--rule)] flex items-center shrink-0">
          <button
            type="button"
            onClick={handleNewThread}
            className="w-full h-9 inline-flex items-center justify-center gap-2 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] text-white text-xs font-semibold transition-colors cursor-pointer"
          >
            <MessageSquarePlus className="w-3.5 h-3.5" />
            <span>New chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-none p-2 space-y-1">
          {loadingThreads ? (
            <div className="p-4 text-xs text-[var(--ink-faint)]">Loading sessions...</div>
          ) : threads.length === 0 ? (
            <div className="p-4 text-xs text-[var(--ink-faint)] leading-relaxed">
              No conversations yet. Start a new chat with Nemo.
            </div>
          ) : (
            threads.map((t) => {
              const selected = t.id === activeThreadId;
              return (
                <div
                  key={t.id}
                  className={`group flex items-center justify-between gap-1 rounded-lg px-2.5 py-2 text-xs transition-colors ${
                    selected
                      ? 'bg-[var(--blush)] text-[var(--signal)] font-medium'
                      : 'text-[var(--ink-soft)] hover:bg-[var(--paper-sunk)] hover:text-[var(--ink)]'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => setActiveThreadId(t.id)}
                    className="flex-1 text-left truncate cursor-pointer"
                  >
                    {t.title || 'Untitled session'}
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleDeleteThread(t.id)}
                    aria-label={`Delete ${t.title}`}
                    className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 rounded hover:text-[var(--flare)] transition-opacity cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Main Conversation Area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        {/* Conversation Header */}
        <div className="h-16 px-5 sm:px-6 border-b border-[var(--rule)] flex items-center justify-between gap-4 shrink-0 bg-[var(--paper)]">
          <div className="min-w-0">
            <h1 className="font-display text-base font-semibold text-[var(--ink)] truncate">
              {activeThread?.title || 'Chat with Nemo'}
            </h1>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {llmConfig && (
              <button
                type="button"
                onClick={onOpenLLMSettings}
                className="hidden sm:inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-[var(--rule)] bg-[var(--paper-sunk)] hover:border-[var(--rule-strong)] text-xs text-[var(--ink-soft)] hover:text-[var(--ink)] transition-colors cursor-pointer"
                title="Configure Nemo and Sub-Agent LLM settings"
              >
                <SlidersHorizontal className="w-3.5 h-3.5 text-[var(--signal)]" />
                <span className="font-mono truncate max-w-[200px]">
                  {llmConfig.primary.provider}: {llmConfig.primary.model}
                </span>
              </button>
            )}
            <button
              type="button"
              onClick={handleNewThread}
              className="md:hidden inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[var(--signal)] text-white text-xs font-semibold cursor-pointer"
            >
              <MessageSquarePlus className="w-3.5 h-3.5" />
              <span>New</span>
            </button>
          </div>
        </div>

        {/* Messages Scroll Pane */}
        <div
          ref={messagesScrollRef}
          onScroll={handleMessagesScroll}
          onWheel={(e) => {
            if (e.deltaY < 0) {
              isPinnedToBottomRef.current = false;
            }
          }}
          className="flex-1 overflow-y-auto scrollbar-none px-5 sm:px-8 py-6"
        >
          {!activeThread || activeThread.messages.length === 0 ? (
            <div className="max-w-2xl mx-auto mt-8 sm:mt-16">
              <div className="inline-flex items-center gap-2 text-xs font-medium text-[var(--signal)] mb-3">
                <Sparkles className="w-4 h-4" />
                <span>Nemo · Nemofy Agent</span>
              </div>
              <h2 className="font-display text-2xl sm:text-3xl font-semibold tracking-tight text-[var(--ink)]">
                Hi, I&apos;m Nemo. How can I help today?
              </h2>
              <p className="mt-2.5 text-sm text-[var(--ink-soft)] leading-relaxed">
                Your diligent, honest, and friendly AI inside Nemofy Agent — ready for quick questions, data tables, ASCII charts, research, coding, and tasks in <code className="font-mono text-xs">/workspace</code>.
              </p>

              <div className="mt-7 grid gap-2.5">
                {STARTER_PROMPTS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => void submitPrompt(suggestion)}
                    className="text-left p-3.5 rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] hover:border-[var(--signal)] hover:bg-[var(--blush)]/40 text-sm text-[var(--ink)] transition-colors cursor-pointer"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-5">
              {activeThread.messages.map((msg) => {
                const isUser = msg.role === 'user';
                return (
                  <div
                    key={msg.id}
                    className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[92%] sm:max-w-[88%] rounded-xl px-4 py-3.5 text-sm ${
                        isUser
                          ? 'bg-[var(--signal)] text-white'
                          : 'bg-[var(--paper-raised)] border border-[var(--rule)] text-[var(--ink)]'
                      }`}
                    >
                      <div
                        className={`text-[11px] font-medium mb-1.5 ${
                          isUser ? 'text-white/75' : 'text-[var(--ink-faint)]'
                        }`}
                      >
                        {isUser ? 'You' : 'Nemo'}
                        {!isUser && msg.activeModels?.primary?.model && (
                          <span className="ml-2 font-mono text-[10px]">
                            ({msg.activeModels.primary.model}
                            {formatThinkingBadge(msg.activeModels.primary.reasoning_effort)
                              ? ` · ${formatThinkingBadge(msg.activeModels.primary.reasoning_effort)}`
                              : ''}
                            )
                          </span>
                        )}
                      </div>
                      {!isUser && (
                        <AgentActivityPanel
                          mode="completed"
                          thinking={msg.thinking}
                          toolEvents={msg.toolEvents}
                          fileChanges={msg.fileChanges}
                          prefs={activityPrefs}
                          onUpdatePrefs={handleUpdateActivityPrefs}
                        />
                      )}
                      <MarkdownMessage content={msg.content} isUser={isUser} />
                      {!isUser && <MessageMetadata msg={msg} />}
                    </div>
                  </div>
                );
              })}

              {/* Pending HITL Approval Banner */}
              {activeThread.pendingApproval && (
                <div className="rounded-xl border border-[var(--honey)]/50 bg-[var(--paper-raised)] p-4 space-y-3">
                  <div className="flex items-start gap-2.5">
                    <ShieldAlert className="w-5 h-5 text-[var(--honey)] shrink-0 mt-0.5" />
                    <div className="min-w-0 flex-1">
                      <h3 className="font-display text-sm font-semibold text-[var(--ink)]">
                        Approval required before executing tool:{' '}
                        <span className="font-mono">
                          {String(activeThread.pendingApproval.tool_name || 'sensitive_action')}
                        </span>
                      </h3>
                      {activeThread.pendingApproval.risk_reason && (
                        <p className="mt-1 text-xs text-[var(--ink-soft)]">
                          {String(activeThread.pendingApproval.risk_reason)}
                        </p>
                      )}
                    </div>
                  </div>

                  {editingApproval ? (
                    <textarea
                      value={editedArgsJson}
                      onChange={(e) => setEditedArgsJson(e.target.value)}
                      rows={5}
                      className="w-full p-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                    />
                  ) : (
                    <pre className="p-3 rounded-lg bg-[var(--paper-sunk)] border border-[var(--rule)] font-mono text-xs text-[var(--ink-soft)] overflow-x-auto">
                      {JSON.stringify(activeThread.pendingApproval.arguments || {}, null, 2)}
                    </pre>
                  )}

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      disabled={resuming}
                      onClick={() => void handleApprovalDecision(editingApproval ? 'edit' : 'approve')}
                      className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-[var(--jade)] hover:opacity-90 disabled:opacity-50 text-white text-xs font-semibold cursor-pointer"
                    >
                      <Check className="w-3.5 h-3.5" />
                      <span>{editingApproval ? 'Save args & continue' : 'Approve & run'}</span>
                    </button>
                    {!editingApproval ? (
                      <button
                        type="button"
                        disabled={resuming}
                        onClick={() => {
                          setEditedArgsJson(
                            JSON.stringify(activeThread.pendingApproval?.arguments || {}, null, 2),
                          );
                          setEditingApproval(true);
                        }}
                        className="px-3.5 py-2 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-xs font-medium text-[var(--ink)] hover:bg-[var(--paper-sunk)] cursor-pointer"
                      >
                        Edit arguments
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setEditingApproval(false)}
                        className="px-3.5 py-2 rounded-lg border border-[var(--rule)] text-xs text-[var(--ink-soft)] cursor-pointer"
                      >
                        Cancel edit
                      </button>
                    )}
                    <button
                      type="button"
                      disabled={resuming}
                      onClick={() => void handleApprovalDecision('reject')}
                      className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-[var(--flare)]/40 text-[var(--flare)] hover:bg-[var(--flare)]/10 text-xs font-medium cursor-pointer"
                    >
                      <X className="w-3.5 h-3.5" />
                      <span>Reject</span>
                    </button>
                  </div>
                </div>
              )}

              {sending && liveActivity && (
                <div className="flex justify-start">
                  <AgentActivityPanel
                    mode="live"
                    activity={liveActivity}
                    prefs={activityPrefs}
                    onUpdatePrefs={handleUpdateActivityPrefs}
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Prompt Composer */}
        <div className="border-t border-[var(--rule)] bg-[var(--paper)] px-5 sm:px-8 py-4 shrink-0">
          <div className="max-w-3xl mx-auto">
            {error && (
              <div
                role="alert"
                className="mb-3 rounded-lg border border-[var(--flare)]/30 bg-[var(--flare)]/10 px-3.5 py-2 text-xs text-[var(--flare)]"
              >
                {error}
              </div>
            )}

            <form
              onSubmit={handleFormSubmit}
              className="flex items-end gap-2 rounded-xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] p-2 focus-within:border-[var(--signal)] focus-within:ring-2 focus-within:ring-[var(--signal)]/15 transition-all"
            >
              <label htmlFor="chat-composer" className="sr-only">
                Message Nemo
              </label>
              <textarea
                ref={composerRef}
                id="chat-composer"
                rows={1}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleTextareaKeyDown}
                placeholder="Message Nemo — ask anything, build tables, charts, or code..."
                disabled={sending}
                className="flex-1 min-w-0 resize-none bg-transparent px-2.5 py-2 text-sm leading-5 text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none min-h-[36px] max-h-[220px] overflow-hidden scrollbar-none"
              />

              {/* Adaptive Model Thinking Mode Selector (just left of Send) */}
              <div ref={thinkingMenuRef} className="relative shrink-0">
                <button
                  type="button"
                  disabled={sending}
                  onClick={() => setThinkingMenuOpen((v) => !v)}
                  aria-haspopup="listbox"
                  aria-expanded={thinkingMenuOpen}
                  aria-label={`Thinking mode: ${
                    thinkingProfile.options.find((o) => o.value === selectedThinking)?.label ||
                    'Standard'
                  } (${thinkingProfile.familyLabel})`}
                  title={`Thinking mode for ${primaryModel} (${thinkingProfile.familyLabel})`}
                  className={`h-9 px-2.5 inline-flex items-center gap-1.5 rounded-lg border text-xs font-medium transition-colors cursor-pointer disabled:opacity-50 ${
                    !thinkingProfile.supported
                      ? 'border-[var(--rule)] bg-[var(--paper-sunk)] text-[var(--ink-faint)] hover:text-[var(--ink-soft)]'
                      : selectedThinking !== 'off'
                        ? 'border-[var(--signal)]/45 bg-[var(--blush)] text-[var(--signal)] hover:border-[var(--signal)]'
                        : 'border-[var(--rule)] bg-[var(--paper-sunk)] text-[var(--ink-soft)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]'
                  }`}
                >
                  <Brain className="w-3.5 h-3.5 shrink-0" />
                  <span className="max-w-[76px] truncate">
                    {thinkingProfile.options.find((o) => o.value === selectedThinking)?.shortLabel ||
                      'Standard'}
                  </span>
                  {thinkingMenuOpen ? (
                    <ChevronDown className="w-3 h-3 opacity-70 shrink-0" />
                  ) : (
                    <ChevronUp className="w-3 h-3 opacity-70 shrink-0" />
                  )}
                </button>

                {thinkingMenuOpen && (
                  <div
                    role="listbox"
                    aria-label="Select model thinking mode"
                    className="absolute bottom-full right-0 mb-2 w-64 rounded-xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-lg overflow-hidden z-30"
                  >
                    <div className="px-3 py-2 border-b border-[var(--rule)] bg-[var(--paper-sunk)]">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[11px] font-semibold text-[var(--ink)]">
                          Thinking Mode
                        </span>
                        <span className="text-[10px] font-mono text-[var(--signal)] truncate">
                          {thinkingProfile.familyLabel}
                        </span>
                      </div>
                      <div className="text-[10px] font-mono text-[var(--ink-faint)] truncate mt-0.5">
                        {primaryModel}
                      </div>
                    </div>

                    {thinkingProfile.supported ? (
                      <div className="p-1.5 space-y-0.5 max-h-[420px] overflow-y-auto scrollbar-none">
                        {thinkingProfile.options.map((opt) => {
                          const isSelected = selectedThinking === opt.value;
                          return (
                            <button
                              key={opt.value}
                              type="button"
                              role="option"
                              aria-selected={isSelected}
                              onClick={() => handleSelectThinking(opt.value)}
                              className={`w-full px-2.5 py-1.5 rounded-lg text-left flex items-start justify-between gap-2 transition-colors cursor-pointer ${
                                isSelected
                                  ? 'bg-[var(--blush)] text-[var(--signal)]'
                                  : 'text-[var(--ink)] hover:bg-[var(--paper-sunk)]'
                              }`}
                            >
                              <div className="min-w-0">
                                <div className="text-xs font-semibold leading-tight">
                                  {opt.label}
                                </div>
                                <div
                                  className={`text-[11px] leading-snug mt-0.5 ${
                                    isSelected ? 'text-[var(--signal)]/80' : 'text-[var(--ink-faint)]'
                                  }`}
                                >
                                  {opt.description}
                                </div>
                              </div>
                              {isSelected && <Check className="w-3.5 h-3.5 shrink-0 mt-0.5" />}
                            </button>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="p-3 space-y-2.5 text-xs text-[var(--ink-soft)]">
                        <p className="leading-relaxed">
                          <span className="font-mono text-[var(--ink)]">{primaryModel}</span> is a
                          standard chat model and does not support configurable thinking levels.
                        </p>
                        <button
                          type="button"
                          onClick={() => {
                            setThinkingMenuOpen(false);
                            onOpenLLMSettings();
                          }}
                          className="w-full py-1.5 px-2.5 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] text-[var(--ink)] text-xs font-medium transition-colors cursor-pointer"
                        >
                          Switch model in Settings
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {sending ? (
                <button
                  type="button"
                  onClick={() => void handleStopGeneration()}
                  disabled={stopping}
                  aria-label="Stop generation"
                  title="Stop running agent"
                  className="h-9 px-3.5 inline-flex items-center gap-1.5 rounded-lg bg-[var(--flare)] hover:opacity-90 disabled:opacity-60 text-white text-xs font-semibold transition-opacity cursor-pointer shrink-0"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>{stopping ? 'Stopping...' : 'Stop'}</span>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!prompt.trim()}
                  aria-label="Send message"
                  className="h-9 px-3.5 inline-flex items-center gap-1.5 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-45 text-white text-xs font-semibold transition-colors cursor-pointer shrink-0"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>Send</span>
                </button>
              )}
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
