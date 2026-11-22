import {
  AudioLines,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Image as ImageIcon,
  Layers,
  Loader2,
  MessageSquarePlus,
  Paperclip,
  Send,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Video as VideoIcon,
  X,
} from 'lucide-react';
import {
  type ClipboardEvent,
  type DragEvent,
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
  MAX_CONSECUTIVE_POLL_ERRORS,
  computePollRetryDelayMs,
  readActivityPrefs,
  shouldRetryPollError,
  writeActivityPrefs,
} from '../lib/activityUtils.ts';
import {
  cancelChatJob,
  createThread,
  deleteThread,
  listThreads,
  pollChatJob,
  startChatJob,
  startResumeApprovalJob,
  uploadChatMedia,
} from '../lib/apiClient.ts';
import {
  classifyFileModality,
  detectModelInputModalities,
  formatFileSize,
  normalizeAccessoriesConfig,
} from '../lib/llmPresets.ts';
import { computeModelContextUsage } from '../lib/contextUtils.ts';
import { computeAutoGrowHeight, resolveScrollPinState } from '../lib/scrollUtils.ts';
import {
  type ThinkingLevel,
  formatThinkingBadge,
  getThinkingProfile,
  normalizeThinkingLevel,
} from '../lib/thinkingModes.ts';
import type {
  ChatAttachment,
  ChatMessage,
  ChatThread,
  LiveAgentActivity,
  PersistedLLMConfig,
} from '../types/agent.ts';
import { AgentActivityPanel } from './AgentActivityPanel.tsx';
import { ContextUsageRing } from './ContextUsageRing.tsx';
import { MarkdownMessage } from './MarkdownMessage.tsx';

const MEDIA_ACCEPT_ATTR =
  'image/png,image/jpeg,image/webp,image/gif,video/mp4,video/webm,video/quicktime,video/x-matroska,audio/mpeg,audio/wav,audio/ogg,audio/flac,audio/mp4,audio/aac,audio/webm';

interface ChatViewProps {
  llmConfig: PersistedLLMConfig | null;
  onPrimaryThinkingChange?: (level: ThinkingLevel) => void;
  onOpenLLMSettings: () => void;
  onOpenAccessoriesSettings?: () => void;
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
          {msg.subagentReports!.map((rep, idx) => {
            const findings = typeof rep.findings === 'string' ? rep.findings : null;
            const modality = typeof rep.modality === 'string' ? rep.modality : null;
            const modelName = typeof rep.model === 'string' ? rep.model : null;
            const providerName = typeof rep.provider === 'string' ? rep.provider : null;
            if (findings && modality) {
              return (
                <div
                  key={idx}
                  className="p-3 rounded-lg bg-[var(--paper-sunk)] border border-[var(--rule)] space-y-1.5"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
                    <span className="font-semibold uppercase tracking-wide text-[var(--signal)]">
                      {modality} Accessory Sub-Agent
                    </span>
                    {modelName && (
                      <span className="font-mono text-[10px] text-[var(--ink-faint)]">
                        {providerName ? `${providerName}: ` : ''}
                        {modelName}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-[var(--ink-soft)] whitespace-pre-wrap leading-relaxed">
                    {findings}
                  </p>
                </div>
              );
            }
            return (
              <pre
                key={idx}
                className="p-2.5 rounded-md bg-[var(--paper-sunk)] border border-[var(--rule)] font-mono text-[11px] text-[var(--ink-soft)] overflow-x-auto"
              >
                {JSON.stringify(rep, null, 2)}
              </pre>
            );
          })}
        </div>
      )}
    </div>
  );
}

function MessageAttachments({
  attachments,
  isUser,
  onPreviewImage,
}: {
  attachments: ChatAttachment[];
  isUser: boolean;
  onPreviewImage: (att: ChatAttachment) => void;
}) {
  if (!attachments || attachments.length === 0) return null;
  return (
    <div className="mb-2.5 flex flex-wrap gap-2">
      {attachments.map((att) => {
        if (att.kind === 'image') {
          return (
            <button
              key={att.id}
              type="button"
              onClick={() => onPreviewImage(att)}
              title={`Preview ${att.name}`}
              className={`group relative overflow-hidden rounded-lg border cursor-pointer text-left ${
                isUser
                  ? 'border-white/25 bg-black/15'
                  : 'border-[var(--rule)] bg-[var(--paper-sunk)]'
              }`}
            >
              <img
                src={att.url}
                alt={att.name}
                className="max-h-52 max-w-[260px] object-cover block"
              />
              <div className="px-2 py-1 text-[10px] font-mono truncate max-w-[260px] bg-black/45 text-white">
                {att.name}
              </div>
            </button>
          );
        }
        if (att.kind === 'video') {
          return (
            <div
              key={att.id}
              className={`w-full max-w-sm rounded-lg overflow-hidden border ${
                isUser
                  ? 'border-white/25 bg-black/25'
                  : 'border-[var(--rule)] bg-[var(--paper-sunk)]'
              }`}
            >
              <video
                src={att.url}
                controls
                preload="metadata"
                className="w-full max-h-56 bg-black block"
              />
              <div
                className={`px-2.5 py-1.5 flex items-center justify-between gap-2 text-[11px] ${
                  isUser ? 'text-white/90' : 'text-[var(--ink-soft)]'
                }`}
              >
                <span className="inline-flex items-center gap-1.5 truncate font-mono">
                  <VideoIcon className="w-3.5 h-3.5 shrink-0" />
                  <span className="truncate">{att.name}</span>
                </span>
                <span className="text-[10px] opacity-75 shrink-0">{formatFileSize(att.size)}</span>
              </div>
            </div>
          );
        }
        return (
          <div
            key={att.id}
            className={`w-full max-w-sm rounded-lg p-2.5 border space-y-1.5 ${
              isUser
                ? 'border-white/25 bg-white/10 text-white'
                : 'border-[var(--rule)] bg-[var(--paper-sunk)] text-[var(--ink)]'
            }`}
          >
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="inline-flex items-center gap-1.5 truncate font-mono">
                <AudioLines className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">{att.name}</span>
              </span>
              <span className="text-[10px] opacity-75 shrink-0">{formatFileSize(att.size)}</span>
            </div>
            <audio src={att.url} controls preload="metadata" className="w-full h-8" />
          </div>
        );
      })}
    </div>
  );
}

export function ChatView({
  llmConfig,
  onPrimaryThinkingChange,
  onOpenLLMSettings,
  onOpenAccessoriesSettings,
}: ChatViewProps) {
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [loadingThreads, setLoadingThreads] = useState(true);
  const [prompt, setPrompt] = useState('');
  const [pendingAttachments, setPendingAttachments] = useState<ChatAttachment[]>([]);
  const [uploadingFiles, setUploadingFiles] = useState<string[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [lightboxImage, setLightboxImage] = useState<ChatAttachment | null>(null);
  const [sending, setSending] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [liveActivity, setLiveActivity] = useState<LiveAgentActivity | null>(null);
  const [activityPrefs, setActivityPrefs] = useState<ActivityUserPrefs>(() => readActivityPrefs());
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const pollCancelledRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const activeJobIdRef = useRef<string | null>(null);
  const activeAbortRef = useRef<AbortController | null>(null);

  // Adaptive Thinking Mode & Multimodal Capability Detection
  const primaryProvider = llmConfig?.primary.provider || 'river_ai';
  const primaryModel = (llmConfig?.primary.model || '').trim();
  const hasPrimaryModel = primaryModel.length > 0;
  const primaryModalities = useMemo(
    () => detectModelInputModalities(primaryProvider, primaryModel),
    [primaryProvider, primaryModel],
  );
  const normalizedAccessories = useMemo(
    () => normalizeAccessoriesConfig(llmConfig?.accessories),
    [llmConfig?.accessories],
  );
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

  const contextUsage = useMemo(
    () => computeModelContextUsage({ thread: activeThread, liveActivity, llmConfig }),
    [activeThread, liveActivity, llmConfig],
  );

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

  // Close image lightbox on Escape
  useEffect(() => {
    if (!lightboxImage) return;
    function handleKey(e: globalThis.KeyboardEvent) {
      if (e.key === 'Escape') setLightboxImage(null);
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [lightboxImage]);

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

  async function handleAttachFiles(fileList: FileList | File[]) {
    const files = Array.from(fileList || []);
    if (files.length === 0 || sending) return;
    setError(null);

    const remainingSlots = Math.max(0, 6 - pendingAttachments.length);
    if (remainingSlots === 0) {
      setError('You can attach up to 6 media files per message.');
      return;
    }

    const toProcess = files.slice(0, remainingSlots);
    const validFiles: File[] = [];
    for (const file of toProcess) {
      const modality = classifyFileModality(file.type, file.name);
      if (!modality) {
        setError(
          `Unsupported file "${file.name}". Attach PNG, JPG, WebP, GIF, MP4, WebM, MOV, MP3, WAV, OGG, FLAC, M4A, or AAC.`,
        );
        continue;
      }
      validFiles.push(file);
    }
    if (validFiles.length === 0) return;

    const names = validFiles.map((f) => f.name);
    setUploadingFiles((prev) => [...prev, ...names]);

    try {
      const uploaded: ChatAttachment[] = [];
      for (const file of validFiles) {
        try {
          const att = await uploadChatMedia(file, activeThreadId || undefined);
          uploaded.push(att);
        } catch (err) {
          setError(err instanceof Error ? err.message : `Failed to upload ${file.name}.`);
        }
      }
      if (uploaded.length > 0) {
        setPendingAttachments((prev) => [...prev, ...uploaded].slice(0, 6));
      }
    } finally {
      setUploadingFiles((prev) => prev.filter((n) => !names.includes(n)));
    }
  }

  function handleRemoveAttachment(id: string) {
    setPendingAttachments((prev) => prev.filter((a) => a.id !== id));
  }

  function handleComposerPaste(e: ClipboardEvent<HTMLTextAreaElement>) {
    const items = e.clipboardData?.items;
    if (!items) return;
    const mediaFiles: File[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file && classifyFileModality(file.type, file.name)) {
          mediaFiles.push(file);
        }
      }
    }
    if (mediaFiles.length > 0) {
      e.preventDefault();
      void handleAttachFiles(mediaFiles);
    }
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    if (e.dataTransfer?.types?.includes('Files')) {
      e.preventDefault();
      if (!dragActive) setDragActive(true);
    }
  }

  function handleDragLeave(e: DragEvent<HTMLDivElement>) {
    if (e.currentTarget === e.target) {
      setDragActive(false);
    }
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
      void handleAttachFiles(e.dataTransfer.files);
    }
  }

  async function submitPrompt(textToSend: string, overrideAttachments?: ChatAttachment[]) {
    const clean = textToSend.trim();
    const attachmentsToSend = overrideAttachments ?? pendingAttachments;
    if ((!clean && attachmentsToSend.length === 0) || sending || uploadingFiles.length > 0) return;

    if (!hasPrimaryModel) {
      setError(
        'No Primary model selected yet. Choose your preferred Primary model in Settings → LLM Config before sending.',
      );
      return;
    }

    const displayPrompt =
      clean ||
      `Analyze the attached ${Array.from(new Set(attachmentsToSend.map((a) => a.kind))).join(', ')} file(s).`;

    setError(null);
    setSending(true);
    setStopping(false);
    setThinkingMenuOpen(false);
    setPrompt('');
    if (!overrideAttachments) {
      setPendingAttachments([]);
    }

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
      const derivedTitle =
        cleanThreadTitle(clean || attachmentsToSend[0]?.name || '') || 'New Agent Session';
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
                    content: displayPrompt,
                    timestamp: new Date().toISOString(),
                    attachments: attachmentsToSend.length > 0 ? attachmentsToSend : undefined,
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
        lastEvent:
          attachmentsToSend.length > 0
            ? 'Routing attached media and reviewing your request'
            : 'Reviewing your request and choosing the best approach',
        elapsedSeconds: 0,
        turnCount: 1,
        thinking: null,
        activityFeed: [
          {
            id: 'init-1',
            phase: 'thinking',
            text:
              attachmentsToSend.length > 0
                ? 'Routing attached media and reviewing your request'
                : 'Reviewing your request and choosing the best approach',
          },
        ],
        toolEvents: [],
        fileChanges: [],
      });

      const started = await startChatJob(
        targetId,
        clean,
        effectiveThinking,
        controller.signal,
        attachmentsToSend.length > 0 ? attachmentsToSend : undefined,
      );
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

      await pollJobUntilComplete(started.jobId, controller);
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
          if (!overrideAttachments && attachmentsToSend.length > 0) {
            setPendingAttachments(attachmentsToSend);
          }
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
        if (!overrideAttachments && attachmentsToSend.length > 0) {
          setPendingAttachments(attachmentsToSend);
        }
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

  async function pollJobUntilComplete(jobId: string, controller: AbortController) {
    let consecutivePollErrors = 0;

    while (!pollCancelledRef.current && !stopRequestedRef.current) {
      try {
        const polled = await pollChatJob(jobId, 300, controller.signal);
        consecutivePollErrors = 0;
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
      } catch (pollErr) {
        const isAbort =
          stopRequestedRef.current ||
          pollCancelledRef.current ||
          (pollErr instanceof DOMException && pollErr.name === 'AbortError') ||
          (pollErr instanceof Error && pollErr.name === 'AbortError');
        consecutivePollErrors += 1;
        if (!shouldRetryPollError(consecutivePollErrors, MAX_CONSECUTIVE_POLL_ERRORS, isAbort)) {
          throw pollErr;
        }
        const delayMs = computePollRetryDelayMs(consecutivePollErrors);
        await new Promise<void>((resolve) => {
          const timer = window.setTimeout(resolve, delayMs);
          controller.signal.addEventListener(
            'abort',
            () => {
              window.clearTimeout(timer);
              resolve();
            },
            { once: true },
          );
        });
      }
    }
  }

  async function handleApprovalDecision(decision: 'approve' | 'reject' | 'edit') {
    if (!activeThread || resuming || sending) return;
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

    const threadId = activeThread.id;
    const toolLabel = String(activeThread.pendingApproval?.tool_name || 'approved tool');
    const actionLabel =
      decision === 'reject'
        ? `Rejecting ${toolLabel}`
        : decision === 'edit'
          ? `Applying edited arguments for ${toolLabel}`
          : `Running approved tool: ${toolLabel}`;

    const controller = new AbortController();
    activeAbortRef.current = controller;
    stopRequestedRef.current = false;
    setResuming(true);
    setSending(true);
    setStopping(false);
    isPinnedToBottomRef.current = true;
    setLiveActivity({
      jobId: 'resuming',
      status: 'running',
      currentPhase: 'resuming_approval',
      lastEvent: actionLabel,
      elapsedSeconds: 0,
      turnCount: 1,
      thinking: null,
      activityFeed: [
        {
          id: 'resume-init-1',
          phase: 'resuming_approval',
          text: actionLabel,
        },
      ],
      toolEvents: [],
      fileChanges: [],
    });

    try {
      const started = await startResumeApprovalJob(
        threadId,
        decision,
        parsedArgs,
        undefined,
        controller.signal,
      );
      setEditingApproval(false);

      if (started.thread) {
        const updatedThread = started.thread;
        setThreads((prev) => prev.map((t) => (t.id === updatedThread.id ? updatedThread : t)));
      }

      if (!started.jobId) {
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

      await pollJobUntilComplete(started.jobId, controller);
    } catch (err) {
      const isAbort =
        stopRequestedRef.current ||
        (err instanceof DOMException && err.name === 'AbortError') ||
        (err instanceof Error && err.name === 'AbortError');
      if (!isAbort) {
        setError(err instanceof Error ? err.message : 'Unable to submit approval decision.');
      }
    } finally {
      activeJobIdRef.current = null;
      activeAbortRef.current = null;
      stopRequestedRef.current = false;
      setResuming(false);
      setSending(false);
      setStopping(false);
      setLiveActivity(null);
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
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className="relative flex-1 flex flex-col min-w-0 min-h-0"
      >
        {dragActive && (
          <div className="pointer-events-none absolute inset-3 z-40 rounded-2xl border-2 border-dashed border-[var(--signal)] bg-[var(--blush)]/75 backdrop-blur-[1px] flex flex-col items-center justify-center gap-2 text-center p-6">
            <Paperclip className="w-7 h-7 text-[var(--signal)]" />
            <div className="font-display text-base font-semibold text-[var(--ink)]">
              Drop image, video, or audio files to attach
            </div>
            <p className="text-xs text-[var(--ink-soft)]">
              Supported modalities go straight to your Main Agent; unsupported ones route automatically to your Accessory sub-agent.
            </p>
          </div>
        )}
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
                data-testid="active-primary-model-button"
                className={`hidden sm:inline-flex items-center gap-2 h-9 px-2.5 rounded-lg border text-xs transition-colors cursor-pointer ${
                  hasPrimaryModel
                    ? 'border-[var(--rule)] bg-[var(--paper-sunk)] hover:border-[var(--rule-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)]'
                    : 'border-[var(--signal)]/50 bg-[var(--blush)]/60 hover:border-[var(--signal)] text-[var(--signal)] font-medium'
                }`}
                title={
                  hasPrimaryModel
                    ? `Active Primary Model: ${llmConfig.primary.provider}: ${primaryModel}`
                    : 'Select your Primary LLM model in Settings'
                }
              >
                <SlidersHorizontal className="w-3.5 h-3.5 text-[var(--signal)] shrink-0" />
                <span className="font-mono truncate max-w-[200px]">
                  {hasPrimaryModel
                    ? `${llmConfig.primary.provider}: ${primaryModel}`
                    : 'Select primary model'}
                </span>
              </button>
            )}
            <button
              type="button"
              onClick={handleNewThread}
              className="md:hidden inline-flex items-center gap-1.5 h-9 px-2.5 rounded-lg bg-[var(--signal)] text-white text-xs font-semibold cursor-pointer"
            >
              <MessageSquarePlus className="w-3.5 h-3.5" />
              <span>New</span>
            </button>
            <ContextUsageRing
              usage={contextUsage}
              modelConfigured={hasPrimaryModel}
              onOpenLLMSettings={onOpenLLMSettings}
            />
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
                      {msg.attachments && msg.attachments.length > 0 && (
                        <MessageAttachments
                          attachments={msg.attachments}
                          isUser={isUser}
                          onPreviewImage={setLightboxImage}
                        />
                      )}
                      <MarkdownMessage content={msg.content} isUser={isUser} />
                      {!isUser && <MessageMetadata msg={msg} />}
                    </div>
                  </div>
                );
              })}

              {/* Pending HITL Approval Banner */}
              {activeThread.pendingApproval && !resuming && (
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
                className="mb-3 rounded-lg border border-[var(--flare)]/30 bg-[var(--flare)]/10 px-3.5 py-2 text-xs text-[var(--flare)] flex items-center justify-between gap-3"
              >
                <span>{error}</span>
                {!hasPrimaryModel && (
                  <button
                    type="button"
                    onClick={onOpenLLMSettings}
                    className="px-2.5 py-1 rounded-md border border-[var(--flare)]/40 bg-[var(--paper)] text-[var(--ink)] hover:border-[var(--signal)] font-medium cursor-pointer shrink-0"
                  >
                    Choose model
                  </button>
                )}
              </div>
            )}

            <form
              onSubmit={handleFormSubmit}
              className="rounded-xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] p-2 focus-within:border-[var(--signal)] focus-within:ring-2 focus-within:ring-[var(--signal)]/15 transition-all"
            >
              {/* Pending Multimodal Attachments Preview Tray */}
              {(pendingAttachments.length > 0 || uploadingFiles.length > 0) && (
                <div
                  data-testid="composer-attachments-tray"
                  className="mb-2 pb-2 border-b border-[var(--rule)] flex flex-wrap gap-2 px-1"
                >
                  {pendingAttachments.map((att) => {
                    const isNative = primaryModalities.has(att.kind);
                    const accCfg = normalizedAccessories[att.kind];
                    const routeLabel = isNative
                      ? 'Native'
                      : accCfg.enabled
                        ? `${att.kind[0].toUpperCase()}${att.kind.slice(1)} accessory`
                        : 'Accessory off';
                    const routeClass = isNative
                      ? 'text-[var(--jade)]'
                      : accCfg.enabled
                        ? 'text-[var(--signal)]'
                        : 'text-[var(--flare)]';

                    return (
                      <div
                        key={att.id}
                        className="group flex items-center gap-2.5 rounded-lg border border-[var(--rule)] bg-[var(--paper)] p-1.5 pr-2 max-w-[290px]"
                      >
                        {att.kind === 'image' ? (
                          <img
                            src={att.url}
                            alt={att.name}
                            className="w-10 h-10 rounded object-cover shrink-0 border border-[var(--rule)]"
                          />
                        ) : (
                          <div className="w-10 h-10 rounded bg-[var(--paper-sunk)] border border-[var(--rule)] flex items-center justify-center shrink-0 text-[var(--signal)]">
                            {att.kind === 'video' ? (
                              <VideoIcon className="w-4 h-4" />
                            ) : (
                              <AudioLines className="w-4 h-4" />
                            )}
                          </div>
                        )}
                        <div className="min-w-0 flex-1">
                          <div className="text-xs font-medium text-[var(--ink)] truncate" title={att.name}>
                            {att.name}
                          </div>
                          <div className="flex items-center gap-1.5 text-[10px] font-mono whitespace-nowrap">
                            <span className="text-[var(--ink-faint)] shrink-0">
                              {formatFileSize(att.size)}
                            </span>
                            <span className="text-[var(--rule-strong)] shrink-0">·</span>
                            {onOpenAccessoriesSettings && !isNative ? (
                              <button
                                type="button"
                                onClick={onOpenAccessoriesSettings}
                                title={`Routed through ${accCfg.provider}: ${accCfg.model} — click to edit Accessories`}
                                className={`${routeClass} hover:underline cursor-pointer truncate`}
                              >
                                {routeLabel}
                              </button>
                            ) : (
                              <span className={`${routeClass} truncate`}>{routeLabel}</span>
                            )}
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => handleRemoveAttachment(att.id)}
                          aria-label={`Remove ${att.name}`}
                          className="p-1 rounded text-[var(--ink-faint)] hover:text-[var(--flare)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer shrink-0"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    );
                  })}

                  {uploadingFiles.map((fname, idx) => (
                    <div
                      key={`${fname}-${idx}`}
                      className="flex items-center gap-2 rounded-lg border border-[var(--rule)] bg-[var(--paper-sunk)] px-2.5 py-2 text-xs text-[var(--ink-soft)]"
                    >
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--signal)] shrink-0" />
                      <span className="truncate max-w-[160px]">Uploading {fname}...</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="flex items-end gap-2">
                <input
                  ref={fileInputRef}
                  id="chat-media-input"
                  type="file"
                  multiple
                  accept={MEDIA_ACCEPT_ATTR}
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      void handleAttachFiles(e.target.files);
                    }
                    e.target.value = '';
                  }}
                />
                <button
                  type="button"
                  disabled={sending || pendingAttachments.length >= 6}
                  onClick={() => fileInputRef.current?.click()}
                  aria-label="Attach image, video, or audio"
                  title="Attach image, video, or audio (or drag & drop / paste)"
                  className="h-9 w-9 inline-flex items-center justify-center rounded-lg border border-[var(--rule)] bg-[var(--paper-sunk)] hover:border-[var(--signal)] text-[var(--ink-soft)] hover:text-[var(--signal)] disabled:opacity-45 transition-colors cursor-pointer shrink-0"
                >
                  <Paperclip className="w-4 h-4" />
                </button>

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
                  onPaste={handleComposerPaste}
                  placeholder="Message Nemo, search the web, or attach media..."
                  disabled={sending}
                  className="flex-1 min-w-0 resize-none bg-transparent px-2 py-2 text-sm leading-5 text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none min-h-[36px] max-h-[220px] overflow-hidden scrollbar-none"
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
                    title={
                      hasPrimaryModel
                        ? `Thinking mode for ${primaryModel} (${thinkingProfile.familyLabel})`
                        : 'Select a Primary model in Settings to configure thinking mode'
                    }
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
                        {hasPrimaryModel ? primaryModel : 'No primary model selected'}
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
                          {hasPrimaryModel ? (
                            <>
                              <span className="font-mono text-[var(--ink)]">{primaryModel}</span> is a
                              standard chat model and does not support configurable thinking levels.
                            </>
                          ) : (
                            'Select your Primary model in Settings to view its supported thinking modes.'
                          )}
                        </p>
                        <button
                          type="button"
                          onClick={() => {
                            setThinkingMenuOpen(false);
                            onOpenLLMSettings();
                          }}
                          className="w-full py-1.5 px-2.5 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] text-[var(--ink)] text-xs font-medium transition-colors cursor-pointer"
                        >
                          {hasPrimaryModel ? 'Switch model in Settings' : 'Select primary model'}
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
                  disabled={
                    (!prompt.trim() && pendingAttachments.length === 0) ||
                    uploadingFiles.length > 0
                  }
                  aria-label="Send message"
                  className="h-9 px-3.5 inline-flex items-center gap-1.5 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-45 text-white text-xs font-semibold transition-colors cursor-pointer shrink-0"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>Send</span>
                </button>
              )}
              </div>
            </form>
          </div>
        </div>
      </div>

      {/* Image Preview Lightbox Modal */}
      {lightboxImage && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Image preview: ${lightboxImage.name}`}
          onClick={() => setLightboxImage(null)}
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-xs flex items-center justify-center p-4 sm:p-8"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative max-w-4xl max-h-[88vh] flex flex-col items-center gap-3"
          >
            <button
              type="button"
              onClick={() => setLightboxImage(null)}
              aria-label="Close image preview"
              className="self-end inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/15 hover:bg-white/25 text-white text-xs font-medium cursor-pointer"
            >
              <X className="w-4 h-4" />
              <span>Close</span>
            </button>
            <img
              src={lightboxImage.url}
              alt={lightboxImage.name}
              className="max-h-[78vh] max-w-full rounded-xl shadow-2xl object-contain border border-white/15"
            />
            <div className="inline-flex items-center gap-2 text-xs font-mono text-white/85">
              <ImageIcon className="w-3.5 h-3.5" />
              <span>{lightboxImage.name}</span>
              <span>·</span>
              <span>{formatFileSize(lightboxImage.size)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
