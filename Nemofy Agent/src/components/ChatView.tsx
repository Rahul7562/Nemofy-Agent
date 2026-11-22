import {
  ArrowDown,
  Check,
  ChevronDown,
  ChevronRight,
  Image as ImageIcon,
  Paperclip,
  ShieldAlert,
  X,
} from 'lucide-react';
import {
  type ClipboardEvent,
  type DragEvent,
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  type ActivityUserPrefs,
  MAX_CONSECUTIVE_POLL_ERRORS,
  computePollRetryDelayMs,
  dedupeFileChanges,
  readActivityPrefs,
  shouldRetryPollError,
  writeActivityPrefs,
} from '../lib/activityUtils.ts';
import {
  cancelChatJob,
  pollChatJob,
  startChatJob,
  startResumeApprovalJob,
  uploadChatMedia,
} from '../lib/apiClient.ts';
import { computeModelContextUsage } from '../lib/contextUtils.ts';
import {
  classifyFileModality,
  detectModelInputModalities,
  formatFileSize,
  normalizeAccessoriesConfig,
} from '../lib/llmPresets.ts';
import { isNearBottom, resolveScrollPinState } from '../lib/scrollUtils.ts';
import {
  type ThinkingLevel,
  getThinkingProfile,
  normalizeThinkingLevel,
} from '../lib/thinkingModes.ts';
import type {
  ChatAttachment,
  ChatThread,
  FileChangeItem,
  LiveAgentActivity,
  PersistedLLMConfig,
  ProviderModelOption,
} from '../types/agent.ts';
import { AgentActivityPanel } from './AgentActivityPanel.tsx';
import { ChatComposer } from './ChatComposer.tsx';
import { ChatHeader } from './ChatHeader.tsx';
import { ChatMessageItem } from './ChatMessageItem.tsx';
import { NemofyLogo } from './NemofyLogo.tsx';

interface ChatViewProps {
  activeThread: ChatThread | null;
  activeThreadId: string | null;
  threadsError?: string | null;
  onCreateThread: (title?: string) => Promise<ChatThread | null>;
  onUpsertThread: (thread: ChatThread, activate?: boolean) => void;
  onUpdateThreadById: (id: string, updater: (thread: ChatThread) => ChatThread) => void;
  llmConfig: PersistedLLMConfig | null;
  onQuickSwitchPrimaryModel: (nextModel: string, option?: ProviderModelOption) => void;
  onPrimaryThinkingChange?: (level: ThinkingLevel) => void;
  onOpenLLMSettings: () => void;
  onOpenAccessoriesSettings?: () => void;
  sidebarCollapsed: boolean;
  onExpandSidebar: () => void;
  onOpenMobileSidebar: () => void;
  workspaceOpen: boolean;
  onToggleWorkspace: () => void;
  onOpenWorkspaceFile: (path: string) => void;
  onWorkspaceFilesChanged?: () => void;
}

function cleanThreadTitle(raw: string): string {
  return raw
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[*_~`#>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 56);
}

const STARTER_PROMPTS: Array<{ title: string; subtitle: string; prompt: string }> = [
  {
    title: 'Inspect /workspace',
    subtitle: 'Summarize current files and project structure',
    prompt: 'Inspect /workspace files and summarize what is currently in my project.',
  },
  {
    title: 'Compare tech stacks',
    subtitle: 'Markdown table + ASCII bar chart of adoption',
    prompt:
      'Compare Python, TypeScript, and Rust in a Markdown data table and include an ASCII bar chart of developer adoption.',
  },
  {
    title: 'SaaS launch checklist',
    subtitle: 'Actionable 5-step rollout plan',
    prompt: 'Draft a clear, actionable 5-step launch checklist for a new SaaS product.',
  },
  {
    title: 'Write & verify code',
    subtitle: 'Create a utility script and test it in sandbox',
    prompt:
      'Write a clean Python utility script in /workspace to analyze JSON log metrics and verify it with sample data.',
  },
];

export function ChatView({
  activeThread,
  activeThreadId,
  threadsError,
  onCreateThread,
  onUpsertThread,
  onUpdateThreadById,
  llmConfig,
  onQuickSwitchPrimaryModel,
  onPrimaryThinkingChange,
  onOpenLLMSettings,
  onOpenAccessoriesSettings,
  sidebarCollapsed,
  onExpandSidebar,
  onOpenMobileSidebar,
  workspaceOpen,
  onToggleWorkspace,
  onOpenWorkspaceFile,
  onWorkspaceFilesChanged,
}: ChatViewProps) {
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
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);

  const pollCancelledRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const activeJobIdRef = useRef<string | null>(null);
  const activeAbortRef = useRef<AbortController | null>(null);

  // HITL approval state
  const [editingApproval, setEditingApproval] = useState(false);
  const [approvalArgsOpen, setApprovalArgsOpen] = useState(true);
  const [editedArgsJson, setEditedArgsJson] = useState('');
  const [resuming, setResuming] = useState(false);

  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const isPinnedToBottomRef = useRef(true);
  const lastScrollTopRef = useRef(0);

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

  const contextUsage = useMemo(
    () => computeModelContextUsage({ thread: activeThread, liveActivity, llmConfig }),
    [activeThread, liveActivity, llmConfig],
  );

  const workspaceChangesCount = useMemo(() => {
    const all: FileChangeItem[] = [];
    if (activeThread) {
      for (const msg of activeThread.messages) {
        if (msg.fileChanges && msg.fileChanges.length > 0) {
          all.push(...msg.fileChanges);
        }
      }
    }
    if (liveActivity?.fileChanges && liveActivity.fileChanges.length > 0) {
      all.push(...liveActivity.fileChanges);
    }
    const uniquePaths = new Set(dedupeFileChanges(all).map((c) => c.path));
    return uniquePaths.size;
  }, [activeThread, liveActivity?.fileChanges]);

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
    onPrimaryThinkingChange?.(level);
  }

  function scrollMessagesToBottom(smooth = false) {
    const el = messagesScrollRef.current;
    if (!el) return;
    if (smooth && typeof el.scrollTo === 'function') {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    } else {
      el.scrollTop = el.scrollHeight;
    }
    lastScrollTopRef.current = el.scrollHeight;
    isPinnedToBottomRef.current = true;
    setShowScrollToBottom(false);
  }

  function handleMessagesScroll() {
    const el = messagesScrollRef.current;
    if (!el) return;
    const metrics = {
      scrollTop: el.scrollTop,
      scrollHeight: el.scrollHeight,
      clientHeight: el.clientHeight,
    };
    isPinnedToBottomRef.current = resolveScrollPinState(
      metrics,
      lastScrollTopRef.current,
      isPinnedToBottomRef.current,
      48,
    );
    lastScrollTopRef.current = el.scrollTop;
    setShowScrollToBottom(!isNearBottom(metrics, 96));
  }

  // Re-pin cleanly when switching threads
  useEffect(() => {
    isPinnedToBottomRef.current = true;
    setShowScrollToBottom(false);
    setEditingApproval(false);
    setError(null);
    requestAnimationFrame(() => scrollMessagesToBottom(false));
  }, [activeThreadId]);

  // Follow new messages or activity ONLY when the user has not scrolled up
  useEffect(() => {
    if (!isPinnedToBottomRef.current) return;
    scrollMessagesToBottom(false);
  }, [
    activeThread?.messages.length,
    sending,
    liveActivity?.toolEvents.length,
    liveActivity?.fileChanges.length,
  ]);

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
          onUpsertThread(res.thread, true);
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
            onUpsertThread(polled.thread, true);
            const lastMsg = polled.thread.messages[polled.thread.messages.length - 1];
            if (lastMsg?.fileChanges && lastMsg.fileChanges.length > 0) {
              onWorkspaceFilesChanged?.();
            }
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

  async function submitPrompt(textToSend: string, overrideAttachments?: ChatAttachment[]) {
    const clean = textToSend.trim();
    const attachmentsToSend = overrideAttachments ?? pendingAttachments;
    if ((!clean && attachmentsToSend.length === 0) || sending || uploadingFiles.length > 0) return;

    if (!hasPrimaryModel) {
      setError(
        'No Primary model selected yet. Choose your preferred Primary model in the header or Settings → LLM Config before sending.',
      );
      return;
    }

    const displayPrompt =
      clean ||
      `Analyze the attached ${Array.from(new Set(attachmentsToSend.map((a) => a.kind))).join(', ')} file(s).`;

    setError(null);
    setSending(true);
    setStopping(false);
    setPrompt('');
    if (!overrideAttachments) {
      setPendingAttachments([]);
    }

    stopRequestedRef.current = false;
    activeJobIdRef.current = null;
    const controller = new AbortController();
    activeAbortRef.current = controller;

    isPinnedToBottomRef.current = true;

    const optimisticId = `temp-${Date.now()}`;
    let targetId = activeThreadId;
    let committedOnServer = false;

    try {
      const derivedTitle =
        cleanThreadTitle(clean || attachmentsToSend[0]?.name || '') || 'New Agent Session';
      if (!targetId) {
        const created = await onCreateThread(derivedTitle);
        if (!created) {
          throw new Error('Unable to create session.');
        }
        targetId = created.id;
      }

      // Optimistic user message display
      onUpdateThreadById(targetId, (t) => ({
        ...t,
        title: t.messages.length === 0 ? derivedTitle : t.title,
        updatedAt: new Date().toISOString(),
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
      }));

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

      onUpsertThread(started.thread, true);

      if (!started.jobId) {
        setLiveActivity(null);
        return;
      }

      activeJobIdRef.current = started.jobId;

      if (stopRequestedRef.current) {
        const stopped = await cancelChatJob(started.jobId);
        if (stopped.thread) {
          onUpsertThread(stopped.thread, false);
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
        if (!committedOnServer && targetId) {
          onUpdateThreadById(targetId, (t) => ({
            ...t,
            messages: t.messages.filter((m) => m.id !== optimisticId),
          }));
          setPrompt(clean);
          if (!overrideAttachments && attachmentsToSend.length > 0) {
            setPendingAttachments(attachmentsToSend);
          }
        }
        return;
      }

      if (!committedOnServer && targetId) {
        onUpdateThreadById(targetId, (t) => ({
          ...t,
          messages: t.messages.filter((m) => m.id !== optimisticId),
        }));
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
        onUpsertThread(started.thread, false);
      }

      if (!started.jobId) {
        setLiveActivity(null);
        return;
      }

      activeJobIdRef.current = started.jobId;

      if (stopRequestedRef.current) {
        const stopped = await cancelChatJob(started.jobId);
        if (stopped.thread) {
          onUpsertThread(stopped.thread, false);
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

  const isEmptyThread = !activeThread || activeThread.messages.length === 0;
  const visibleError = error || threadsError || null;

  const composerElement = (
    <ChatComposer
      prompt={prompt}
      onPromptChange={setPrompt}
      onSubmit={handleFormSubmit}
      onKeyDown={handleTextareaKeyDown}
      onPaste={handleComposerPaste}
      sending={sending}
      stopping={stopping}
      onStopGeneration={() => void handleStopGeneration()}
      error={visibleError}
      hasPrimaryModel={hasPrimaryModel}
      primaryModel={primaryModel}
      primaryModalities={primaryModalities}
      normalizedAccessories={normalizedAccessories}
      pendingAttachments={pendingAttachments}
      uploadingFiles={uploadingFiles}
      onAttachFiles={(files) => void handleAttachFiles(files)}
      onRemoveAttachment={handleRemoveAttachment}
      thinkingProfile={thinkingProfile}
      selectedThinking={selectedThinking}
      onSelectThinking={handleSelectThinking}
      onOpenLLMSettings={onOpenLLMSettings}
      onOpenAccessoriesSettings={onOpenAccessoriesSettings}
      autoFocus={isEmptyThread}
    />
  );

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className="relative flex flex-col flex-1 h-full min-w-0 min-h-0 bg-[var(--paper)]"
    >
      {/* Drag-and-drop overlay */}
      {dragActive && (
        <div className="pointer-events-none absolute inset-3 z-40 rounded-2xl border-2 border-dashed border-[var(--signal)] bg-[var(--blush)]/80 backdrop-blur-[1px] flex flex-col items-center justify-center gap-2 text-center p-6">
          <Paperclip className="w-7 h-7 text-[var(--signal)]" />
          <div className="font-display text-base font-semibold text-[var(--ink)]">
            Drop image, video, or audio files to attach
          </div>
          <p className="text-xs text-[var(--ink-soft)] max-w-md">
            Supported modalities go straight to your Main Agent; unsupported ones route automatically to your Accessory sub-agent.
          </p>
        </div>
      )}

      {/* Minimal Top Header */}
      <ChatHeader
        sidebarCollapsed={sidebarCollapsed}
        onExpandSidebar={onExpandSidebar}
        onOpenMobileSidebar={onOpenMobileSidebar}
        onNewThread={() => void onCreateThread()}
        llmConfig={llmConfig}
        onQuickSwitchPrimaryModel={onQuickSwitchPrimaryModel}
        onOpenLLMSettings={onOpenLLMSettings}
        contextUsage={contextUsage}
        workspaceOpen={workspaceOpen}
        onToggleWorkspace={onToggleWorkspace}
        workspaceChangesCount={workspaceChangesCount}
      />

      {isEmptyThread ? (
        /* Centered Empty State + Floating Two-Row Composer */
        <div className="flex-1 overflow-y-auto scrollbar-none flex flex-col items-center justify-center px-4 sm:px-6 py-8">
          <div className="w-full max-w-3xl mx-auto flex flex-col items-center">
            <div className="mb-4 w-12 h-12 rounded-2xl bg-[var(--paper-sunk)] border border-[var(--rule)] flex items-center justify-center">
              <NemofyLogo size={28} />
            </div>

            <h1 className="font-display text-2xl sm:text-3xl font-semibold tracking-tight text-[var(--ink)] text-center [text-wrap:balance]">
              Hi, I&apos;m Nemo. How can I help today?
            </h1>
            <p className="mt-2 text-sm text-[var(--ink-soft)] text-center max-w-xl leading-relaxed">
              Ask questions, analyze media, run tools, or build inside{' '}
              <code className="font-mono text-xs px-1.5 py-0.5 rounded bg-[var(--paper-sunk)] border border-[var(--rule)] text-[var(--ink)]">
                /workspace
              </code>
              .
            </p>

            {/* Compact 2x2 Grid of Starter Prompts directly above composer */}
            <div className="mt-7 w-full grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {STARTER_PROMPTS.map((item) => (
                <button
                  key={item.title}
                  type="button"
                  onClick={() => void submitPrompt(item.prompt)}
                  className="group text-left px-4 py-3 rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] hover:border-[var(--rule-strong)] hover:bg-[var(--paper-sunk)]/75 transition-colors cursor-pointer"
                >
                  <div className="text-[13px] font-semibold text-[var(--ink)] group-hover:text-[var(--signal)] transition-colors">
                    {item.title}
                  </div>
                  <div className="mt-0.5 text-xs text-[var(--ink-soft)] line-clamp-1">
                    {item.subtitle}
                  </div>
                </button>
              ))}
            </div>

            {/* Floating Two-Row Composer */}
            <div className="mt-4 w-full">{composerElement}</div>
          </div>
        </div>
      ) : (
        /* Active Conversation Stream + Floating Bottom Composer */
        <>
          <div
            ref={messagesScrollRef}
            onScroll={handleMessagesScroll}
            onWheel={(e) => {
              if (e.deltaY < 0) {
                isPinnedToBottomRef.current = false;
              }
            }}
            className="flex-1 overflow-y-auto scrollbar-none px-4 sm:px-6 pt-6 pb-8"
          >
            <div className="max-w-3xl mx-auto space-y-7">
              {activeThread.messages.map((msg) => (
                <ChatMessageItem
                  key={msg.id}
                  message={msg}
                  activityPrefs={activityPrefs}
                  onUpdateActivityPrefs={handleUpdateActivityPrefs}
                  onPreviewImage={setLightboxImage}
                  onOpenWorkspaceFile={onOpenWorkspaceFile}
                />
              ))}

              {/* Pending HITL Approval Inline Action Card */}
              {activeThread.pendingApproval && !resuming && (
                <div
                  data-testid="hitl-approval-card"
                  className="rounded-2xl border border-[var(--honey)]/55 bg-[var(--paper-raised)] shadow-sm overflow-hidden"
                >
                  <div className="px-4 py-3.5 bg-[var(--honey)]/8 border-b border-[var(--rule)] flex items-start gap-3">
                    <div className="w-8 h-8 rounded-lg bg-[var(--honey)]/15 text-[var(--honey)] flex items-center justify-center shrink-0 mt-0.5">
                      <ShieldAlert className="w-4 h-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="font-display text-sm font-semibold text-[var(--ink)]">
                          Approval required
                        </h3>
                        <span className="font-mono text-xs font-semibold text-[var(--honey)]">
                          {String(activeThread.pendingApproval.tool_name || 'sensitive_action')}
                        </span>
                      </div>
                      {activeThread.pendingApproval.risk_reason && (
                        <p className="mt-1 text-xs text-[var(--ink-soft)] leading-relaxed">
                          {String(activeThread.pendingApproval.risk_reason)}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="p-4 space-y-3">
                    <div>
                      <button
                        type="button"
                        onClick={() => setApprovalArgsOpen((v) => !v)}
                        aria-expanded={approvalArgsOpen || editingApproval}
                        className="inline-flex items-center gap-1.5 text-xs font-medium text-[var(--ink-soft)] hover:text-[var(--ink)] cursor-pointer mb-2"
                      >
                        {approvalArgsOpen || editingApproval ? (
                          <ChevronDown className="w-3.5 h-3.5" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5" />
                        )}
                        <span>Tool arguments (JSON)</span>
                      </button>

                      {(approvalArgsOpen || editingApproval) &&
                        (editingApproval ? (
                          <textarea
                            aria-label="Edit tool arguments JSON"
                            value={editedArgsJson}
                            onChange={(e) => setEditedArgsJson(e.target.value)}
                            rows={6}
                            spellCheck={false}
                            className="w-full p-3 rounded-xl border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                          />
                        ) : (
                          <pre className="p-3 rounded-xl bg-[var(--paper-sunk)] border border-[var(--rule)] font-mono text-xs text-[var(--ink-soft)] overflow-x-auto leading-relaxed">
                            {JSON.stringify(activeThread.pendingApproval.arguments || {}, null, 2)}
                          </pre>
                        ))}
                    </div>

                    <div className="flex flex-wrap items-center gap-2 pt-1">
                      <button
                        type="button"
                        disabled={resuming}
                        onClick={() =>
                          void handleApprovalDecision(editingApproval ? 'edit' : 'approve')
                        }
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-[var(--jade)] hover:opacity-90 disabled:opacity-50 text-white text-xs font-semibold transition-opacity cursor-pointer"
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
                            setApprovalArgsOpen(true);
                            setEditingApproval(true);
                          }}
                          className="px-3.5 py-2 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-xs font-medium text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer"
                        >
                          Edit arguments
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => setEditingApproval(false)}
                          className="px-3.5 py-2 rounded-lg border border-[var(--rule)] text-xs text-[var(--ink-soft)] hover:text-[var(--ink)] cursor-pointer"
                        >
                          Cancel edit
                        </button>
                      )}

                      <button
                        type="button"
                        disabled={resuming}
                        onClick={() => void handleApprovalDecision('reject')}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-[var(--flare)]/40 text-[var(--flare)] hover:bg-[var(--flare)]/10 text-xs font-medium transition-colors cursor-pointer"
                      >
                        <X className="w-3.5 h-3.5" />
                        <span>Reject</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Live Streaming Agent Activity */}
              {sending && liveActivity && (
                <div className="flex items-start gap-3.5 sm:gap-4">
                  <div className="w-7 h-7 mt-0.5 rounded-lg bg-[var(--paper-sunk)] border border-[var(--rule)] flex items-center justify-center shrink-0">
                    <NemofyLogo size={17} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <AgentActivityPanel
                      mode="live"
                      activity={liveActivity}
                      prefs={activityPrefs}
                      onUpdatePrefs={handleUpdateActivityPrefs}
                      onOpenWorkspaceFile={onOpenWorkspaceFile}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Floating Scroll-to-Bottom Pill & Floating Two-Row Composer */}
          <div className="relative px-4 pb-3 sm:pb-5 pt-1 shrink-0 bg-gradient-to-t from-[var(--paper)] via-[var(--paper)] to-transparent">
            {showScrollToBottom && (
              <div className="absolute -top-11 left-1/2 -translate-x-1/2 z-20">
                <button
                  type="button"
                  onClick={() => scrollMessagesToBottom(true)}
                  aria-label="Scroll to bottom"
                  className="inline-flex items-center gap-1.5 h-8 px-3 rounded-full border border-[var(--rule-strong)] bg-[var(--paper-raised)] hover:bg-[var(--paper-sunk)] text-xs font-medium text-[var(--ink)] shadow-md transition-all cursor-pointer whitespace-nowrap"
                >
                  <ArrowDown className="w-3.5 h-3.5" />
                  <span>Scroll to bottom</span>
                </button>
              </div>
            )}

            {composerElement}
          </div>
        </>
      )}

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
