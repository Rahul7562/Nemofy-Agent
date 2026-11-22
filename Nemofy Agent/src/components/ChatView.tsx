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
  dedupeFileChanges,
  readActivityPrefs,
  writeActivityPrefs,
} from '../lib/activityUtils.ts';
import { computeModelContextUsage } from '../lib/contextUtils.ts';
import type { ThreadDraftState, ThreadRunState } from '../lib/executionState.ts';
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
  ExecutionAccessMode,
  FileChangeItem,
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
  threadRun: ThreadRunState;
  threadDraft: ThreadDraftState;
  onPromptChange: (value: string) => void;
  onAttachFiles: (files: FileList | File[]) => void;
  onRemoveAttachment: (id: string) => void;
  onSendMessage: (params: {
    prompt: string;
    reasoningEffort: string;
    hasPrimaryModel: boolean;
  }) => void;
  onStopGeneration: () => void;
  onApprovalDecision: (params: {
    decision: 'approve' | 'reject' | 'edit';
    toolName?: string;
    editedArgs?: Record<string, unknown>;
  }) => void;
  onCreateThread: (title?: string) => Promise<ChatThread | null>;
  llmConfig: PersistedLLMConfig | null;
  onQuickSwitchPrimaryModel: (
    nextModel: string,
    option?: ProviderModelOption,
    options?: { persist?: boolean },
  ) => void;
  onPrimaryThinkingChange?: (level: ThinkingLevel) => void;
  accessMode: ExecutionAccessMode;
  onChangeAccessMode: (mode: ExecutionAccessMode) => void;
  onOpenLLMSettings: () => void;
  onOpenAccessoriesSettings?: () => void;
  sidebarCollapsed: boolean;
  onExpandSidebar: () => void;
  onOpenMobileSidebar: () => void;
  workspaceOpen: boolean;
  onToggleWorkspace: () => void;
  onOpenWorkspaceFile: (path: string) => void;
}

export function ChatView({
  activeThread,
  activeThreadId,
  threadsError,
  threadRun,
  threadDraft,
  onPromptChange,
  onAttachFiles,
  onRemoveAttachment,
  onSendMessage,
  onStopGeneration,
  onApprovalDecision,
  onCreateThread,
  llmConfig,
  onQuickSwitchPrimaryModel,
  onPrimaryThinkingChange,
  accessMode,
  onChangeAccessMode,
  onOpenLLMSettings,
  onOpenAccessoriesSettings,
  sidebarCollapsed,
  onExpandSidebar,
  onOpenMobileSidebar,
  workspaceOpen,
  onToggleWorkspace,
  onOpenWorkspaceFile,
}: ChatViewProps) {
  const [dragActive, setDragActive] = useState(false);
  const [lightboxImage, setLightboxImage] = useState<ChatAttachment | null>(null);
  const [activityPrefs, setActivityPrefs] = useState<ActivityUserPrefs>(() => readActivityPrefs());
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [localValidationError, setLocalValidationError] = useState<string | null>(null);

  // Local UI-only HITL approval editor state (always reset on thread switch)
  const [editingApproval, setEditingApproval] = useState(false);
  const [approvalArgsOpen, setApprovalArgsOpen] = useState(true);
  const [editedArgsJson, setEditedArgsJson] = useState('');

  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const isPinnedToBottomRef = useRef(true);
  const lastScrollTopRef = useRef(0);

  const { sending, stopping, resuming, liveActivity, error: runError } = threadRun;
  const { prompt, pendingAttachments, uploadingFiles } = threadDraft;

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

  // Reset UI-only local state and re-pin cleanly when switching threads
  useEffect(() => {
    isPinnedToBottomRef.current = true;
    setShowScrollToBottom(false);
    setEditingApproval(false);
    setEditedArgsJson('');
    setLocalValidationError(null);
    setLightboxImage(null);
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
      onAttachFiles(mediaFiles);
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
      onAttachFiles(e.dataTransfer.files);
    }
  }

  function dispatchSubmit() {
    if (sending || uploadingFiles.length > 0) return;
    setLocalValidationError(null);
    isPinnedToBottomRef.current = true;
    const effectiveThinking = thinkingProfile.supported ? selectedThinking : 'off';
    onSendMessage({
      prompt,
      reasoningEffort: effectiveThinking,
      hasPrimaryModel,
    });
  }

  function handleFormSubmit(e: FormEvent) {
    e.preventDefault();
    dispatchSubmit();
  }

  function handleTextareaKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      dispatchSubmit();
    }
  }

  function handleApprovalAction(decision: 'approve' | 'reject' | 'edit') {
    if (!activeThread || resuming || sending) return;
    setLocalValidationError(null);
    let parsedArgs: Record<string, unknown> | undefined;
    if (decision === 'edit') {
      try {
        parsedArgs = JSON.parse(editedArgsJson) as Record<string, unknown>;
      } catch {
        setLocalValidationError('Edited tool arguments must be valid JSON.');
        return;
      }
    }
    setEditingApproval(false);
    isPinnedToBottomRef.current = true;
    onApprovalDecision({
      decision,
      toolName: String(activeThread.pendingApproval?.tool_name || 'approved tool'),
      editedArgs: parsedArgs,
    });
  }

  const isEmptyThread = !activeThread || activeThread.messages.length === 0;
  const visibleError = localValidationError || runError || threadsError || null;

  const composerElement = (
    <ChatComposer
      prompt={prompt}
      onPromptChange={onPromptChange}
      onSubmit={handleFormSubmit}
      onKeyDown={handleTextareaKeyDown}
      onPaste={handleComposerPaste}
      sending={sending}
      stopping={stopping}
      onStopGeneration={onStopGeneration}
      error={visibleError}
      hasPrimaryModel={hasPrimaryModel}
      primaryModel={primaryModel}
      primaryModalities={primaryModalities}
      normalizedAccessories={normalizedAccessories}
      pendingAttachments={pendingAttachments}
      uploadingFiles={uploadingFiles}
      onAttachFiles={onAttachFiles}
      onRemoveAttachment={onRemoveAttachment}
      thinkingProfile={thinkingProfile}
      selectedThinking={selectedThinking}
      onSelectThinking={handleSelectThinking}
      accessMode={accessMode}
      onChangeAccessMode={onChangeAccessMode}
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
      data-active-thread-id={activeThreadId || ''}
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
            <p
              data-testid="empty-chat-subtitle"
              className="mt-2 text-sm text-[var(--ink-soft)] text-center max-w-xl leading-relaxed"
            >
              Build inside a workspace.
            </p>

            {/* Floating Two-Row Composer */}
            <div className="mt-7 w-full">{composerElement}</div>
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
                        disabled={resuming || sending}
                        onClick={() => handleApprovalAction(editingApproval ? 'edit' : 'approve')}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-[var(--jade)] hover:opacity-90 disabled:opacity-50 text-white text-xs font-semibold transition-opacity cursor-pointer"
                      >
                        <Check className="w-3.5 h-3.5" />
                        <span>{editingApproval ? 'Save args & continue' : 'Approve & run'}</span>
                      </button>

                      {!editingApproval ? (
                        <button
                          type="button"
                          disabled={resuming || sending}
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
                        disabled={resuming || sending}
                        onClick={() => handleApprovalAction('reject')}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-[var(--flare)]/40 text-[var(--flare)] hover:bg-[var(--flare)]/10 text-xs font-medium transition-colors cursor-pointer"
                      >
                        <X className="w-3.5 h-3.5" />
                        <span>Reject</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Live Streaming Agent Activity (strictly scoped to this thread) */}
              {sending && liveActivity && (
                <div className="flex items-start gap-3.5 sm:gap-4" data-testid="live-agent-activity">
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
