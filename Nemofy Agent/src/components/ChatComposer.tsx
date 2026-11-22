import {
  ArrowUp,
  AudioLines,
  Brain,
  Check,
  ChevronDown,
  ChevronUp,
  Loader2,
  Paperclip,
  Shield,
  ShieldCheck,
  Square,
  Video as VideoIcon,
  X,
} from 'lucide-react';
import {
  type ClipboardEvent,
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { formatFileSize } from '../lib/llmPresets.ts';
import { computeAutoGrowHeight } from '../lib/scrollUtils.ts';
import type { ModelThinkingProfile, ThinkingLevel } from '../lib/thinkingModes.ts';
import type {
  AccessoriesConfig,
  ChatAttachment,
  ExecutionAccessMode,
  MediaModality,
} from '../types/agent.ts';

const MEDIA_ACCEPT_ATTR =
  'image/png,image/jpeg,image/webp,image/gif,video/mp4,video/webm,video/quicktime,video/x-matroska,audio/mpeg,audio/wav,audio/ogg,audio/flac,audio/mp4,audio/aac,audio/webm';

const ACCESS_MODE_OPTIONS: Array<{
  value: ExecutionAccessMode;
  label: string;
  shortLabel: string;
  description: string;
}> = [
  {
    value: 'default',
    label: 'Default',
    shortLabel: 'Default',
    description: 'Ask for approval before running sensitive commands and shell actions.',
  },
  {
    value: 'full_access',
    label: 'Full Access',
    shortLabel: 'Full Access',
    description:
      'Auto-approve command execution requests within the workspace sandbox without prompts.',
  },
];

interface ChatComposerProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  onSubmit: (e: FormEvent) => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  onPaste: (e: ClipboardEvent<HTMLTextAreaElement>) => void;
  sending: boolean;
  stopping: boolean;
  onStopGeneration: () => void;
  error: string | null;
  hasPrimaryModel: boolean;
  primaryModel: string;
  primaryModalities: Set<'text' | MediaModality>;
  normalizedAccessories: AccessoriesConfig;
  pendingAttachments: ChatAttachment[];
  uploadingFiles: string[];
  onAttachFiles: (files: FileList | File[]) => void;
  onRemoveAttachment: (id: string) => void;
  thinkingProfile: ModelThinkingProfile;
  selectedThinking: ThinkingLevel;
  onSelectThinking: (level: ThinkingLevel) => void;
  accessMode: ExecutionAccessMode;
  onChangeAccessMode: (mode: ExecutionAccessMode) => void;
  onOpenLLMSettings: () => void;
  onOpenAccessoriesSettings?: () => void;
  autoFocus?: boolean;
}

export function ChatComposer({
  prompt,
  onPromptChange,
  onSubmit,
  onKeyDown,
  onPaste,
  sending,
  stopping,
  onStopGeneration,
  error,
  hasPrimaryModel,
  primaryModel,
  primaryModalities,
  normalizedAccessories,
  pendingAttachments,
  uploadingFiles,
  onAttachFiles,
  onRemoveAttachment,
  thinkingProfile,
  selectedThinking,
  onSelectThinking,
  accessMode,
  onChangeAccessMode,
  onOpenLLMSettings,
  onOpenAccessoriesSettings,
  autoFocus = false,
}: ChatComposerProps) {
  const [thinkingMenuOpen, setThinkingMenuOpen] = useState(false);
  const [accessMenuOpen, setAccessMenuOpen] = useState(false);
  const thinkingMenuRef = useRef<HTMLDivElement>(null);
  const accessMenuRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const { heightPx, isCapped } = computeAutoGrowHeight(el.scrollHeight, 44, 220);
    el.style.height = `${heightPx}px`;
    el.style.overflowY = isCapped ? 'auto' : 'hidden';
  }, [prompt]);

  useEffect(() => {
    if (autoFocus && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [autoFocus]);

  useEffect(() => {
    if (!thinkingMenuOpen && !accessMenuOpen) return;
    function handleOutside(e: MouseEvent) {
      const target = e.target as Node;
      if (thinkingMenuOpen && thinkingMenuRef.current && !thinkingMenuRef.current.contains(target)) {
        setThinkingMenuOpen(false);
      }
      if (accessMenuOpen && accessMenuRef.current && !accessMenuRef.current.contains(target)) {
        setAccessMenuOpen(false);
      }
    }
    function handleEscape(e: globalThis.KeyboardEvent) {
      if (e.key === 'Escape') {
        setThinkingMenuOpen(false);
        setAccessMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [thinkingMenuOpen, accessMenuOpen]);

  const canSend =
    (prompt.trim().length > 0 || pendingAttachments.length > 0) && uploadingFiles.length === 0;

  return (
    <div className="w-full max-w-3xl mx-auto">
      {error && (
        <div
          role="alert"
          className="mb-3 rounded-xl border border-[var(--flare)]/30 bg-[var(--flare)]/10 px-3.5 py-2.5 text-xs text-[var(--flare)] flex items-center justify-between gap-3"
        >
          <span>{error}</span>
          {!hasPrimaryModel && (
            <button
              type="button"
              onClick={onOpenLLMSettings}
              className="px-2.5 py-1 rounded-lg border border-[var(--flare)]/40 bg-[var(--paper)] text-[var(--ink)] hover:border-[var(--signal)] font-medium cursor-pointer shrink-0"
            >
              Choose model
            </button>
          )}
        </div>
      )}

      <form
        onSubmit={onSubmit}
        className="rounded-2xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-[0_2px_12px_rgba(22,28,45,0.06)] dark:shadow-[0_4px_20px_rgba(0,0,0,0.32)] focus-within:border-[var(--signal)] focus-within:ring-2 focus-within:ring-[var(--signal)]/15 transition-all"
      >
        {/* Integrated Pending Multimodal Attachments Preview Tray */}
        {(pendingAttachments.length > 0 || uploadingFiles.length > 0) && (
          <div
            data-testid="composer-attachments-tray"
            className="px-3.5 pt-3 pb-2.5 border-b border-[var(--rule)] flex flex-wrap gap-2"
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
                  className="group flex items-center gap-2.5 rounded-xl border border-[var(--rule)] bg-[var(--paper)] p-1.5 pr-2 max-w-[290px]"
                >
                  {att.kind === 'image' ? (
                    <img
                      src={att.url}
                      alt={att.name}
                      className="w-10 h-10 rounded-lg object-cover shrink-0 border border-[var(--rule)]"
                    />
                  ) : (
                    <div className="w-10 h-10 rounded-lg bg-[var(--paper-sunk)] border border-[var(--rule)] flex items-center justify-center shrink-0 text-[var(--signal)]">
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
                      <span className="text-[var(--ink-faint)] tabular-nums shrink-0">
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
                    onClick={() => onRemoveAttachment(att.id)}
                    aria-label={`Remove ${att.name}`}
                    className="p-1 rounded-md text-[var(--ink-faint)] hover:text-[var(--flare)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer shrink-0"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}

            {uploadingFiles.map((fname, idx) => (
              <div
                key={`${fname}-${idx}`}
                className="flex items-center gap-2 rounded-xl border border-[var(--rule)] bg-[var(--paper-sunk)] px-3 py-2 text-xs text-[var(--ink-soft)]"
              >
                <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--signal)] shrink-0" />
                <span className="truncate max-w-[160px]">Uploading {fname}...</span>
              </div>
            ))}
          </div>
        )}

        {/* Row 1: Full-width auto-growing textarea */}
        <div className="px-4 pt-3 pb-1">
          <label htmlFor="chat-composer" className="sr-only">
            Message Nemo
          </label>
          <textarea
            ref={textareaRef}
            id="chat-composer"
            rows={1}
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={onKeyDown}
            onPaste={onPaste}
            placeholder="Message Nemo, search the web, or attach media..."
            disabled={sending}
            className="w-full resize-none bg-transparent text-[15px] leading-6 text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none min-h-[44px] max-h-[220px] overflow-hidden scrollbar-none"
          />
        </div>

        {/* Row 2: Quiet two-sided toolbar */}
        <div className="px-3 pb-2.5 pt-1 flex items-center justify-between gap-2">
          {/* Left: Attach Media */}
          <div className="flex items-center gap-1">
            <input
              ref={fileInputRef}
              id="chat-media-input"
              type="file"
              multiple
              accept={MEDIA_ACCEPT_ATTR}
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  onAttachFiles(e.target.files);
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
              className="h-9 w-9 inline-flex items-center justify-center rounded-full text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] disabled:opacity-40 transition-colors cursor-pointer shrink-0"
            >
              <Paperclip className="w-4 h-4" />
            </button>
          </div>

          {/* Right: Adaptive Thinking Mode + Execution Access Mode + Circular Send/Stop Button */}
          <div className="flex items-center gap-1.5 sm:gap-2">
            <div ref={thinkingMenuRef} className="relative shrink-0">
              <button
                type="button"
                data-testid="thinking-mode-trigger"
                onClick={() => {
                  setAccessMenuOpen(false);
                  setThinkingMenuOpen((v) => !v);
                }}
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
                className={`h-8 px-2.5 inline-flex items-center gap-1.5 rounded-full border text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
                  !thinkingProfile.supported
                    ? 'border-[var(--rule)] bg-transparent text-[var(--ink-faint)] hover:text-[var(--ink-soft)] hover:bg-[var(--paper-sunk)]'
                    : selectedThinking !== 'off'
                      ? 'border-[var(--signal)]/40 bg-[var(--blush)] text-[var(--signal)] hover:border-[var(--signal)]'
                      : 'border-[var(--rule)] bg-transparent text-[var(--ink-soft)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)]'
                }`}
              >
                <Brain className="w-3.5 h-3.5 shrink-0" />
                <span className="max-w-[78px] truncate">
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
                  className="absolute bottom-full right-0 mb-2 w-64 rounded-2xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-xl overflow-hidden z-30"
                >
                  <div className="px-3.5 py-2.5 border-b border-[var(--rule)] bg-[var(--paper-sunk)]">
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
                    <div className="p-1.5 space-y-0.5 max-h-[360px] overflow-y-auto scrollbar-none">
                      {thinkingProfile.options.map((opt) => {
                        const isSelected = selectedThinking === opt.value;
                        return (
                          <button
                            key={opt.value}
                            type="button"
                            role="option"
                            data-testid={`thinking-option-${opt.value}`}
                            aria-selected={isSelected}
                            onClick={() => {
                              onSelectThinking(opt.value);
                              setThinkingMenuOpen(false);
                            }}
                            className={`w-full px-2.5 py-2 rounded-xl text-left flex items-start justify-between gap-2 transition-colors cursor-pointer ${
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

            {/* Execution Access Mode (Default / Full Access) directly beside Thinking Mode */}
            <div ref={accessMenuRef} className="relative shrink-0">
              <button
                type="button"
                data-testid="access-mode-trigger"
                data-access-mode={accessMode}
                onClick={() => {
                  setThinkingMenuOpen(false);
                  setAccessMenuOpen((v) => !v);
                }}
                aria-haspopup="listbox"
                aria-expanded={accessMenuOpen}
                aria-label={`Execution access mode: ${
                  accessMode === 'full_access' ? 'Full Access' : 'Default'
                }`}
                title={
                  accessMode === 'full_access'
                    ? 'Full Access: command approval prompts are automatically approved inside the sandbox'
                    : 'Default: sensitive commands require your explicit approval before executing'
                }
                className={`h-8 px-2.5 inline-flex items-center gap-1.5 rounded-full border text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
                  accessMode === 'full_access'
                    ? 'border-[var(--honey)]/60 bg-[var(--honey)]/12 text-[var(--honey)] hover:border-[var(--honey)]'
                    : 'border-[var(--rule)] bg-transparent text-[var(--ink-soft)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)]'
                }`}
              >
                {accessMode === 'full_access' ? (
                  <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
                ) : (
                  <Shield className="w-3.5 h-3.5 shrink-0" />
                )}
                <span>{accessMode === 'full_access' ? 'Full Access' : 'Default'}</span>
                {accessMenuOpen ? (
                  <ChevronDown className="w-3 h-3 opacity-70 shrink-0" />
                ) : (
                  <ChevronUp className="w-3 h-3 opacity-70 shrink-0" />
                )}
              </button>

              {accessMenuOpen && (
                <div
                  role="listbox"
                  aria-label="Select execution access mode"
                  className="absolute bottom-full right-0 mb-2 w-64 rounded-2xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-xl overflow-hidden z-30"
                >
                  <div className="px-3.5 py-2.5 border-b border-[var(--rule)] bg-[var(--paper-sunk)]">
                    <div className="text-[11px] font-semibold text-[var(--ink)]">
                      Execution Access Mode
                    </div>
                    <div className="text-[10px] text-[var(--ink-faint)] mt-0.5">
                      Takes effect immediately without restarting the session.
                    </div>
                  </div>

                  <div className="p-1.5 space-y-0.5">
                    {ACCESS_MODE_OPTIONS.map((opt) => {
                      const isSelected = accessMode === opt.value;
                      const isFull = opt.value === 'full_access';
                      return (
                        <button
                          key={opt.value}
                          type="button"
                          role="option"
                          data-testid={`access-mode-option-${opt.value}`}
                          aria-selected={isSelected}
                          onClick={() => {
                            onChangeAccessMode(opt.value);
                            setAccessMenuOpen(false);
                          }}
                          className={`w-full px-2.5 py-2 rounded-xl text-left flex items-start justify-between gap-2 transition-colors cursor-pointer ${
                            isSelected
                              ? isFull
                                ? 'bg-[var(--honey)]/15 text-[var(--honey)]'
                                : 'bg-[var(--blush)] text-[var(--signal)]'
                              : 'text-[var(--ink)] hover:bg-[var(--paper-sunk)]'
                          }`}
                        >
                          <div className="min-w-0">
                            <div className="text-xs font-semibold leading-tight flex items-center gap-1.5">
                              {isFull ? (
                                <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
                              ) : (
                                <Shield className="w-3.5 h-3.5 shrink-0" />
                              )}
                              <span>{opt.label}</span>
                            </div>
                            <div
                              className={`text-[11px] leading-snug mt-0.5 ${
                                isSelected ? 'opacity-85' : 'text-[var(--ink-faint)]'
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
                </div>
              )}
            </div>

            {sending ? (
              <button
                type="button"
                onClick={onStopGeneration}
                disabled={stopping}
                aria-label="Stop generation"
                title={stopping ? 'Stopping...' : 'Stop running agent'}
                className="h-9 w-9 inline-flex items-center justify-center rounded-full bg-[var(--flare)] hover:opacity-90 disabled:opacity-60 text-white transition-opacity cursor-pointer shrink-0"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!canSend}
                aria-label="Send message"
                title="Send message"
                className="h-9 w-9 inline-flex items-center justify-center rounded-full bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-35 text-white transition-all cursor-pointer shrink-0"
              >
                <ArrowUp className="w-4 h-4 stroke-[2.25]" />
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}
