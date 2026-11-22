import {
  AudioLines,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Layers,
  Video as VideoIcon,
} from 'lucide-react';
import { useState } from 'react';
import type { ActivityUserPrefs } from '../lib/activityUtils.ts';
import { formatFileSize } from '../lib/llmPresets.ts';
import { formatThinkingBadge } from '../lib/thinkingModes.ts';
import type { ChatAttachment, ChatMessage } from '../types/agent.ts';
import { AgentActivityPanel } from './AgentActivityPanel.tsx';
import { MarkdownMessage } from './MarkdownMessage.tsx';
import { NemofyLogo } from './NemofyLogo.tsx';

interface ChatMessageItemProps {
  message: ChatMessage;
  activityPrefs: ActivityUserPrefs;
  onUpdateActivityPrefs: (patch: Partial<ActivityUserPrefs>) => void;
  onPreviewImage: (att: ChatAttachment) => void;
  onOpenWorkspaceFile?: (path: string) => void;
}

function MessageAttachments({
  attachments,
  onPreviewImage,
}: {
  attachments: ChatAttachment[];
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
              className="group relative overflow-hidden rounded-xl border border-[var(--rule)] bg-[var(--paper)] cursor-pointer text-left hover:border-[var(--signal)] transition-colors"
            >
              <img
                src={att.url}
                alt={att.name}
                className="max-h-52 max-w-[260px] object-cover block"
              />
              <div className="px-2.5 py-1 text-[10px] font-mono truncate max-w-[260px] bg-black/55 text-white">
                {att.name}
              </div>
            </button>
          );
        }
        if (att.kind === 'video') {
          return (
            <div
              key={att.id}
              className="w-full max-w-sm rounded-xl overflow-hidden border border-[var(--rule)] bg-[var(--paper)]"
            >
              <video
                src={att.url}
                controls
                preload="metadata"
                className="w-full max-h-56 bg-black block"
              />
              <div className="px-2.5 py-1.5 flex items-center justify-between gap-2 text-[11px] text-[var(--ink-soft)]">
                <span className="inline-flex items-center gap-1.5 truncate font-mono">
                  <VideoIcon className="w-3.5 h-3.5 shrink-0" />
                  <span className="truncate">{att.name}</span>
                </span>
                <span className="text-[10px] tabular-nums opacity-75 shrink-0">
                  {formatFileSize(att.size)}
                </span>
              </div>
            </div>
          );
        }
        return (
          <div
            key={att.id}
            className="w-full max-w-sm rounded-xl p-2.5 border border-[var(--rule)] bg-[var(--paper)] text-[var(--ink)] space-y-1.5"
          >
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="inline-flex items-center gap-1.5 truncate font-mono">
                <AudioLines className="w-3.5 h-3.5 shrink-0 text-[var(--signal)]" />
                <span className="truncate">{att.name}</span>
              </span>
              <span className="text-[10px] tabular-nums text-[var(--ink-faint)] shrink-0">
                {formatFileSize(att.size)}
              </span>
            </div>
            <audio src={att.url} controls preload="metadata" className="w-full h-8" />
          </div>
        );
      })}
    </div>
  );
}

function MessageMetadata({ msg }: { msg: ChatMessage }) {
  const [planOpen, setPlanOpen] = useState(false);
  const [reportsOpen, setReportsOpen] = useState(false);
  const hasPlan = Boolean(msg.plan && msg.plan.length > 0);
  const hasReports = Boolean(msg.subagentReports && msg.subagentReports.length > 0);
  const hasSkills = Boolean(msg.activeSkills && msg.activeSkills.length > 0);

  if (!hasPlan && !hasReports && !hasSkills) return null;

  return (
    <div className="mt-3 pt-2.5 border-t border-[var(--rule)]/70 space-y-2 text-xs">
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
                  className="p-3 rounded-xl bg-[var(--paper-sunk)] border border-[var(--rule)] space-y-1.5"
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
                className="p-2.5 rounded-lg bg-[var(--paper-sunk)] border border-[var(--rule)] font-mono text-[11px] text-[var(--ink-soft)] overflow-x-auto"
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

export function ChatMessageItem({
  message,
  activityPrefs,
  onUpdateActivityPrefs,
  onPreviewImage,
  onOpenWorkspaceFile,
}: ChatMessageItemProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  async function handleCopyResponse() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore clipboard errors in headless environments
    }
  }

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[84%] sm:max-w-[78%] rounded-2xl bg-[var(--paper-sunk)] border border-[var(--rule)] text-[var(--ink)] px-4 py-3">
          {message.attachments && message.attachments.length > 0 && (
            <MessageAttachments
              attachments={message.attachments}
              onPreviewImage={onPreviewImage}
            />
          )}
          <MarkdownMessage content={message.content} isUser />
        </div>
      </div>
    );
  }

  const modelLabel = message.activeModels?.primary?.model || null;
  const thinkingLabel = formatThinkingBadge(message.activeModels?.primary?.reasoning_effort);

  return (
    <article className="group flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-4">
      <div className="flex items-center gap-2 sm:block shrink-0">
        <div className="w-7 h-7 sm:mt-0.5 rounded-lg bg-[var(--paper-sunk)] border border-[var(--rule)] flex items-center justify-center shrink-0">
          <NemofyLogo size={17} />
        </div>
        <span className="sm:hidden font-display text-xs font-semibold text-[var(--ink)]">Nemo</span>
      </div>

      <div className="min-w-0 flex-1 w-full">
        <AgentActivityPanel
          mode="completed"
          thinking={message.thinking}
          toolEvents={message.toolEvents}
          fileChanges={message.fileChanges}
          prefs={activityPrefs}
          onUpdatePrefs={onUpdateActivityPrefs}
          onOpenWorkspaceFile={onOpenWorkspaceFile}
        />

        {message.attachments && message.attachments.length > 0 && (
          <MessageAttachments
            attachments={message.attachments}
            onPreviewImage={onPreviewImage}
          />
        )}

        <MarkdownMessage content={message.content} isUser={false} />

        <MessageMetadata msg={message} />

        {/* Subtle hover-revealed action bar */}
        <div className="mt-2.5 flex flex-wrap items-center gap-2.5 text-[11px] text-[var(--ink-faint)] opacity-85 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity">
          <button
            type="button"
            onClick={() => void handleCopyResponse()}
            aria-label="Copy response"
            className="inline-flex items-center gap-1 px-2 py-1 -ml-2 rounded-md text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer"
          >
            {copied ? (
              <Check className="w-3.5 h-3.5 text-[var(--jade)]" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>

          {modelLabel && (
            <span className="font-mono text-[11px] text-[var(--ink-faint)] truncate max-w-[260px]">
              {modelLabel}
              {thinkingLabel ? ` · ${thinkingLabel}` : ''}
            </span>
          )}
        </div>
      </div>
    </article>
  );
}
