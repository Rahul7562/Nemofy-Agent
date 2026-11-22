import {
  Activity,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileMinus2,
  FilePenLine,
  FilePlus2,
  FolderMinus,
  FolderPlus,
  Loader2,
  Maximize2,
  Minimize2,
  ShieldAlert,
  Wrench,
  XCircle,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import {
  type ActivityUserPrefs,
  dedupeFileChanges,
  extractLatestThoughtLine,
  formatElapsedSeconds,
  formatPhaseLabel,
  summarizeActivity,
} from '../lib/activityUtils.ts';
import { resolveScrollPinState } from '../lib/scrollUtils.ts';
import type { FileChangeItem, LiveAgentActivity, ToolEventItem } from '../types/agent.ts';

interface LiveActivityPanelProps {
  mode: 'live';
  activity: LiveAgentActivity;
  prefs: ActivityUserPrefs;
  onUpdatePrefs: (patch: Partial<ActivityUserPrefs>) => void;
  onOpenWorkspaceFile?: (path: string) => void;
}

interface CompletedActivityPanelProps {
  mode: 'completed';
  thinking?: string | null;
  toolEvents?: ToolEventItem[];
  fileChanges?: FileChangeItem[];
  prefs: ActivityUserPrefs;
  onUpdatePrefs: (patch: Partial<ActivityUserPrefs>) => void;
  onOpenWorkspaceFile?: (path: string) => void;
}

type AgentActivityPanelProps = LiveActivityPanelProps | CompletedActivityPanelProps;

function FileChangeRow({
  item,
  onOpenWorkspaceFile,
}: {
  item: FileChangeItem;
  onOpenWorkspaceFile?: (path: string) => void;
}) {
  const isDir = item.kind === 'dir';
  let Icon = FilePlus2;
  let colorClass = 'text-[var(--jade)]';
  let badgeText = isDir ? '+dir' : '+file';

  if (item.action === 'created') {
    Icon = isDir ? FolderPlus : FilePlus2;
    colorClass = 'text-[var(--jade)]';
    badgeText = isDir ? '+folder' : '+file';
  } else if (item.action === 'modified') {
    Icon = FilePenLine;
    colorClass = 'text-[var(--honey)]';
    badgeText = isDir ? '~folder' : '~edited';
  } else {
    Icon = isDir ? FolderMinus : FileMinus2;
    colorClass = 'text-[var(--flare)]';
    badgeText = isDir ? '-folder' : '-deleted';
  }

  const canOpen = Boolean(onOpenWorkspaceFile && !isDir && item.action !== 'deleted');

  return (
    <li className="flex items-center justify-between gap-2 py-1.5 text-xs">
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <Icon className={`w-3.5 h-3.5 shrink-0 ${colorClass}`} />
        <span className={`font-mono text-[10px] font-semibold uppercase shrink-0 ${colorClass}`}>
          {badgeText}
        </span>
        {canOpen ? (
          <button
            type="button"
            onClick={() => onOpenWorkspaceFile?.(item.path)}
            title={`Open /workspace/${item.path} in Workspace drawer`}
            className="group inline-flex items-center gap-1.5 min-w-0 font-mono text-[11px] text-[var(--ink)] hover:text-[var(--signal)] transition-colors cursor-pointer text-left"
          >
            <span className="truncate">/workspace/{item.path}</span>
            <ExternalLink className="w-3 h-3 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
          </button>
        ) : (
          <span
            className={`font-mono text-[11px] truncate ${
              item.action === 'deleted'
                ? 'line-through text-[var(--ink-faint)]'
                : 'text-[var(--ink)]'
            }`}
            title={`/workspace/${item.path}`}
          >
            /workspace/{item.path}
          </span>
        )}
      </div>
      {item.tool && (
        <span className="font-mono text-[10px] text-[var(--ink-faint)] shrink-0">
          {item.tool}
        </span>
      )}
    </li>
  );
}

function formatToolDisplayName(toolName: string): string {
  if (toolName === 'accessory_image_subagent') return 'Image Accessory Sub-Agent';
  if (toolName === 'accessory_video_subagent') return 'Video Accessory Sub-Agent';
  if (toolName === 'accessory_audio_subagent') return 'Audio Accessory Sub-Agent';
  if (toolName === 'auto_compact_context') return 'Auto-Compact Context (80%)';
  return toolName;
}

function ToolEventRow({ item }: { item: ToolEventItem }) {
  const isRunning = item.status === 'running';
  const isError = item.status === 'error';
  const isRejected = item.status === 'rejected';

  return (
    <li className="py-1.5 border-b border-[var(--rule)]/60 last:border-b-0 text-xs">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {isRunning ? (
            <Loader2 className="w-3.5 h-3.5 text-[var(--signal)] animate-spin shrink-0" />
          ) : isError ? (
            <XCircle className="w-3.5 h-3.5 text-[var(--flare)] shrink-0" />
          ) : isRejected ? (
            <ShieldAlert className="w-3.5 h-3.5 text-[var(--honey)] shrink-0" />
          ) : (
            <CheckCircle2 className="w-3.5 h-3.5 text-[var(--jade)] shrink-0" />
          )}
          <span className="font-mono text-[11px] font-semibold text-[var(--ink)] shrink-0">
            {formatToolDisplayName(item.tool)}
          </span>
          {item.args_summary && (
            <span
              className="font-mono text-[11px] text-[var(--ink-soft)] truncate"
              title={item.args_summary}
            >
              {item.args_summary}
            </span>
          )}
        </div>
        <span
          className={`font-mono text-[10px] uppercase shrink-0 ${
            isRunning
              ? 'text-[var(--signal)]'
              : isError
                ? 'text-[var(--flare)]'
                : isRejected
                  ? 'text-[var(--honey)]'
                  : 'text-[var(--jade)]'
          }`}
        >
          {item.status}
        </span>
      </div>
      {item.output_summary && !isRunning && (
        <p className="mt-1 pl-5.5 text-[11px] text-[var(--ink-faint)] font-mono truncate">
          {item.output_summary}
        </p>
      )}
    </li>
  );
}

export function AgentActivityPanel(props: AgentActivityPanelProps) {
  const isLive = props.mode === 'live';
  const [completedExpanded, setCompletedExpanded] = useState(false);
  const thinkingScrollRef = useRef<HTMLDivElement>(null);
  const isThinkingPinnedRef = useRef(true);
  const lastThinkingScrollTopRef = useRef(0);
  const liveJobId = isLive ? props.activity.jobId : null;

  const thinking = isLive ? props.activity.thinking : props.thinking;
  const activityFeed = isLive ? props.activity.activityFeed || [] : [];
  const activeTasks = isLive ? props.activity.activeTasks || [] : [];
  const toolEvents = isLive ? props.activity.toolEvents : props.toolEvents || [];
  const fileChanges = dedupeFileChanges(isLive ? props.activity.fileChanges : props.fileChanges);

  const summary = summarizeActivity({ thinking, toolEvents, fileChanges });

  useEffect(() => {
    if (isLive) {
      isThinkingPinnedRef.current = true;
      lastThinkingScrollTopRef.current = 0;
    }
  }, [isLive, liveJobId]);

  useEffect(() => {
    const el = thinkingScrollRef.current;
    if (isLive && thinking && el && isThinkingPinnedRef.current) {
      el.scrollTop = el.scrollHeight;
      lastThinkingScrollTopRef.current = el.scrollTop;
    }
  }, [isLive, thinking]);

  function handleThinkingScroll() {
    const el = thinkingScrollRef.current;
    if (!el) return;
    isThinkingPinnedRef.current = resolveScrollPinState(
      {
        scrollTop: el.scrollTop,
        scrollHeight: el.scrollHeight,
        clientHeight: el.clientHeight,
      },
      lastThinkingScrollTopRef.current,
      isThinkingPinnedRef.current,
      24,
    );
    lastThinkingScrollTopRef.current = el.scrollTop;
  }

  if (!isLive && !summary.hasAnyActivity) {
    return null;
  }

  const expanded = isLive ? props.prefs.liveExpanded : completedExpanded;

  const toggleMain = () => {
    if (isLive) {
      props.onUpdatePrefs({ liveExpanded: !props.prefs.liveExpanded });
    } else {
      setCompletedExpanded((v) => !v);
    }
  };

  const livePhase = isLive ? (props.activity.currentPhase || '').toLowerCase() : '';
  const isActivelyThinking = isLive && (livePhase === 'thinking' || livePhase === 'plan_start');
  const latestThoughtSnippet = isActivelyThinking ? extractLatestThoughtLine(thinking) : null;

  const completedSummaryParts = summary.summaryParts.map((p) => (p === 'Thinking' ? 'Thought' : p));
  const phaseTitle = isLive
    ? latestThoughtSnippet || formatPhaseLabel(props.activity.currentPhase, props.activity.lastEvent)
    : completedSummaryParts.join(' · ') || 'Activity details';

  const liveSubtitle = isLive ? summary.summaryParts.join(' · ') : '';

  const clickableFiles = fileChanges.filter((ch) => ch.kind === 'file' && ch.action !== 'deleted');

  // Compact single-line collapsible pill for completed runs when collapsed
  if (!isLive && !expanded) {
    return (
      <div
        data-testid="completed-agent-activity"
        className="mb-3 flex flex-wrap items-center gap-2"
      >
        <button
          type="button"
          onClick={toggleMain}
          aria-expanded={false}
          aria-label="Expand agent activity"
          className="inline-flex items-center gap-1.5 rounded-full border border-[var(--rule)] bg-[var(--paper-sunk)]/80 hover:bg-[var(--paper-sunk)] hover:border-[var(--rule-strong)] px-3 py-1 text-xs font-medium text-[var(--ink-soft)] hover:text-[var(--ink)] transition-colors cursor-pointer"
        >
          <Brain className="w-3.5 h-3.5 text-[var(--signal)] shrink-0" />
          <span className="truncate max-w-[280px] sm:max-w-[420px]">{phaseTitle}</span>
          <ChevronRight className="w-3.5 h-3.5 opacity-70 shrink-0" />
        </button>

        {props.onOpenWorkspaceFile &&
          clickableFiles.slice(0, 3).map((ch) => (
            <button
              key={`${ch.action}-${ch.path}`}
              type="button"
              onClick={() => props.onOpenWorkspaceFile?.(ch.path)}
              title={`Inspect /workspace/${ch.path}`}
              className="inline-flex items-center gap-1.5 rounded-full border border-[var(--rule)] bg-[var(--paper)] hover:border-[var(--signal)]/60 hover:text-[var(--signal)] px-2.5 py-1 text-[11px] font-mono text-[var(--ink-soft)] transition-colors cursor-pointer max-w-[200px]"
            >
              {ch.action === 'created' ? (
                <FilePlus2 className="w-3 h-3 text-[var(--jade)] shrink-0" />
              ) : (
                <FilePenLine className="w-3 h-3 text-[var(--honey)] shrink-0" />
              )}
              <span className="truncate">{ch.path.split('/').pop() || ch.path}</span>
            </button>
          ))}
      </div>
    );
  }

  return (
    <div
      data-testid={isLive ? 'live-agent-activity' : 'completed-agent-activity'}
      className={
        isLive
          ? 'w-full rounded-2xl border border-[var(--signal)]/30 bg-[var(--paper-raised)] shadow-sm overflow-hidden transition-all'
          : 'mb-3.5 rounded-xl border border-[var(--rule)] bg-[var(--paper-sunk)]/65 overflow-hidden'
      }
    >
      {/* Header Bar */}
      <div
        className={`flex items-center justify-between gap-3 ${
          isLive ? 'px-4 py-2.5 bg-[var(--blush)]/45' : 'px-3.5 py-2'
        }`}
      >
        <button
          type="button"
          onClick={toggleMain}
          aria-expanded={expanded}
          aria-label={
            isLive
              ? expanded
                ? 'Minimize live activity'
                : 'Expand live activity'
              : 'Collapse agent activity'
          }
          className="flex-1 min-w-0 flex items-center gap-2 text-left cursor-pointer group"
        >
          {isLive ? (
            <span className="relative flex h-2.5 w-2.5 shrink-0">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-60 ${
                  livePhase === 'rate_limited' ? 'bg-[var(--honey)]' : 'bg-[var(--signal)]'
                }`}
              />
              <span
                className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
                  livePhase === 'rate_limited' ? 'bg-[var(--honey)]' : 'bg-[var(--signal)]'
                }`}
              />
            </span>
          ) : (
            <Brain className="w-3.5 h-3.5 text-[var(--signal)] shrink-0" />
          )}

          <div className="min-w-0 flex-1 flex flex-wrap items-center gap-x-2 gap-y-0.5">
            <span
              className={`truncate ${
                isLive
                  ? 'text-xs font-semibold text-[var(--ink)]'
                  : 'text-xs font-medium text-[var(--ink)]'
              }`}
            >
              {phaseTitle}
            </span>

            {isLive && liveSubtitle && (
              <span className="text-[11px] font-mono text-[var(--signal)] truncate">
                · {liveSubtitle}
              </span>
            )}
          </div>
        </button>

        <div className="flex items-center gap-2 shrink-0">
          {isLive && (
            <span className="font-mono text-[11px] tabular-nums text-[var(--ink-faint)]">
              {formatElapsedSeconds(props.activity.elapsedSeconds)}
            </span>
          )}
          <button
            type="button"
            onClick={toggleMain}
            aria-label={expanded ? 'Minimize activity panel' : 'Expand activity panel'}
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
              isLive
                ? 'border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] text-[var(--ink-soft)] hover:text-[var(--ink)]'
                : 'text-[var(--ink-soft)] hover:text-[var(--ink)]'
            }`}
          >
            {isLive ? (
              expanded ? (
                <>
                  <Minimize2 className="w-3 h-3" />
                  <span>Minimize</span>
                </>
              ) : (
                <>
                  <Maximize2 className="w-3 h-3" />
                  <span>Expand</span>
                </>
              )
            ) : (
              <>
                <span>Hide</span>
                <ChevronDown className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Expandable Body */}
      {expanded && (
        <div className="divide-y divide-[var(--rule)] border-t border-[var(--rule)]">
          {/* 1. Thinking & Live Activity Section */}
          {(isLive || summary.hasThinking) && (
            <div>
              <button
                type="button"
                onClick={() =>
                  props.onUpdatePrefs({ showThinkingSection: !props.prefs.showThinkingSection })
                }
                aria-expanded={props.prefs.showThinkingSection}
                className="w-full px-3.5 py-2 flex items-center justify-between gap-2 text-[11px] font-medium text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)]/60 transition-colors cursor-pointer"
              >
                <span className="inline-flex items-center gap-1.5">
                  {thinking ? (
                    <Brain className="w-3.5 h-3.5 text-[var(--signal)]" />
                  ) : (
                    <Activity className="w-3.5 h-3.5 text-[var(--signal)]" />
                  )}
                  <span>{thinking ? 'Thinking' : 'Live activity'}</span>
                  {isLive && (isActivelyThinking || livePhase === 'responding' || livePhase === 'planning_tool') && (
                    <Loader2 className="w-3 h-3 animate-spin text-[var(--signal)]" />
                  )}
                </span>
                {props.prefs.showThinkingSection ? (
                  <ChevronDown className="w-3.5 h-3.5" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5" />
                )}
              </button>
              {props.prefs.showThinkingSection && (
                <div className="px-3.5 pb-2.5 pt-0.5 space-y-2">
                  {thinking && (
                    <div
                      ref={thinkingScrollRef}
                      onScroll={handleThinkingScroll}
                      onWheel={(e) => {
                        if (e.deltaY < 0) {
                          isThinkingPinnedRef.current = false;
                        }
                      }}
                      data-testid={isLive ? 'live-thinking-text' : 'completed-thinking-text'}
                      className="text-xs text-[var(--ink-soft)] whitespace-pre-wrap leading-relaxed max-h-52 overflow-y-auto scrollbar-none font-mono"
                    >
                      {thinking}
                      {isActivelyThinking && (
                        <span
                          aria-hidden="true"
                          className="inline-block w-1.5 h-3.5 ml-0.5 align-middle bg-[var(--signal)] animate-pulse"
                        />
                      )}
                    </div>
                  )}

                  {isLive && !thinking && (
                    <ul data-testid="live-activity-feed" className="space-y-1.5">
                      {activityFeed.length > 0 ? (
                        activityFeed.slice(-6).map((item, idx, arr) => {
                          const isLatest = idx === arr.length - 1;
                          return (
                            <li
                              key={item.id}
                              className={`flex items-start gap-2 text-xs ${
                                isLatest ? 'text-[var(--ink)] font-medium' : 'text-[var(--ink-faint)]'
                              }`}
                            >
                              {isLatest ? (
                                <Loader2 className="w-3.5 h-3.5 mt-0.5 text-[var(--signal)] animate-spin shrink-0" />
                              ) : (
                                <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 text-[var(--jade)]/80 shrink-0" />
                              )}
                              <span className="leading-snug">{item.text}</span>
                            </li>
                          );
                        })
                      ) : (
                        <li className="flex items-center gap-2 text-xs text-[var(--ink)]">
                          <Loader2 className="w-3.5 h-3.5 text-[var(--signal)] animate-spin shrink-0" />
                          <span>{formatPhaseLabel(props.activity.currentPhase, props.activity.lastEvent)}</span>
                        </li>
                      )}
                    </ul>
                  )}

                  {isLive && thinking && !isActivelyThinking && props.activity.lastEvent && (
                    <div className="pt-1.5 border-t border-[var(--rule)]/60 flex items-center gap-2 text-[11px] text-[var(--ink-soft)]">
                      <Loader2 className="w-3 h-3 text-[var(--signal)] animate-spin shrink-0" />
                      <span className="truncate">
                        {formatPhaseLabel(props.activity.currentPhase, props.activity.lastEvent)}
                      </span>
                    </div>
                  )}

                  {isLive && activeTasks.length > 0 && (
                    <div
                      data-testid="live-active-tasks"
                      className="pt-1.5 border-t border-[var(--rule)]/60 space-y-1"
                    >
                      {activeTasks.slice(0, 3).map((task) => (
                        <div
                          key={task.id}
                          className="flex items-center justify-between gap-2 text-[11px] text-[var(--ink-soft)]"
                        >
                          <span className="inline-flex items-center gap-1.5 min-w-0">
                            <Loader2 className="w-3 h-3 text-[var(--jade)] animate-spin shrink-0" />
                            <span className="font-mono truncate" title={task.label}>
                              {task.label}
                            </span>
                          </span>
                          <span className="font-mono text-[10px] tabular-nums text-[var(--ink-faint)] shrink-0">
                            {formatElapsedSeconds(task.elapsed_seconds)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* 2. Tools Used Section */}
          {toolEvents.length > 0 && (
            <div>
              <button
                type="button"
                onClick={() =>
                  props.onUpdatePrefs({ showToolsSection: !props.prefs.showToolsSection })
                }
                aria-expanded={props.prefs.showToolsSection}
                className="w-full px-3.5 py-2 flex items-center justify-between gap-2 text-[11px] font-medium text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)]/60 transition-colors cursor-pointer"
              >
                <span className="inline-flex items-center gap-1.5">
                  <Wrench className="w-3.5 h-3.5 text-[var(--signal)]" />
                  <span>Tools used ({toolEvents.length})</span>
                </span>
                {props.prefs.showToolsSection ? (
                  <ChevronDown className="w-3.5 h-3.5" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5" />
                )}
              </button>
              {props.prefs.showToolsSection && (
                <ul className="px-3.5 pb-2 max-h-52 overflow-y-auto scrollbar-none">
                  {toolEvents.map((ev) => (
                    <ToolEventRow key={ev.id} item={ev} />
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* 3. Workspace File & Directory Changes Section */}
          {fileChanges.length > 0 && (
            <div>
              <button
                type="button"
                onClick={() =>
                  props.onUpdatePrefs({ showFilesSection: !props.prefs.showFilesSection })
                }
                aria-expanded={props.prefs.showFilesSection}
                className="w-full px-3.5 py-2 flex items-center justify-between gap-2 text-[11px] font-medium text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)]/60 transition-colors cursor-pointer"
              >
                <span className="inline-flex items-center gap-1.5">
                  <FolderPlus className="w-3.5 h-3.5 text-[var(--jade)]" />
                  <span>Files &amp; folders ({fileChanges.length})</span>
                </span>
                {props.prefs.showFilesSection ? (
                  <ChevronDown className="w-3.5 h-3.5" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5" />
                )}
              </button>
              {props.prefs.showFilesSection && (
                <ul className="px-3.5 pb-2 max-h-52 overflow-y-auto scrollbar-none divide-y divide-[var(--rule)]/50">
                  {fileChanges.map((ch, idx) => (
                    <FileChangeRow
                      key={`${ch.kind}-${ch.action}-${ch.path}-${idx}`}
                      item={ch}
                      onOpenWorkspaceFile={props.onOpenWorkspaceFile}
                    />
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
