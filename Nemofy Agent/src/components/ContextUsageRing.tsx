import { Layers, RefreshCw, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import {
  AUTO_COMPACT_THRESHOLD_PERCENT,
  CONTEXT_CATEGORY_META,
  MIN_CONTEXT_WINDOW,
  formatTokenCount,
} from '../lib/contextUtils.ts';
import type { ContextClassificationLevel, ContextUsageSnapshot } from '../types/agent.ts';

interface ContextUsageRingProps {
  usage: ContextUsageSnapshot;
  modelConfigured: boolean;
  onOpenLLMSettings?: () => void;
  onSetManualContextWindow?: (manualWindow: number | null) => Promise<void> | void;
  onRefreshMetadata?: () => Promise<void> | void;
}

const CATEGORY_COLOR_BY_KEY = Object.fromEntries(
  CONTEXT_CATEGORY_META.map((c) => [c.key, c.color]),
) as Record<string, string>;

function getLevelColorVar(level: ContextClassificationLevel): string {
  switch (level) {
    case 'critical':
      return 'var(--flare)';
    case 'high':
      return 'var(--honey)';
    case 'moderate':
      return 'var(--signal)';
    case 'unknown':
      return 'var(--ink-faint)';
    default:
      return 'var(--jade)';
  }
}

function getLevelTextClass(level: ContextClassificationLevel): string {
  switch (level) {
    case 'critical':
      return 'text-[var(--flare)]';
    case 'high':
      return 'text-[var(--honey)]';
    case 'moderate':
      return 'text-[var(--signal)]';
    case 'unknown':
      return 'text-[var(--ink-faint)]';
    default:
      return 'text-[var(--jade)]';
  }
}

function formatWindowSource(source: string): string {
  switch (source) {
    case 'manual':
      return 'Manual override';
    case 'custom_provider_config':
      return 'Custom provider';
    case 'provider_catalog':
      return 'Auto · Provider API';
    case 'provider_registry':
      return 'Auto · Provider catalog';
    case 'reference_catalog':
      return 'Auto · Reference catalog';
    case 'model_registry':
      return 'Auto · Model registry';
    case 'unknown':
      return 'Unknown';
    default:
      return 'Auto-detected';
  }
}

export function ContextUsageRing({
  usage,
  modelConfigured,
  onOpenLLMSettings,
  onSetManualContextWindow,
  onRefreshMetadata,
}: ContextUsageRingProps) {
  const [popupOpen, setPopupOpen] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [manualDraft, setManualDraft] = useState('');
  const [manualError, setManualError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<'save' | 'refresh' | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setManualDraft(
      typeof usage.manual_context_window === 'number' && usage.manual_context_window >= MIN_CONTEXT_WINDOW
        ? String(usage.manual_context_window)
        : '',
    );
    setManualError(null);
  }, [usage.manual_context_window, usage.provider, usage.model]);

  useEffect(() => {
    if (!popupOpen) return;
    function handleOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setPopupOpen(false);
      }
    }
    function handleKey(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setPopupOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('keydown', handleKey);
    };
  }, [popupOpen]);

  const hasKnownWindow =
    typeof usage.context_window === 'number' &&
    Number.isFinite(usage.context_window) &&
    usage.context_window > 0;
  const pct =
    hasKnownWindow && typeof usage.usage_percent === 'number' && Number.isFinite(usage.usage_percent)
      ? Math.max(0, Math.min(100, usage.usage_percent))
      : null;
  const roundedPctLabel =
    pct === null ? '—' : pct > 0 && pct < 1 ? '<1%' : `${Math.round(pct)}%`;
  const strokeColor = getLevelColorVar(usage.classification.level);
  const levelTextClass = getLevelTextClass(usage.classification.level);

  const isManual =
    usage.is_manual_override === true || usage.context_window_source === 'manual';
  const sourceLabel = formatWindowSource(usage.context_window_source);
  const usableInput =
    typeof usage.usable_input_window === 'number' && usage.usable_input_window > 0
      ? usage.usable_input_window
      : null;
  const remainingInput =
    typeof usage.remaining_input_budget === 'number' && usage.remaining_input_budget >= 0
      ? usage.remaining_input_budget
      : usage.free_tokens;
  const reservedOut =
    typeof usage.reserved_output_tokens === 'number' && usage.reserved_output_tokens > 0
      ? usage.reserved_output_tokens
      : 0;
  const safetyBuffer =
    typeof usage.safety_buffer_tokens === 'number' && usage.safety_buffer_tokens > 0
      ? usage.safety_buffer_tokens
      : 0;

  // SVG Ring geometry (36x36 viewBox, centered at 18,18, radius 13.5)
  const radius = 13.5;
  const circumference = 2 * Math.PI * radius;
  const visibleFillPct = pct !== null && pct > 0 ? Math.max(pct, 1.8) : 0;
  const dashOffset = circumference * (1 - visibleFillPct / 100);

  // 80% threshold marker tick coordinates (clockwise from 12 o'clock: -90deg + 0.8 * 360deg = 198deg)
  const thresholdAngleRad = ((usage.auto_compact_threshold_percent / 100) * 360 - 90) * (Math.PI / 180);
  const tickInnerR = radius - 2.6;
  const tickOuterR = radius + 2.6;
  const tickX1 = 18 + tickInnerR * Math.cos(thresholdAngleRad);
  const tickY1 = 18 + tickInnerR * Math.sin(thresholdAngleRad);
  const tickX2 = 18 + tickOuterR * Math.cos(thresholdAngleRad);
  const tickY2 = 18 + tickOuterR * Math.sin(thresholdAngleRad);

  const showHoverTooltip = hovered && !popupOpen;

  const ariaSummary = hasKnownWindow && pct !== null
    ? `Context usage ${pct.toFixed(1)}% (${usage.classification.label}), ${formatTokenCount(usage.used_tokens)} of ${formatTokenCount(usage.context_window!)} tokens (${sourceLabel}). Click for breakdown.`
    : `Context window unknown (${formatTokenCount(usage.used_tokens)} tokens used). Click to view details or set a manual override.`;

  async function handleApplyManualOverride() {
    if (!onSetManualContextWindow) return;
    const raw = manualDraft.trim();
    if (!raw) {
      setManualError(null);
      setBusyAction('save');
      try {
        await onSetManualContextWindow(null);
      } finally {
        setBusyAction(null);
      }
      return;
    }
    const parsed = Number(raw.replace(/[_,]/g, ''));
    if (!Number.isFinite(parsed) || parsed < MIN_CONTEXT_WINDOW || parsed > 20_000_000) {
      setManualError(`Enter a token limit between ${MIN_CONTEXT_WINDOW} and 20,000,000.`);
      return;
    }
    setManualError(null);
    setBusyAction('save');
    try {
      await onSetManualContextWindow(Math.round(parsed));
    } finally {
      setBusyAction(null);
    }
  }

  async function handleClearManualOverride() {
    if (!onSetManualContextWindow) return;
    setManualDraft('');
    setManualError(null);
    setBusyAction('save');
    try {
      await onSetManualContextWindow(null);
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRefresh() {
    if (!onRefreshMetadata) return;
    setBusyAction('refresh');
    try {
      await onRefreshMetadata();
    } finally {
      setBusyAction(null);
    }
  }

  return (
    <div
      ref={containerRef}
      className="relative shrink-0"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        data-testid="context-usage-ring"
        aria-haspopup="dialog"
        aria-expanded={popupOpen}
        aria-label={ariaSummary}
        onClick={() => setPopupOpen((v) => !v)}
        onFocus={() => setHovered(true)}
        onBlur={() => setHovered(false)}
        className={`h-9 px-2 inline-flex items-center gap-1.5 rounded-lg border transition-colors cursor-pointer ${
          popupOpen
            ? 'border-[var(--signal)] bg-[var(--blush)]/60 text-[var(--ink)]'
            : 'border-[var(--rule)] bg-[var(--paper-sunk)] hover:border-[var(--rule-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)]'
        }`}
      >
        <svg
          width="24"
          height="24"
          viewBox="0 0 36 36"
          aria-hidden="true"
          className="shrink-0"
        >
          {/* Base track */}
          <circle
            cx="18"
            cy="18"
            r={radius}
            fill="none"
            stroke="var(--rule-strong)"
            strokeOpacity="0.45"
            strokeWidth="3.2"
          />
          {/* Progress arc */}
          {hasKnownWindow && (
            <circle
              cx="18"
              cy="18"
              r={radius}
              fill="none"
              stroke={strokeColor}
              strokeWidth="3.4"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={dashOffset}
              transform="rotate(-90 18 18)"
              className="transition-[stroke-dashoffset,stroke] duration-300 ease-out"
            />
          )}
          {/* 80% Auto-compact threshold tick */}
          {hasKnownWindow && (
            <line
              x1={tickX1}
              y1={tickY1}
              x2={tickX2}
              y2={tickY2}
              stroke="var(--ink-faint)"
              strokeWidth="1.4"
              strokeLinecap="round"
            />
          )}
        </svg>
        <span className="font-mono text-[11px] font-semibold tabular-nums text-[var(--ink)] min-w-[26px] text-right">
          {roundedPctLabel}
        </span>
      </button>

      {/* Hover Tooltip: Current Context Usage & Classification */}
      {showHoverTooltip && (
        <div
          role="tooltip"
          data-testid="context-usage-hover-tooltip"
          className="pointer-events-none absolute right-0 top-full mt-2 z-40 w-68 rounded-xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-lg p-3 space-y-1.5 text-left"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[var(--ink-faint)]">
              Context Usage
            </span>
            <span className={`text-[11px] font-semibold ${levelTextClass}`}>
              {usage.classification.label}
            </span>
          </div>

          <div className="flex items-baseline justify-between gap-2">
            <span className="font-mono text-xs font-semibold tabular-nums text-[var(--ink)]">
              {hasKnownWindow
                ? `${formatTokenCount(usage.used_tokens)} / ${formatTokenCount(usage.context_window!)} tokens`
                : `${formatTokenCount(usage.used_tokens)} tokens · Window unknown`}
            </span>
            <span className={`font-mono text-xs font-semibold tabular-nums ${levelTextClass}`}>
              {pct !== null ? `${pct.toFixed(1)}%` : '—'}
            </span>
          </div>

          {hasKnownWindow && (
            <div className="font-mono text-[10px] tabular-nums text-[var(--ink-faint)]">
              {usableInput ? `${formatTokenCount(usableInput)} usable in` : ''}
              {reservedOut > 0 ? ` · ${formatTokenCount(reservedOut)} reserved out` : ''}
              {typeof usage.max_output_tokens === 'number' && usage.max_output_tokens > 0
                ? ` · ${formatTokenCount(usage.max_output_tokens)} max out`
                : ''}
            </div>
          )}

          <p className="text-[11px] text-[var(--ink-soft)] leading-snug">
            {usage.classification.description}
          </p>

          <div className="pt-1 border-t border-[var(--rule)] flex items-center justify-between gap-2 text-[10px] text-[var(--ink-faint)]">
            <span className="truncate font-mono">
              {modelConfigured && usage.model ? `${usage.provider} · ${usage.model}` : 'No primary model'}
            </span>
            <span className="shrink-0">{sourceLabel}</span>
          </div>
        </div>
      )}

      {/* Click Popup: Full Context & Category Breakdown */}
      {popupOpen && (
        <div
          role="dialog"
          aria-label="Context usage breakdown"
          data-testid="context-usage-popup"
          className="absolute right-0 top-full mt-2 z-40 w-[364px] max-w-[calc(100vw-2rem)] rounded-xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-xl overflow-hidden text-left"
        >
          {/* Popup Header */}
          <div className="px-4 py-3 border-b border-[var(--rule)] bg-[var(--paper-sunk)] flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-display text-sm font-semibold text-[var(--ink)]">
                  Context Window
                </span>
                <span className={`text-xs font-semibold ${levelTextClass}`}>
                  · {usage.classification.label}
                </span>
              </div>
              <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-[var(--ink-faint)]">
                {modelConfigured && usage.model ? (
                  <span className="font-mono truncate" title={`${usage.provider}: ${usage.model}`}>
                    {usage.model}
                  </span>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      setPopupOpen(false);
                      onOpenLLMSettings?.();
                    }}
                    className="text-[var(--signal)] hover:underline cursor-pointer"
                  >
                    Select a primary model
                  </button>
                )}
                <span>·</span>
                <span
                  data-testid="context-window-source"
                  className={`shrink-0 font-medium ${
                    isManual
                      ? 'text-[var(--signal)]'
                      : hasKnownWindow
                        ? 'text-[var(--jade)]'
                        : 'text-[var(--honey)]'
                  }`}
                >
                  {sourceLabel}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1 shrink-0">
              {onRefreshMetadata && modelConfigured && (
                <button
                  type="button"
                  data-testid="context-refresh-metadata"
                  disabled={busyAction !== null}
                  onClick={() => void handleRefresh()}
                  title="Refresh model context-window metadata from provider"
                  aria-label="Refresh model metadata"
                  className="p-1 rounded-md text-[var(--ink-faint)] hover:text-[var(--signal)] hover:bg-[var(--paper)] disabled:opacity-50 transition-colors cursor-pointer"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${busyAction === 'refresh' ? 'animate-spin' : ''}`} />
                </button>
              )}
              <button
                type="button"
                onClick={() => setPopupOpen(false)}
                aria-label="Close context breakdown"
                className="p-1 rounded-md text-[var(--ink-faint)] hover:text-[var(--ink)] hover:bg-[var(--paper)] transition-colors cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Usage Summary + Segmented Bar */}
          <div className="px-4 pt-3.5 pb-3 border-b border-[var(--rule)] space-y-2.5">
            <div className="flex items-baseline justify-between gap-2">
              <div className="font-mono text-sm font-semibold tabular-nums text-[var(--ink)]">
                {formatTokenCount(usage.used_tokens)}{' '}
                <span className="text-xs font-normal text-[var(--ink-faint)]">
                  {hasKnownWindow
                    ? `/ ${formatTokenCount(usage.context_window!)} tokens`
                    : 'tokens used'}
                </span>
              </div>
              <span className={`font-mono text-sm font-semibold tabular-nums ${levelTextClass}`}>
                {pct !== null ? `${pct.toFixed(1)}%` : 'Unknown'}
              </span>
            </div>

            {/* Model-Specific Total / Usable Input / Reserved & Max Output Specs */}
            <div
              data-testid="context-model-limits"
              className="grid grid-cols-3 gap-2 pt-0.5 text-[10px] font-mono tabular-nums"
            >
              <div className="rounded-md bg-[var(--paper-sunk)] border border-[var(--rule)] px-2 py-1.5">
                <div className="text-[var(--ink-faint)] uppercase tracking-wide text-[9px] font-sans">
                  Total Window
                </div>
                <div className="text-[var(--ink)] font-semibold mt-0.5" data-testid="context-total-window">
                  {hasKnownWindow ? formatTokenCount(usage.context_window!) : 'Unknown'}
                </div>
              </div>
              <div className="rounded-md bg-[var(--paper-sunk)] border border-[var(--rule)] px-2 py-1.5">
                <div className="text-[var(--ink-faint)] uppercase tracking-wide text-[9px] font-sans">
                  Usable Input
                </div>
                <div className="text-[var(--ink)] font-semibold mt-0.5" data-testid="context-usable-input">
                  {usableInput !== null ? formatTokenCount(usableInput) : 'Unknown'}
                </div>
              </div>
              <div className="rounded-md bg-[var(--paper-sunk)] border border-[var(--rule)] px-2 py-1.5">
                <div className="text-[var(--ink-faint)] uppercase tracking-wide text-[9px] font-sans">
                  Max Output
                </div>
                <div className="text-[var(--ink)] font-semibold mt-0.5">
                  {typeof usage.max_output_tokens === 'number' && usage.max_output_tokens > 0
                    ? formatTokenCount(usage.max_output_tokens)
                    : 'Unknown'}
                </div>
              </div>
            </div>

            {hasKnownWindow && (reservedOut > 0 || safetyBuffer > 0) && (
              <div className="flex items-center justify-between gap-2 text-[10px] font-mono tabular-nums text-[var(--ink-faint)]">
                <span>
                  Reserved: {formatTokenCount(reservedOut)} out + {formatTokenCount(safetyBuffer)} buffer
                </span>
                <span>
                  {remainingInput !== null ? `${formatTokenCount(remainingInput)} in left` : ''}
                </span>
              </div>
            )}

            {hasKnownWindow ? (
              <div className="relative pt-0.5 pb-3.5">
                <div className="h-2.5 w-full rounded-sm bg-[var(--paper-sunk)] border border-[var(--rule)] overflow-hidden flex">
                  {usage.categories.map((cat) => {
                    if (cat.tokens <= 0) return null;
                    const widthPct = Math.max(
                      0.5,
                      (cat.tokens / Math.max(1, usage.context_window!)) * 100,
                    );
                    return (
                      <div
                        key={cat.key}
                        title={`${cat.label}: ${formatTokenCount(cat.tokens)} tokens (${cat.percent.toFixed(1)}%)`}
                        style={{
                          width: `${Math.min(100, widthPct)}%`,
                          backgroundColor: CATEGORY_COLOR_BY_KEY[cat.key] || 'var(--signal)',
                        }}
                        className="h-full transition-[width] duration-300"
                      />
                    );
                  })}
                </div>
                {/* 80% Auto-Compact Threshold Line */}
                <div
                  aria-hidden="true"
                  style={{ left: `${AUTO_COMPACT_THRESHOLD_PERCENT}%` }}
                  className="absolute top-0 bottom-3 w-px bg-[var(--ink)]/55 pointer-events-none"
                />
                <span
                  style={{ left: `${AUTO_COMPACT_THRESHOLD_PERCENT}%` }}
                  className="absolute bottom-0 -translate-x-1/2 font-mono text-[10px] text-[var(--ink-faint)] pointer-events-none whitespace-nowrap"
                >
                  80% auto-compact
                </span>
              </div>
            ) : (
              <p className="text-[11px] text-[var(--ink-soft)] leading-snug rounded-lg border border-[var(--rule)] bg-[var(--paper-sunk)] px-2.5 py-2">
                Context window could not be verified from provider metadata. Token counts remain tracked without assuming a default limit. Set a manual override below if you know this model&apos;s limit.
              </p>
            )}
          </div>

          {/* Per-Session Manual Context Window Override */}
          {onSetManualContextWindow && (
            <div className="px-4 py-2.5 border-b border-[var(--rule)] bg-[var(--paper-sunk)]/45 space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <label
                  htmlFor="session-manual-context-window"
                  className="text-[11px] font-medium text-[var(--ink-soft)]"
                >
                  Manual Context Window Override (this session)
                </label>
                {isManual && (
                  <button
                    type="button"
                    data-testid="clear-manual-context-window"
                    disabled={busyAction !== null}
                    onClick={() => void handleClearManualOverride()}
                    className="text-[11px] font-medium text-[var(--signal)] hover:underline cursor-pointer"
                  >
                    Use Auto
                  </button>
                )}
              </div>
              <div className="flex items-center gap-1.5">
                <input
                  id="session-manual-context-window"
                  data-testid="manual-context-window-input"
                  type="text"
                  inputMode="numeric"
                  value={manualDraft}
                  onChange={(e) => {
                    setManualDraft(e.target.value);
                    setManualError(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      void handleApplyManualOverride();
                    }
                  }}
                  placeholder={hasKnownWindow ? `Auto: ${usage.context_window}` : 'e.g. 131072'}
                  className="flex-1 h-8 px-2.5 rounded-md border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs tabular-nums text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none focus:border-[var(--signal)]"
                />
                <button
                  type="button"
                  data-testid="apply-manual-context-window"
                  disabled={busyAction !== null}
                  onClick={() => void handleApplyManualOverride()}
                  className="h-8 px-2.5 rounded-md border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] hover:text-[var(--signal)] text-xs font-medium text-[var(--ink)] disabled:opacity-50 transition-colors cursor-pointer shrink-0"
                >
                  {busyAction === 'save' ? 'Saving...' : 'Apply'}
                </button>
              </div>
              {manualError && (
                <p className="text-[10px] text-[var(--flare)]">{manualError}</p>
              )}
            </div>
          )}

          {/* All Context Categories */}
          <div className="px-4 py-3 space-y-1.5">
            <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wide text-[var(--ink-faint)] pb-1">
              <span>Category</span>
              <span>{hasKnownWindow ? 'Tokens · Window' : 'Tokens · Share'}</span>
            </div>

            <ul className="space-y-1.5" data-testid="context-category-list">
              {usage.categories.map((cat) => (
                <li
                  key={cat.key}
                  title={cat.description}
                  className="flex items-center justify-between gap-2 text-xs py-0.5"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      aria-hidden="true"
                      style={{
                        backgroundColor: CATEGORY_COLOR_BY_KEY[cat.key] || 'var(--signal)',
                      }}
                      className="w-2.5 h-2.5 rounded-[2px] shrink-0"
                    />
                    <span
                      className={`truncate ${
                        cat.tokens > 0 ? 'text-[var(--ink)] font-medium' : 'text-[var(--ink-faint)]'
                      }`}
                    >
                      {cat.label}
                    </span>
                  </div>
                  <div className="font-mono text-[11px] tabular-nums shrink-0 flex items-center gap-2">
                    <span className={cat.tokens > 0 ? 'text-[var(--ink)]' : 'text-[var(--ink-faint)]'}>
                      {formatTokenCount(cat.tokens)}
                    </span>
                    <span className="text-[var(--ink-faint)] w-10 text-right">
                      {hasKnownWindow ? `${cat.percent.toFixed(1)}%` : `${cat.share_of_used.toFixed(0)}%`}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* Footer: Remaining Space & 80% Auto-Compaction Policy */}
          <div className="px-4 py-2.5 border-t border-[var(--rule)] bg-[var(--paper-sunk)]/75 space-y-1">
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="text-[var(--ink-soft)] font-mono tabular-nums">
                {hasKnownWindow && typeof usage.free_tokens === 'number'
                  ? `${formatTokenCount(usage.free_tokens)} remaining`
                  : 'Limit not reported'}
              </span>
              <span className="text-[var(--ink-faint)]">
                {hasKnownWindow
                  ? `Auto-compacts at ${usage.auto_compact_threshold_percent}% only`
                  : 'Auto-compact requires known window'}
              </span>
            </div>
            {usage.compacted_count > 0 && (
              <div className="flex items-center gap-1.5 text-[11px] text-[var(--jade)]">
                <Layers className="w-3 h-3 shrink-0" />
                <span>
                  Compacted {usage.compacted_count}{' '}
                  {usage.compacted_count === 1 ? 'time' : 'times'}
                  {usage.tokens_saved_last_compaction > 0
                    ? ` · saved ${formatTokenCount(usage.tokens_saved_last_compaction)} tokens`
                    : ''}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
