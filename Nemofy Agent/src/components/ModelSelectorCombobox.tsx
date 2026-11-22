import { Check, ChevronDown, Loader2, RefreshCw, Search, Sparkles, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchProviderModels } from '../lib/apiClient.ts';
import {
  filterProviderModels,
  formatContextLength,
  formatOutputLength,
  getProviderPreset,
  mergeProviderModels,
} from '../lib/llmPresets.ts';
import { resolveModelLimits } from '../lib/contextUtils.ts';
import type { ModelRoleKey, ProviderModelOption } from '../types/agent.ts';

interface ModelSelectorComboboxProps {
  id: string;
  role: ModelRoleKey;
  provider: string;
  baseUrl: string;
  apiKeyDraft: string;
  value: string;
  onChange: (nextModel: string) => void;
  onSelectOption?: (option: ProviderModelOption) => void;
  onCatalogLoaded?: (models: ProviderModelOption[]) => void;
}

const MAX_VISIBLE_OPTIONS = 120;

export function ModelSelectorCombobox({
  id,
  role,
  provider,
  baseUrl,
  apiKeyDraft,
  value,
  onChange,
  onSelectOption,
  onCatalogLoaded,
}: ModelSelectorComboboxProps) {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [liveModels, setLiveModels] = useState<ProviderModelOption[]>([]);
  const [source, setSource] = useState<'idle' | 'live' | 'preset' | 'error'>('idle');
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [highlightIndex, setHighlightIndex] = useState(0);
  const [hasNavigatedList, setHasNavigatedList] = useState(false);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const requestSeqRef = useRef(0);
  const lastFetchedKeyRef = useRef<string>('');
  const lastFetchedLiveRef = useRef<boolean>(false);
  const wasOpenRef = useRef<boolean>(false);
  const onCatalogLoadedRef = useRef(onCatalogLoaded);
  onCatalogLoadedRef.current = onCatalogLoaded;

  const preset = getProviderPreset(provider);
  const providerLabel = preset?.label || provider;

  const loadModels = useCallback(
    async (force = false) => {
      const trimmedKey = apiKeyDraft.trim();
      const cacheKey = `${provider}|${baseUrl.trim()}|${trimmedKey}`;
      if (!force && lastFetchedKeyRef.current === cacheKey && lastFetchedLiveRef.current) {
        return;
      }

      const seq = ++requestSeqRef.current;
      setLoading(true);
      setFetchError(null);

      try {
        const res = await fetchProviderModels({
          provider,
          base_url: baseUrl.trim() || undefined,
          api_key: trimmedKey || undefined,
          role,
        });
        if (requestSeqRef.current !== seq) return;

        lastFetchedKeyRef.current = cacheKey;
        if (res.ok && res.models.length > 0) {
          lastFetchedLiveRef.current = res.source !== 'preset';
          setLiveModels(res.models);
          setSource(res.source === 'preset' ? 'preset' : 'live');
          setFetchError(null);
          onCatalogLoadedRef.current?.(res.models);
        } else {
          lastFetchedLiveRef.current = false;
          setLiveModels([]);
          setSource(res.error ? 'error' : 'preset');
          setFetchError(res.error || 'No models returned by this endpoint.');
        }
      } catch (err) {
        if (requestSeqRef.current !== seq) return;
        lastFetchedKeyRef.current = cacheKey;
        lastFetchedLiveRef.current = false;
        setLiveModels([]);
        setSource('error');
        setFetchError(err instanceof Error ? err.message : 'Unable to fetch models.');
      } finally {
        if (requestSeqRef.current === seq) {
          setLoading(false);
        }
      }
    },
    [provider, baseUrl, apiKeyDraft, role],
  );

  // Reset cached catalog whenever provider or baseUrl changes
  useEffect(() => {
    requestSeqRef.current += 1;
    lastFetchedKeyRef.current = '';
    lastFetchedLiveRef.current = false;
    setLiveModels([]);
    setSource('idle');
    setFetchError(null);
    setSearchQuery('');
    setHighlightIndex(0);
  }, [provider, baseUrl]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!open) return;
    function handleOutsideClick(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [open]);

  // Trigger live model discovery once when dropdown transitions closed -> open
  useEffect(() => {
    if (open && !wasOpenRef.current) {
      void loadModels(false);
    }
    wasOpenRef.current = open;
  }, [open, loadModels]);

  const mergedOptions = useMemo(
    () => mergeProviderModels(liveModels, preset?.defaultModels || [], value),
    [liveModels, preset?.defaultModels, value],
  );

  const filteredOptions = useMemo(
    () => filterProviderModels(mergedOptions, searchQuery),
    [mergedOptions, searchQuery],
  );

  const visibleOptions = useMemo(
    () => filteredOptions.slice(0, MAX_VISIBLE_OPTIONS),
    [filteredOptions],
  );

  useEffect(() => {
    setHighlightIndex(0);
    setHasNavigatedList(false);
  }, [searchQuery, provider, open]);

  function selectModel(modelId: string, option?: ProviderModelOption) {
    if (option && onSelectOption) {
      onSelectOption(option);
    } else {
      onChange(modelId);
    }
    setOpen(false);
    setSearchQuery('');
    setHasNavigatedList(false);
  }

  function handleKeyDown(event: React.KeyboardEvent) {
    if (!open) {
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        setOpen(true);
        setHasNavigatedList(true);
      }
      return;
    }

    if (event.key === 'Escape') {
      event.preventDefault();
      setOpen(false);
      return;
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setHasNavigatedList(true);
      setHighlightIndex((idx) => Math.min(visibleOptions.length - 1, idx + 1));
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHasNavigatedList(true);
      setHighlightIndex((idx) => Math.max(0, idx - 1));
      return;
    }
    if (event.key === 'Enter') {
      const inSearchBox =
        typeof document !== 'undefined' && document.activeElement === searchInputRef.current;
      // Only select from the dropdown if the user actively navigated the list or filtered inside search.
      // Never hijack an empty/manually-typed primary model input with visibleOptions[0].
      if (hasNavigatedList || (inSearchBox && searchQuery.trim().length > 0)) {
        event.preventDefault();
        const target = visibleOptions[highlightIndex];
        if (target) {
          selectModel(target.id, target);
        } else if (searchQuery.trim()) {
          selectModel(searchQuery.trim());
        }
      } else {
        event.preventDefault();
        setOpen(false);
      }
    }
  }

  const trimmedQuery = searchQuery.trim();
  const hasExactQueryMatch = Boolean(
    trimmedQuery &&
      mergedOptions.some((m) => m.id.toLowerCase() === trimmedQuery.toLowerCase()),
  );

  return (
    <div ref={containerRef} className="relative" onKeyDown={handleKeyDown}>
      <div className="flex items-center gap-1.5">
        <div className="relative flex-1 min-w-0">
          <input
            id={id}
            type="text"
            role="combobox"
            aria-expanded={open}
            aria-controls={`${id}-listbox`}
            aria-autocomplete="list"
            value={value}
            onClick={() => setOpen(true)}
            onFocus={() => setOpen(true)}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Select or type a model ID..."
            className="w-full h-10 pl-3 pr-14 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none focus:border-[var(--signal)]"
          />
          {value.trim().length > 0 && (
            <button
              type="button"
              tabIndex={-1}
              onClick={() => onChange('')}
              aria-label="Clear model selection"
              title="Clear model selection"
              className="absolute right-8 top-1/2 -translate-y-1/2 p-1 text-[var(--ink-faint)] hover:text-[var(--ink)] cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            type="button"
            tabIndex={-1}
            onClick={() => setOpen((prev) => !prev)}
            aria-label={open ? 'Close model list' : 'Open model list'}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-[var(--ink-faint)] hover:text-[var(--ink)] cursor-pointer"
          >
            <ChevronDown className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`} />
          </button>
        </div>

        <button
          type="button"
          onClick={() => {
            setOpen(true);
            void loadModels(true);
          }}
          disabled={loading}
          title={`Fetch live models from ${providerLabel}`}
          aria-label={`Refresh models from ${providerLabel}`}
          className="h-10 w-10 inline-flex items-center justify-center rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] text-[var(--ink-soft)] hover:text-[var(--signal)] disabled:opacity-50 transition-colors cursor-pointer shrink-0"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Subtle status hint below input */}
      <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px] text-[var(--ink-faint)]">
        <span className="truncate">
          {loading
            ? `Analyzing ${providerLabel} catalog...`
            : source === 'live'
              ? `${liveModels.length} live models available from ${providerLabel}`
              : `Click field to load & select from ${providerLabel} models`}
        </span>
        {source === 'live' && (
          <span className="inline-flex items-center gap-1 text-[var(--jade)] shrink-0 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--jade)]" />
            Live
          </span>
        )}
      </div>

      {/* Dropdown Popover */}
      {open && (
        <div
          id={`${id}-listbox`}
          role="listbox"
          aria-label={`${providerLabel} models`}
          className="absolute left-0 right-0 top-[44px] z-40 rounded-xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-lg overflow-hidden"
        >
          {/* Search Input */}
          <div className="p-2 border-b border-[var(--rule)] bg-[var(--paper-sunk)]">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[var(--ink-faint)] absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={`Filter ${providerLabel} models (e.g. nemotron, deepseek, free)...`}
                aria-label={`Filter ${providerLabel} models`}
                className="w-full h-8 pl-8 pr-3 rounded-md border border-[var(--rule)] bg-[var(--paper)] text-xs text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none focus:border-[var(--signal)]"
              />
            </div>

            <div className="mt-1.5 flex items-center justify-between gap-2 px-0.5 text-[11px]">
              {loading ? (
                <span className="inline-flex items-center gap-1.5 text-[var(--signal)] font-medium">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Querying {providerLabel} /models endpoint...
                </span>
              ) : source === 'live' ? (
                <span className="text-[var(--jade)] font-medium">
                  {trimmedQuery
                    ? `${filteredOptions.length} of ${mergedOptions.length} matching models`
                    : `${liveModels.length} live models from ${providerLabel}`}
                </span>
              ) : fetchError ? (
                <span className="text-[var(--honey)] truncate" title={fetchError}>
                  {mergedOptions.length > 0
                    ? `Preset catalog (${mergedOptions.length}) — ${fetchError}`
                    : fetchError}
                </span>
              ) : (
                <span className="text-[var(--ink-faint)]">
                  {mergedOptions.length} recommended models for {providerLabel}
                </span>
              )}

              <button
                type="button"
                onClick={() => void loadModels(true)}
                disabled={loading}
                className="text-[11px] font-medium text-[var(--signal)] hover:underline disabled:opacity-50 cursor-pointer shrink-0"
              >
                Refresh
              </button>
            </div>
          </div>

          {/* Model Options List */}
          <div className="max-h-64 overflow-y-auto scrollbar-none divide-y divide-[var(--rule)]/50">
            {visibleOptions.length > 0 ? (
              visibleOptions.map((item, idx) => {
                const isSelected = item.id === value;
                const isHighlighted = hasNavigatedList && idx === highlightIndex;
                const fallbackLimits =
                  !item.context_length || !item.max_output_tokens
                    ? resolveModelLimits(provider, item.id, item.context_length, item.max_output_tokens)
                    : null;
                const displayCtx =
                  item.context_length ||
                  (fallbackLimits?.contextWindowSource === 'model_registry'
                    ? fallbackLimits.contextWindow
                    : null);
                const displayOut =
                  item.max_output_tokens ||
                  (fallbackLimits?.maxOutputTokensSource === 'model_registry'
                    ? fallbackLimits.maxOutputTokens
                    : null);
                const ctxLabel = formatContextLength(displayCtx);
                const outLabel = formatOutputLength(displayOut);

                return (
                  <button
                    key={item.id}
                    type="button"
                    role="option"
                    aria-selected={isSelected}
                    onMouseEnter={() => {
                      setHighlightIndex(idx);
                      setHasNavigatedList(true);
                    }}
                    onClick={() => selectModel(item.id, item)}
                    className={`w-full px-3 py-2 text-left flex items-center justify-between gap-2 transition-colors cursor-pointer ${
                      isSelected
                        ? 'bg-[var(--blush)] text-[var(--signal)]'
                        : isHighlighted
                          ? 'bg-[var(--paper-sunk)] text-[var(--ink)]'
                          : 'bg-transparent text-[var(--ink)] hover:bg-[var(--paper-sunk)]'
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="font-mono text-xs truncate font-medium">{item.id}</div>
                      {item.name && item.name !== item.id && (
                        <div className="text-[11px] text-[var(--ink-faint)] truncate">
                          {item.name}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      {item.input_modalities &&
                        item.input_modalities
                          .filter((m) => m === 'image' || m === 'audio' || m === 'video')
                          .slice(0, 3)
                          .map((mod) => (
                            <span
                              key={mod}
                              className="px-1 py-0.5 rounded text-[9px] font-mono uppercase bg-[var(--paper-sunk)] border border-[var(--rule)] text-[var(--ink-soft)]"
                            >
                              {mod === 'image' ? 'img' : mod === 'audio' ? 'aud' : 'vid'}
                            </span>
                          ))}
                      {item.is_free && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[var(--jade)]/12 text-[var(--jade)]">
                          free
                        </span>
                      )}
                      {(ctxLabel || outLabel) && (
                        <span className="text-[10px] font-mono text-[var(--ink-faint)] tabular-nums">
                          {[ctxLabel, outLabel].filter(Boolean).join(' · ')}
                        </span>
                      )}
                      {item.source === 'preset' && (
                        <span className="text-[10px] text-[var(--ink-faint)]">preset</span>
                      )}
                      {isSelected && <Check className="w-3.5 h-3.5 text-[var(--signal)]" />}
                    </div>
                  </button>
                );
              })
            ) : (
              <div className="px-3 py-5 text-center text-xs text-[var(--ink-faint)]">
                {loading
                  ? 'Loading models...'
                  : 'No matching models found in catalog.'}
              </div>
            )}

            {filteredOptions.length > MAX_VISIBLE_OPTIONS && (
              <div className="px-3 py-2 text-[11px] text-center text-[var(--ink-faint)] bg-[var(--paper-sunk)]">
                Showing top {MAX_VISIBLE_OPTIONS} of {filteredOptions.length} matches — refine search to narrow down.
              </div>
            )}

            {trimmedQuery && !hasExactQueryMatch && (
              <button
                type="button"
                onClick={() => selectModel(trimmedQuery)}
                className="w-full px-3 py-2.5 text-left flex items-center gap-2 bg-[var(--paper-sunk)] hover:bg-[var(--blush)] text-xs text-[var(--signal)] font-medium cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">
                  Use custom model ID: <code className="font-mono">{trimmedQuery}</code>
                </span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
