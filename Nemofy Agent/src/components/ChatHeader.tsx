import {
  ChevronDown,
  FolderTree,
  Menu,
  PanelLeftOpen,
  SlidersHorizontal,
  SquarePen,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { getProviderPreset } from '../lib/llmPresets.ts';
import type {
  ContextUsageSnapshot,
  PersistedLLMConfig,
  ProviderModelOption,
} from '../types/agent.ts';
import { ContextUsageRing } from './ContextUsageRing.tsx';
import { ModelSelectorCombobox } from './ModelSelectorCombobox.tsx';

interface ChatHeaderProps {
  sidebarCollapsed: boolean;
  onExpandSidebar: () => void;
  onOpenMobileSidebar: () => void;
  onNewThread: () => void;
  llmConfig: PersistedLLMConfig | null;
  onQuickSwitchPrimaryModel: (nextModel: string, option?: ProviderModelOption) => void;
  onOpenLLMSettings: () => void;
  contextUsage: ContextUsageSnapshot;
  workspaceOpen: boolean;
  onToggleWorkspace: () => void;
  workspaceChangesCount: number;
}

export function ChatHeader({
  sidebarCollapsed,
  onExpandSidebar,
  onOpenMobileSidebar,
  onNewThread,
  llmConfig,
  onQuickSwitchPrimaryModel,
  onOpenLLMSettings,
  contextUsage,
  workspaceOpen,
  onToggleWorkspace,
  workspaceChangesCount,
}: ChatHeaderProps) {
  const [modelPopoverOpen, setModelPopoverOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  const primaryProvider = llmConfig?.primary.provider || 'river_ai';
  const primaryModel = (llmConfig?.primary.model || '').trim();
  const hasPrimaryModel = primaryModel.length > 0;
  const providerLabel = getProviderPreset(primaryProvider)?.label || primaryProvider;

  useEffect(() => {
    if (!modelPopoverOpen) return;
    function handleOutside(e: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setModelPopoverOpen(false);
      }
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setModelPopoverOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('keydown', handleKey);
    };
  }, [modelPopoverOpen]);

  return (
    <header className="h-14 px-3 sm:px-5 flex items-center justify-between gap-3 bg-[var(--paper)]/85 backdrop-blur-sm border-b border-[var(--rule)]/70 shrink-0 z-20">
      {/* Left: Sidebar trigger + Inline Primary Model Switcher */}
      <div className="flex flex-1 items-center gap-1.5 min-w-0">
        {/* Mobile sidebar drawer trigger */}
        <button
          type="button"
          onClick={onOpenMobileSidebar}
          aria-label="Open sidebar"
          className="lg:hidden h-9 w-9 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer shrink-0"
        >
          <Menu className="w-4 h-4" />
        </button>

        {/* Desktop sidebar expand & quick new-chat buttons when collapsed */}
        {sidebarCollapsed && (
          <div className="hidden lg:flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={onExpandSidebar}
              aria-label="Expand sidebar"
              title="Expand sidebar"
              className="h-9 w-9 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer"
            >
              <PanelLeftOpen className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={onNewThread}
              aria-label="New chat"
              title="New chat"
              className="h-9 w-9 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer"
            >
              <SquarePen className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Inline Primary Model Picker */}
        <div ref={popoverRef} className="relative min-w-0 max-w-full">
          <button
            type="button"
            data-testid="active-primary-model-button"
            onClick={() => setModelPopoverOpen((v) => !v)}
            aria-haspopup="dialog"
            aria-expanded={modelPopoverOpen}
            title={
              hasPrimaryModel
                ? `Active Primary Model: ${primaryProvider}: ${primaryModel}`
                : 'Select Primary LLM model'
            }
            className={`h-9 px-2 sm:px-2.5 inline-flex items-center gap-1.5 sm:gap-2 rounded-xl text-left transition-colors cursor-pointer max-w-full sm:max-w-[320px] ${
              modelPopoverOpen
                ? 'bg-[var(--paper-sunk)] text-[var(--ink)]'
                : hasPrimaryModel
                  ? 'text-[var(--ink)] hover:bg-[var(--paper-sunk)]'
                  : 'bg-[var(--blush)]/70 text-[var(--signal)] hover:bg-[var(--blush)]'
            }`}
          >
            <span className="font-display font-semibold text-sm tracking-tight truncate min-w-0">
              {hasPrimaryModel ? primaryModel : 'Select primary model'}
            </span>
            <ChevronDown
              className={`w-3.5 h-3.5 text-[var(--ink-faint)] shrink-0 transition-transform ${
                modelPopoverOpen ? 'rotate-180' : ''
              }`}
            />
          </button>

          {modelPopoverOpen && (
            <div
              role="dialog"
              aria-label="Quick switch primary model"
              className="absolute left-0 top-full mt-2 w-[320px] sm:w-[376px] max-w-[calc(100vw-1.5rem)] rounded-2xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-xl p-3.5 space-y-3 z-40"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="font-display text-xs font-semibold text-[var(--ink)]">
                    Primary Model
                  </div>
                  <div className="text-[11px] text-[var(--ink-faint)] truncate">
                    Provider: {providerLabel}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setModelPopoverOpen(false);
                    onOpenLLMSettings();
                  }}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-[var(--signal)] hover:bg-[var(--blush)] transition-colors cursor-pointer shrink-0 whitespace-nowrap"
                >
                  <SlidersHorizontal className="w-3.5 h-3.5" />
                  <span>Full LLM Settings</span>
                </button>
              </div>

              {llmConfig ? (
                <ModelSelectorCombobox
                  id="header-quick-primary-model"
                  role="primary"
                  provider={primaryProvider}
                  baseUrl={llmConfig.primary.base_url}
                  apiKeyDraft=""
                  value={llmConfig.primary.model}
                  onChange={(nextModel) => {
                    onQuickSwitchPrimaryModel(nextModel);
                  }}
                  onSelectOption={(option) => {
                    onQuickSwitchPrimaryModel(option.id, option);
                    setModelPopoverOpen(false);
                  }}
                />
              ) : (
                <div className="py-3 text-xs text-[var(--ink-faint)]">
                  Loading model configuration...
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right: Context Usage Ring + Workspace Drawer Toggle */}
      <div className="flex items-center gap-2 shrink-0">
        <ContextUsageRing
          usage={contextUsage}
          modelConfigured={hasPrimaryModel}
          onOpenLLMSettings={onOpenLLMSettings}
        />

        <button
          type="button"
          data-testid="workspace-drawer-toggle"
          onClick={onToggleWorkspace}
          aria-pressed={workspaceOpen}
          aria-label={workspaceOpen ? 'Close workspace files drawer' : 'Open workspace files drawer'}
          title={workspaceOpen ? 'Close /workspace drawer' : 'Inspect /workspace files'}
          className={`relative h-9 px-2.5 inline-flex items-center gap-1.5 rounded-lg border text-xs font-medium transition-colors cursor-pointer shrink-0 whitespace-nowrap ${
            workspaceOpen
              ? 'border-[var(--signal)] bg-[var(--blush)]/75 text-[var(--signal)]'
              : 'border-[var(--rule)] bg-[var(--paper-sunk)] hover:border-[var(--rule-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)]'
          }`}
        >
          <FolderTree className="w-4 h-4 shrink-0" />
          <span className="hidden sm:inline">Workspace</span>
          {workspaceChangesCount > 0 && (
            <span
              className="ml-0.5 inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-[var(--signal)] text-white font-mono text-[10px] font-semibold tabular-nums"
              title={`${workspaceChangesCount} workspace file ${workspaceChangesCount === 1 ? 'change' : 'changes'} in this chat`}
            >
              {workspaceChangesCount}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
