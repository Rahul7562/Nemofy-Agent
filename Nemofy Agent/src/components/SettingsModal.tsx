import { Settings, X } from 'lucide-react';
import { useEffect } from 'react';
import type {
  ExecutionAccessMode,
  PersistedLLMConfig,
  ProviderAvailability,
  SettingsTab,
  ThemeMode,
} from '../types/agent.ts';
import { SettingsView } from './SettingsView.tsx';

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  theme: ThemeMode;
  onSetTheme: (mode: ThemeMode) => void;
  initialTab: SettingsTab;
  onLLMConfigSaved: (
    config: PersistedLLMConfig,
    providers?: Record<string, ProviderAvailability>,
  ) => void;
  activePrimaryReasoningEffort?: string;
  activeAccessMode?: ExecutionAccessMode;
}

export function SettingsModal({
  open,
  onClose,
  theme,
  onSetTheme,
  initialTab,
  onLLMConfigSaved,
  activePrimaryReasoningEffort,
  activeAccessMode,
}: SettingsModalProps) {
  useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onClose();
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Settings"
      data-testid="settings-modal"
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 lg:p-6"
    >
      {/* Backdrop Overlay */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-[2px]"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Centered Dialog Container */}
      <div className="relative z-10 max-w-[1280px] w-[96vw] h-[90vh] rounded-2xl border border-[var(--rule-strong)] bg-[var(--paper)] shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="h-16 px-6 sm:px-8 border-b border-[var(--rule)] flex items-center justify-between gap-4 shrink-0 bg-[var(--paper)]">
          <div className="flex items-center gap-3 min-w-0">
            <Settings className="w-4 h-4 text-[var(--signal)] shrink-0" />
            <h2 className="font-display text-lg font-semibold tracking-tight text-[var(--ink)] truncate">
              Settings
            </h2>
            <span className="hidden sm:inline-block h-4 w-px bg-[var(--rule)] shrink-0" aria-hidden="true" />
            <span className="hidden sm:inline text-xs text-[var(--ink-faint)] truncate">
              Models, security, skills &amp; runtime
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="h-9 w-9 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Settings Content */}
        <div className="flex-1 min-h-0 overflow-hidden">
          <SettingsView
            theme={theme}
            onSetTheme={onSetTheme}
            initialTab={initialTab}
            onLLMConfigSaved={onLLMConfigSaved}
            activePrimaryReasoningEffort={activePrimaryReasoningEffort}
            activeAccessMode={activeAccessMode}
            embedded
          />
        </div>
      </div>
    </div>
  );
}
