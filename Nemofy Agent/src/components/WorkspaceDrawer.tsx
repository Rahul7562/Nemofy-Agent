import { FolderTree, X } from 'lucide-react';
import { useEffect } from 'react';
import { FilesystemView } from './FilesystemView.tsx';

interface WorkspaceDrawerProps {
  open: boolean;
  onClose: () => void;
  targetPath?: string | null;
  refreshKey?: number;
}

export function WorkspaceDrawer({
  open,
  onClose,
  targetPath = null,
  refreshKey = 0,
}: WorkspaceDrawerProps) {
  useEffect(() => {
    if (!open) return;
    function handleEscape(e: KeyboardEvent) {
      if (e.key === 'Escape' && window.innerWidth < 1024) {
        onClose();
      }
    }
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [open, onClose]);

  if (!open) return null;

  const panelBody = (
    <div className="flex flex-col h-full w-full bg-[var(--paper)] border-l border-[var(--rule)]">
      <div className="h-14 px-4 border-b border-[var(--rule)]/70 flex items-center justify-between gap-2 shrink-0 bg-[var(--paper)]">
        <div className="flex items-center gap-2 min-w-0">
          <FolderTree className="w-4 h-4 text-[var(--signal)] shrink-0" />
          <span className="font-display font-semibold text-sm text-[var(--ink)] truncate">
            Workspace
          </span>
          <span className="font-mono text-[11px] text-[var(--ink-faint)] truncate">
            /workspace
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close workspace drawer"
          className="h-8 w-8 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 min-h-0 overflow-hidden">
        <FilesystemView
          compact
          initialSelectedPath={targetPath}
          refreshKey={refreshKey}
        />
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Split Panel */}
      <aside
        data-testid="workspace-drawer"
        aria-label="Workspace files"
        className="hidden lg:flex h-full w-[400px] xl:w-[460px] shrink-0 overflow-hidden"
      >
        {panelBody}
      </aside>

      {/* Mobile / Tablet Slide-Over Drawer */}
      <div className="fixed inset-0 z-50 lg:hidden flex justify-end">
        <div
          className="fixed inset-0 bg-black/45 backdrop-blur-[1px]"
          onClick={onClose}
          aria-hidden="true"
        />
        <aside
          data-testid="workspace-drawer-mobile"
          aria-label="Workspace files"
          className="relative z-10 flex h-full w-full max-w-[420px] shadow-2xl"
        >
          {panelBody}
        </aside>
      </div>
    </>
  );
}
