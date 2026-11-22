import {
  ChevronDown,
  ChevronRight,
  FileCode2,
  FilePlus2,
  Folder,
  FolderOpen,
  FolderPlus,
  Loader2,
  RefreshCw,
  Save,
  Search,
  Trash2,
} from 'lucide-react';
import { type FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import {
  createWorkspaceFolder,
  deleteWorkspaceEntry,
  fetchWorkspaceTree,
  readWorkspaceFile,
  saveWorkspaceFile,
} from '../lib/apiClient.ts';
import {
  countWorkspaceItems,
  detectFileLanguage,
  filterWorkspaceTree,
  formatBytes,
  isValidWorkspacePath,
  normalizeWorkspacePath,
} from '../lib/filesystemUtils.ts';
import type { WorkspaceNode } from '../types/agent.ts';

function mergeFolderChildren(
  nodes: WorkspaceNode[],
  folderPath: string,
  children: WorkspaceNode[],
  truncated = false,
): WorkspaceNode[] {
  return nodes.map((node) => {
    if (node.type === 'folder' && node.path === folderPath) {
      return {
        ...node,
        children,
        hasChildren: children.length > 0,
        childCount: children.length,
        truncated,
      };
    }
    if (node.type === 'folder' && node.children && node.children.length > 0) {
      return {
        ...node,
        children: mergeFolderChildren(node.children, folderPath, children, truncated),
      };
    }
    return node;
  });
}

export function FilesystemView() {
  const [tree, setTree] = useState<WorkspaceNode[]>([]);
  const [loadingTree, setLoadingTree] = useState(true);
  const [loadingFolders, setLoadingFolders] = useState<Record<string, boolean>>({});
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedFolders, setExpandedFolders] = useState<Record<string, boolean>>({});

  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState('');
  const [savedContent, setSavedContent] = useState('');
  const [fileMeta, setFileMeta] = useState<{ size: number; modifiedAt: string } | null>(null);
  const [loadingFile, setLoadingFile] = useState(false);
  const [saving, setSaving] = useState(false);

  const [creatingMode, setCreatingMode] = useState<'file' | 'folder' | null>(null);
  const [newPathInput, setNewPathInput] = useState('');
  const [feedback, setFeedback] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);

  const isDirty = selectedPath !== null && fileContent !== savedContent;

  const refreshTree = useCallback(async () => {
    setLoadingTree(true);
    try {
      const res = await fetchWorkspaceTree();
      setTree(res.entries || []);
    } catch (err) {
      setFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to load workspace tree.',
      });
    } finally {
      setLoadingTree(false);
    }
  }, []);

  useEffect(() => {
    void refreshTree();
  }, [refreshTree]);

  const filteredTree = useMemo(
    () => filterWorkspaceTree(tree, searchQuery),
    [tree, searchQuery],
  );
  const counts = useMemo(() => countWorkspaceItems(tree), [tree]);

  async function handleSelectFile(filePath: string) {
    setFeedback(null);
    setLoadingFile(true);
    try {
      const data = await readWorkspaceFile(filePath);
      setSelectedPath(filePath);
      setFileContent(data.content);
      setSavedContent(data.content);
      setFileMeta({ size: data.size, modifiedAt: data.modifiedAt });
    } catch (err) {
      setFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to read file.',
      });
    } finally {
      setLoadingFile(false);
    }
  }

  async function handleSaveFile() {
    if (!selectedPath || saving) return;
    setSaving(true);
    setFeedback(null);
    try {
      await saveWorkspaceFile(selectedPath, fileContent);
      setSavedContent(fileContent);
      setFileMeta({
        size: new Blob([fileContent]).size,
        modifiedAt: new Date().toISOString(),
      });
      setFeedback({ type: 'ok', text: `Saved /workspace/${selectedPath}` });
      await refreshTree();
    } catch (err) {
      setFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to save file.',
      });
    } finally {
      setSaving(false);
    }
  }

  async function handleCreateSubmit(e: FormEvent) {
    e.preventDefault();
    const clean = normalizeWorkspacePath(newPathInput);
    if (!isValidWorkspacePath(clean)) {
      setFeedback({
        type: 'error',
        text: 'Enter a valid relative path inside /workspace (no ".." segments).',
      });
      return;
    }

    setFeedback(null);
    try {
      if (creatingMode === 'folder') {
        await createWorkspaceFolder(clean);
        setFeedback({ type: 'ok', text: `Created folder /workspace/${clean}` });
      } else {
        await saveWorkspaceFile(clean, '', { createOnly: true });
        setFeedback({ type: 'ok', text: `Created file /workspace/${clean}` });
        await handleSelectFile(clean);
      }
      setCreatingMode(null);
      setNewPathInput('');
      await refreshTree();
    } catch (err) {
      setFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Creation failed.',
      });
    }
  }

  async function handleDelete(targetPath: string) {
    setFeedback(null);
    try {
      await deleteWorkspaceEntry(targetPath);
      if (selectedPath === targetPath || selectedPath?.startsWith(`${targetPath}/`)) {
        setSelectedPath(null);
        setFileContent('');
        setSavedContent('');
        setFileMeta(null);
      }
      setFeedback({ type: 'ok', text: `Deleted /workspace/${targetPath}` });
      await refreshTree();
    } catch (err) {
      setFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Delete failed.',
      });
    }
  }

  async function toggleFolder(node: WorkspaceNode) {
    const folderPath = node.path;
    const currentlyOpen = expandedFolders[folderPath] ?? false;
    const nextOpen = !currentlyOpen;
    setExpandedFolders((prev) => ({ ...prev, [folderPath]: nextOpen }));

    if (nextOpen && (node.children === undefined || node.truncated)) {
      setLoadingFolders((prev) => ({ ...prev, [folderPath]: true }));
      try {
        const res = await fetchWorkspaceTree(folderPath);
        setTree((prev) =>
          mergeFolderChildren(prev, folderPath, res.entries || [], Boolean(res.truncated)),
        );
      } catch (err) {
        setFeedback({
          type: 'error',
          text: err instanceof Error ? err.message : `Unable to list /workspace/${folderPath}.`,
        });
      } finally {
        setLoadingFolders((prev) => ({ ...prev, [folderPath]: false }));
      }
    }
  }

  function renderTreeNodes(nodes: WorkspaceNode[], depth = 0) {
    const isSearching = searchQuery.trim().length > 0;
    return nodes.map((node) => {
      if (node.type === 'folder') {
        const isOpen = isSearching ? true : (expandedFolders[node.path] ?? false);
        const isLoading = Boolean(loadingFolders[node.path]);
        const itemCount =
          typeof node.childCount === 'number'
            ? node.childCount
            : Array.isArray(node.children)
              ? node.children.length
              : null;
        return (
          <div key={node.path}>
            <div
              className="group flex items-center justify-between gap-1 rounded-md px-2 py-1.5 text-xs text-[var(--ink-soft)] hover:bg-[var(--paper-sunk)] hover:text-[var(--ink)]"
              style={{ paddingLeft: `${8 + depth * 14}px` }}
            >
              <button
                type="button"
                onClick={() => void toggleFolder(node)}
                aria-expanded={isOpen}
                className="flex items-center gap-1.5 min-w-0 flex-1 text-left cursor-pointer"
              >
                {isLoading ? (
                  <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin text-[var(--signal)]" />
                ) : isOpen ? (
                  <ChevronDown className="w-3.5 h-3.5 shrink-0" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 shrink-0" />
                )}
                {isOpen ? (
                  <FolderOpen className="w-3.5 h-3.5 text-[var(--honey)] shrink-0" />
                ) : (
                  <Folder className="w-3.5 h-3.5 text-[var(--honey)] shrink-0" />
                )}
                <span className="truncate font-medium">{node.name}</span>
              </button>
              {itemCount !== null && (
                <span className="text-[10px] font-mono tabular-nums text-[var(--ink-faint)] shrink-0 group-hover:hidden">
                  {itemCount}
                </span>
              )}
              <button
                type="button"
                onClick={() => void handleDelete(node.path)}
                aria-label={`Delete folder ${node.name}`}
                className="hidden group-hover:inline-flex focus:inline-flex p-1 rounded hover:text-[var(--flare)] cursor-pointer"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
            {isOpen && node.children && node.children.length > 0 && (
              <div>{renderTreeNodes(node.children, depth + 1)}</div>
            )}
            {isOpen && !isLoading && Array.isArray(node.children) && node.children.length === 0 && (
              <div
                className="py-1 text-[11px] text-[var(--ink-faint)] italic"
                style={{ paddingLeft: `${26 + depth * 14}px` }}
              >
                Empty folder
              </div>
            )}
          </div>
        );
      }

      const active = selectedPath === node.path;
      return (
        <div
          key={node.path}
          className={`group flex items-center justify-between gap-1 rounded-md px-2 py-1.5 text-xs transition-colors ${
            active
              ? 'bg-[var(--blush)] text-[var(--signal)] font-medium'
              : 'text-[var(--ink-soft)] hover:bg-[var(--paper-sunk)] hover:text-[var(--ink)]'
          }`}
          style={{ paddingLeft: `${22 + depth * 14}px` }}
        >
          <button
            type="button"
            onClick={() => void handleSelectFile(node.path)}
            className="flex items-center gap-1.5 min-w-0 flex-1 text-left cursor-pointer"
          >
            <FileCode2 className="w-3.5 h-3.5 shrink-0 opacity-80" />
            <span className="truncate font-mono text-[12px]">{node.name}</span>
          </button>
          <span className="text-[10px] font-mono text-[var(--ink-faint)] shrink-0 group-hover:hidden">
            {formatBytes(node.size)}
          </span>
          <button
            type="button"
            onClick={() => void handleDelete(node.path)}
            aria-label={`Delete file ${node.name}`}
            className="hidden group-hover:inline-flex focus:inline-flex p-1 rounded hover:text-[var(--flare)] cursor-pointer"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      );
    });
  }

  return (
    <div className="flex flex-col md:flex-row h-full min-h-0 bg-[var(--paper)]">
      {/* Left Workspace Tree */}
      <div className="w-full md:w-72 border-b md:border-b-0 md:border-r border-[var(--rule)] bg-[var(--paper-raised)] flex flex-col shrink-0 max-h-72 md:max-h-none">
        <div className="p-3.5 border-b border-[var(--rule)] space-y-2.5">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="font-display text-sm font-semibold text-[var(--ink)]">
                Workspace Filesystem
              </h1>
              <p className="text-[11px] font-mono text-[var(--ink-faint)]">
                /workspace ({counts.files} files, {counts.folders} folders)
              </p>
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => {
                  setCreatingMode('file');
                  setNewPathInput('');
                }}
                title="New file"
                aria-label="New file"
                className="p-1.5 rounded-md border border-[var(--rule)] hover:border-[var(--rule-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)] cursor-pointer"
              >
                <FilePlus2 className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => {
                  setCreatingMode('folder');
                  setNewPathInput('');
                }}
                title="New folder"
                aria-label="New folder"
                className="p-1.5 rounded-md border border-[var(--rule)] hover:border-[var(--rule-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)] cursor-pointer"
              >
                <FolderPlus className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => void refreshTree()}
                title="Refresh workspace"
                aria-label="Refresh workspace"
                className="p-1.5 rounded-md border border-[var(--rule)] hover:border-[var(--rule-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)] cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="relative">
            <Search className="w-3.5 h-3.5 text-[var(--ink-faint)] absolute left-2.5 top-1/2 -translate-y-1/2" />
            <label htmlFor="workspace-search" className="sr-only">
              Filter workspace files
            </label>
            <input
              id="workspace-search"
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter files..."
              className="w-full h-8 pl-8 pr-2.5 rounded-lg border border-[var(--rule)] bg-[var(--paper)] text-xs text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none focus:border-[var(--signal)]"
            />
          </div>

          {creatingMode && (
            <form onSubmit={handleCreateSubmit} className="space-y-2 pt-1">
              <label htmlFor="new-entry-path" className="block text-[11px] font-medium text-[var(--ink)]">
                {creatingMode === 'file' ? 'New file path' : 'New folder path'}
              </label>
              <input
                id="new-entry-path"
                type="text"
                autoFocus
                value={newPathInput}
                onChange={(e) => setNewPathInput(e.target.value)}
                placeholder={creatingMode === 'file' ? 'notes/summary.md' : 'scripts'}
                className="w-full h-8 px-2.5 rounded-md border border-[var(--signal)] bg-[var(--paper)] font-mono text-xs text-[var(--ink)] focus:outline-none"
              />
              <div className="flex items-center gap-1.5">
                <button
                  type="submit"
                  className="px-2.5 py-1 rounded-md bg-[var(--signal)] text-white text-xs font-medium cursor-pointer"
                >
                  Create
                </button>
                <button
                  type="button"
                  onClick={() => setCreatingMode(null)}
                  className="px-2.5 py-1 rounded-md border border-[var(--rule)] text-xs text-[var(--ink-soft)] cursor-pointer"
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-none p-2 space-y-0.5">
          {loadingTree ? (
            <div className="p-3 text-xs text-[var(--ink-faint)]">Scanning /workspace...</div>
          ) : filteredTree.length === 0 ? (
            <div className="p-3 text-xs text-[var(--ink-faint)]">
              {searchQuery ? 'No matching files.' : 'Workspace is empty. Create a file or ask the agent to generate one.'}
            </div>
          ) : (
            renderTreeNodes(filteredTree)
          )}
        </div>
      </div>

      {/* Right File Editor */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        {feedback && (
          <div
            role={feedback.type === 'error' ? 'alert' : 'status'}
            className={`px-5 py-2.5 text-xs border-b border-[var(--rule)] ${
              feedback.type === 'error'
                ? 'bg-[var(--flare)]/10 text-[var(--flare)]'
                : 'bg-[var(--jade)]/10 text-[var(--jade)]'
            }`}
          >
            {feedback.text}
          </div>
        )}

        {selectedPath ? (
          <>
            <div className="h-16 px-5 border-b border-[var(--rule)] flex items-center justify-between gap-3 shrink-0 bg-[var(--paper)]">
              <div className="min-w-0">
                <div className="font-mono text-sm font-medium text-[var(--ink)] truncate">
                  /workspace/{selectedPath}
                  {isDirty && <span className="ml-2 text-xs text-[var(--honey)]">(unsaved)</span>}
                </div>
                <div className="text-[11px] text-[var(--ink-faint)]">
                  {detectFileLanguage(selectedPath)}
                  {fileMeta ? ` · ${formatBytes(fileMeta.size)}` : ''}
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  disabled={!isDirty || saving}
                  onClick={() => void handleSaveFile()}
                  className="h-8 px-3 inline-flex items-center gap-1.5 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-45 text-white text-xs font-semibold transition-colors cursor-pointer"
                >
                  <Save className="w-3.5 h-3.5" />
                  <span>{saving ? 'Saving...' : 'Save'}</span>
                </button>
                <button
                  type="button"
                  onClick={() => void handleDelete(selectedPath)}
                  aria-label="Delete current file"
                  className="h-8 px-2.5 inline-flex items-center gap-1.5 rounded-lg border border-[var(--rule)] hover:border-[var(--flare)]/50 hover:text-[var(--flare)] text-xs text-[var(--ink-soft)] transition-colors cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Delete</span>
                </button>
              </div>
            </div>

            <div className="flex-1 min-h-0 p-4 bg-[var(--paper-sunk)]/40">
              {loadingFile ? (
                <div className="p-4 text-xs text-[var(--ink-faint)]">Reading file...</div>
              ) : (
                <textarea
                  aria-label={`Editor for ${selectedPath}`}
                  value={fileContent}
                  onChange={(e) => setFileContent(e.target.value)}
                  spellCheck={false}
                  className="w-full h-full resize-none rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-4 font-mono text-xs sm:text-[13px] text-[var(--ink)] leading-relaxed focus:outline-none focus:border-[var(--signal)]"
                />
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
            <FileCode2 className="w-10 h-10 text-[var(--ink-faint)] mb-3" />
            <h2 className="font-display text-lg font-semibold text-[var(--ink)]">
              Select a workspace file
            </h2>
            <p className="mt-1.5 text-sm text-[var(--ink-soft)] max-w-md">
              Inspect, edit, and save files shared directly with the Python Agent-Core sandbox inside{' '}
              <code className="font-mono text-xs">/workspace</code>.
            </p>
            <button
              type="button"
              onClick={() => {
                setCreatingMode('file');
                setNewPathInput('notes.md');
              }}
              className="mt-5 inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] text-white text-xs font-semibold cursor-pointer"
            >
              <FilePlus2 className="w-3.5 h-3.5" />
              <span>Create new file</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
