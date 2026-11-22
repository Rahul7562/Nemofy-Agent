import {
  ChevronUp,
  LogOut,
  Moon,
  PanelLeftClose,
  Search,
  Settings,
  SquarePen,
  Sun,
  Trash2,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { filterThreadsByQuery, groupThreadsByDate } from '../lib/threadGrouping.ts';
import type { ChatThread, ThemeMode } from '../types/agent.ts';
import { NemofyLogo } from './NemofyLogo.tsx';

interface SidebarProps {
  threads: ChatThread[];
  activeThreadId: string | null;
  loadingThreads: boolean;
  onNewThread: () => void;
  onSelectThread: (id: string) => void;
  onDeleteThread: (id: string) => void;
  collapsed: boolean;
  onCollapse: () => void;
  onExpand: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  username: string;
  onOpenSettings: () => void;
  onLogout: () => void;
}

export function Sidebar({
  threads,
  activeThreadId,
  loadingThreads,
  onNewThread,
  onSelectThread,
  onDeleteThread,
  collapsed,
  onCollapse,
  onExpand,
  mobileOpen,
  onCloseMobile,
  theme,
  onToggleTheme,
  username,
  onOpenSettings,
  onLogout,
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const desktopSearchRef = useRef<HTMLInputElement>(null);
  const mobileSearchRef = useRef<HTMLInputElement>(null);
  const userMenuRef = useRef<HTMLDivElement>(null);
  const mobileUserMenuRef = useRef<HTMLDivElement>(null);

  const filteredThreads = useMemo(
    () => filterThreadsByQuery(threads, searchQuery),
    [threads, searchQuery],
  );
  const groupedThreads = useMemo(
    () => groupThreadsByDate(filteredThreads),
    [filteredThreads],
  );

  // Cmd/Ctrl+K focuses thread search (expanding sidebar if collapsed on desktop)
  useEffect(() => {
    function handleShortcut(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (window.innerWidth >= 1024) {
          if (collapsed) onExpand();
          window.setTimeout(() => desktopSearchRef.current?.focus(), 60);
        } else if (mobileOpen) {
          mobileSearchRef.current?.focus();
        }
      }
    }
    window.addEventListener('keydown', handleShortcut);
    return () => window.removeEventListener('keydown', handleShortcut);
  }, [collapsed, onExpand, mobileOpen]);

  // Close user profile popover on outside click or Escape
  useEffect(() => {
    if (!userMenuOpen) return;
    function handleOutside(e: MouseEvent) {
      const target = e.target as Node;
      const inDesktop = userMenuRef.current?.contains(target);
      const inMobile = mobileUserMenuRef.current?.contains(target);
      if (!inDesktop && !inMobile) {
        setUserMenuOpen(false);
      }
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setUserMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('keydown', handleKey);
    };
  }, [userMenuOpen]);

  // Close mobile drawer on Escape
  useEffect(() => {
    if (!mobileOpen) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onCloseMobile();
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [mobileOpen, onCloseMobile]);

  function renderSidebarBody(isMobile: boolean) {
    return (
      <div className="flex flex-col h-full w-[260px] bg-[var(--paper-sunk)] border-r border-[var(--rule)] shrink-0 select-none">
        {/* Top Bar: Logo + Brand + Collapse Toggle */}
        <div className="h-14 px-3.5 flex items-center justify-between gap-2 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0 pl-1">
            <NemofyLogo size={24} className="shrink-0" />
            <span className="font-display font-semibold text-[15px] tracking-tight text-[var(--ink)] truncate whitespace-nowrap">
              Nemofy Agent
            </span>
          </div>

          {isMobile ? (
            <button
              type="button"
              onClick={onCloseMobile}
              aria-label="Close sidebar"
              className="h-8 w-8 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper)]/80 transition-colors cursor-pointer shrink-0"
            >
              <X className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="button"
              onClick={onCollapse}
              aria-label="Collapse sidebar"
              title="Collapse sidebar"
              className="h-8 w-8 inline-flex items-center justify-center rounded-lg text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper)]/80 transition-colors cursor-pointer shrink-0"
            >
              <PanelLeftClose className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* New Chat Action + Thread Search */}
        <div className="px-3 pb-2.5 space-y-2 shrink-0">
          <button
            type="button"
            onClick={() => {
              onNewThread();
              if (isMobile) onCloseMobile();
            }}
            className="w-full h-9 px-3 inline-flex items-center justify-between gap-2 rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] hover:border-[var(--rule-strong)] hover:bg-[var(--paper)] text-[13px] font-medium text-[var(--ink)] transition-colors cursor-pointer shadow-[0_1px_2px_rgba(22,28,45,0.03)]"
          >
            <span className="inline-flex items-center gap-2 truncate">
              <SquarePen className="w-3.5 h-3.5 text-[var(--signal)] shrink-0" />
              <span>New chat</span>
            </span>
          </button>

          <div className="relative">
            <Search className="w-3.5 h-3.5 text-[var(--ink-faint)] absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            <label htmlFor={isMobile ? 'sidebar-thread-search-mobile' : 'sidebar-thread-search'} className="sr-only">
              Search conversations
            </label>
            <input
              ref={isMobile ? mobileSearchRef : desktopSearchRef}
              id={isMobile ? 'sidebar-thread-search-mobile' : 'sidebar-thread-search'}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chats..."
              className="w-full h-8 pl-8 pr-7 rounded-lg border border-transparent bg-[var(--paper)]/70 hover:bg-[var(--paper)] focus:bg-[var(--paper)] focus:border-[var(--rule-strong)] text-xs text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none transition-colors"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                aria-label="Clear search"
                className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded text-[var(--ink-faint)] hover:text-[var(--ink)] cursor-pointer"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        {/* Date-Grouped Thread List */}
        <nav
          aria-label="Chat history"
          className="flex-1 overflow-y-auto scrollbar-none px-2.5 pb-3 space-y-4"
        >
          {loadingThreads ? (
            <div className="px-2.5 py-4 space-y-2">
              <div className="h-3 w-16 rounded bg-[var(--rule)]/70 animate-pulse" />
              <div className="h-8 w-full rounded-lg bg-[var(--rule)]/50 animate-pulse" />
              <div className="h-8 w-5/6 rounded-lg bg-[var(--rule)]/40 animate-pulse" />
            </div>
          ) : groupedThreads.length === 0 ? (
            <div className="px-2.5 py-6 text-xs text-[var(--ink-faint)] leading-relaxed">
              {searchQuery.trim()
                ? 'No matching conversations.'
                : 'No conversations yet. Start a new chat with Nemo.'}
            </div>
          ) : (
            groupedThreads.map((group) => (
              <section key={group.label} aria-label={group.label} className="space-y-0.5">
                <div className="px-2.5 pb-1 text-[11px] font-medium text-[var(--ink-faint)]">
                  {group.label}
                </div>
                {group.threads.map((thread) => {
                  const selected = thread.id === activeThreadId;
                  const title = thread.title || 'Untitled session';
                  return (
                    <div
                      key={thread.id}
                      className={`group flex items-center justify-between gap-1 rounded-lg px-2.5 py-1.5 text-[13px] transition-colors ${
                        selected
                          ? 'bg-[var(--blush)] text-[var(--signal)] font-medium'
                          : 'text-[var(--ink-soft)] hover:bg-[var(--paper)]/75 hover:text-[var(--ink)]'
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          onSelectThread(thread.id);
                          if (isMobile) onCloseMobile();
                        }}
                        aria-current={selected ? 'page' : undefined}
                        className="flex-1 min-w-0 text-left truncate py-0.5 cursor-pointer"
                      >
                        {title}
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteThread(thread.id);
                        }}
                        aria-label={`Delete ${title}`}
                        title={`Delete ${title}`}
                        className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 rounded-md text-[var(--ink-faint)] hover:text-[var(--flare)] hover:bg-[var(--flare)]/10 transition-all cursor-pointer shrink-0"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  );
                })}
              </section>
            ))
          )}
        </nav>

        {/* Bottom Footer: Compact User Profile Trigger + Upward Popover Menu */}
        <div
          ref={isMobile ? mobileUserMenuRef : userMenuRef}
          className="relative p-2.5 border-t border-[var(--rule)] shrink-0"
        >
          {userMenuOpen && (
            <div
              role="menu"
              aria-label="Account menu"
              className="absolute left-2.5 right-2.5 bottom-full mb-2 rounded-xl border border-[var(--rule-strong)] bg-[var(--paper-raised)] shadow-xl p-1.5 space-y-0.5 z-40"
            >
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setUserMenuOpen(false);
                  if (isMobile) onCloseMobile();
                  onOpenSettings();
                }}
                className="w-full px-2.5 py-2 rounded-lg flex items-center gap-2.5 text-xs font-medium text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer"
              >
                <Settings className="w-4 h-4 text-[var(--ink-soft)] shrink-0" />
                <span>Settings</span>
              </button>

              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  onToggleTheme();
                }}
                className="w-full px-2.5 py-2 rounded-lg flex items-center justify-between gap-2 text-xs font-medium text-[var(--ink)] hover:bg-[var(--paper-sunk)] transition-colors cursor-pointer"
              >
                <span className="inline-flex items-center gap-2.5">
                  {theme === 'dark' ? (
                    <Sun className="w-4 h-4 text-[var(--honey)] shrink-0" />
                  ) : (
                    <Moon className="w-4 h-4 text-[var(--ink-soft)] shrink-0" />
                  )}
                  <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
                </span>
              </button>

              <div className="my-1 border-t border-[var(--rule)]" />

              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setUserMenuOpen(false);
                  if (isMobile) onCloseMobile();
                  onLogout();
                }}
                className="w-full px-2.5 py-2 rounded-lg flex items-center gap-2.5 text-xs font-medium text-[var(--flare)] hover:bg-[var(--flare)]/10 transition-colors cursor-pointer"
              >
                <LogOut className="w-4 h-4 shrink-0" />
                <span>Sign out</span>
              </button>
            </div>
          )}

          <button
            type="button"
            data-testid="sidebar-user-menu-trigger"
            onClick={() => setUserMenuOpen((v) => !v)}
            aria-haspopup="menu"
            aria-expanded={userMenuOpen}
            className={`w-full px-2.5 py-2 rounded-xl flex items-center justify-between gap-2.5 transition-colors cursor-pointer ${
              userMenuOpen
                ? 'bg-[var(--paper)] text-[var(--ink)]'
                : 'hover:bg-[var(--paper)]/80 text-[var(--ink)]'
            }`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-7 h-7 rounded-full bg-[var(--signal)] text-white font-display font-semibold text-xs flex items-center justify-center shrink-0 uppercase">
                {username.slice(0, 1)}
              </div>
              <span className="text-[13px] font-medium text-[var(--ink)] truncate">{username}</span>
            </div>
            <ChevronUp
              className={`w-4 h-4 text-[var(--ink-faint)] shrink-0 transition-transform ${
                userMenuOpen ? 'rotate-180' : ''
              }`}
            />
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Desktop collapsible sidebar */}
      <aside
        aria-label="Sidebar"
        className={`hidden lg:flex h-full shrink-0 overflow-hidden transition-[width] duration-200 ease-out ${
          collapsed ? 'w-0' : 'w-[260px]'
        }`}
      >
        {!collapsed && renderSidebarBody(false)}
      </aside>

      {/* Mobile slide-over drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/45 backdrop-blur-[1px]"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <aside aria-label="Mobile sidebar" className="relative z-10 flex h-full">
            {renderSidebarBody(true)}
          </aside>
        </div>
      )}
    </>
  );
}
