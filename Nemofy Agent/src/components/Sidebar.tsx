import {
  FolderTree,
  LogOut,
  MessageSquare,
  Moon,
  Settings,
  Sun,
  X,
} from 'lucide-react';
import type { NavSection, ThemeMode } from '../types/agent.ts';
import { NemofyLogo } from './NemofyLogo.tsx';

interface SidebarProps {
  activeSection: NavSection;
  onSelectSection: (section: NavSection) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  username: string;
  onLogout: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

const NAV_ITEMS: Array<{
  id: NavSection;
  label: string;
  description: string;
  icon: typeof MessageSquare;
}> = [
  {
    id: 'chat',
    label: 'Chat',
    description: 'Chat with Nemo',
    icon: MessageSquare,
  },
  {
    id: 'filesystem',
    label: 'Filesystem',
    description: 'Workspace /workspace files',
    icon: FolderTree,
  },
  {
    id: 'settings',
    label: 'Settings',
    description: 'LLMs, security & skills',
    icon: Settings,
  },
];

export function Sidebar({
  activeSection,
  onSelectSection,
  theme,
  onToggleTheme,
  username,
  onLogout,
  mobileOpen,
  onCloseMobile,
}: SidebarProps) {
  const sidebarContent = (
    <div className="flex flex-col h-full bg-[var(--paper-sunk)] border-r border-[var(--rule)] w-[260px] shrink-0">
      {/* Brand Header */}
      <div className="h-16 px-5 flex items-center justify-between border-b border-[var(--rule)]">
        <div className="flex items-center gap-2.5 min-w-0">
          <NemofyLogo size={26} />
          <div className="min-w-0">
            <div className="font-display font-semibold text-[15px] tracking-tight text-[var(--ink)] truncate">
              Nemofy Agent
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={onCloseMobile}
          aria-label="Close navigation"
          className="lg:hidden p-1.5 rounded-md text-[var(--ink-soft)] hover:text-[var(--ink)] cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Primary Workspace Navigation */}
      <nav aria-label="Primary" className="flex-1 p-3 space-y-1 overflow-y-auto scrollbar-none">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = activeSection === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onSelectSection(item.id);
                onCloseMobile();
              }}
              aria-current={active ? 'page' : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors cursor-pointer ${
                active
                  ? 'bg-[var(--blush)] text-[var(--signal)] font-semibold'
                  : 'text-[var(--ink-soft)] hover:bg-[var(--paper)]/70 hover:text-[var(--ink)] font-medium'
              }`}
            >
              <Icon className={`w-4 h-4 shrink-0 ${active ? 'text-[var(--signal)]' : ''}`} />
              <div className="min-w-0">
                <div className="text-sm leading-tight">{item.label}</div>
                <div
                  className={`text-[11px] leading-tight mt-0.5 truncate ${
                    active ? 'text-[var(--signal)]/80 font-normal' : 'text-[var(--ink-faint)] font-normal'
                  }`}
                >
                  {item.description}
                </div>
              </div>
            </button>
          );
        })}
      </nav>

      {/* Bottom Controls: Theme + Signed-in User + Sign Out */}
      <div className="p-3 border-t border-[var(--rule)] space-y-2">
        <button
          type="button"
          onClick={onToggleTheme}
          className="w-full flex items-center justify-between px-3 py-2 rounded-lg border border-[var(--rule)] bg-[var(--paper-raised)] text-xs font-medium text-[var(--ink-soft)] hover:text-[var(--ink)] hover:border-[var(--rule-strong)] transition-colors cursor-pointer"
        >
          <span className="flex items-center gap-2">
            {theme === 'dark' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
            <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
          </span>
          <span className="font-mono text-[10px] uppercase text-[var(--ink-faint)]">
            {theme}
          </span>
        </button>

        <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-[var(--paper-raised)] border border-[var(--rule)]">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-7 h-7 rounded-md bg-[var(--signal)] text-white font-display font-semibold text-xs flex items-center justify-center shrink-0 uppercase">
              {username.slice(0, 1)}
            </div>
            <div className="min-w-0">
              <div className="text-xs font-semibold text-[var(--ink)] truncate">{username}</div>
              <div className="text-[10px] text-[var(--ink-faint)]">Administrator</div>
            </div>
          </div>
          <button
            type="button"
            onClick={onLogout}
            title="Sign out"
            aria-label="Sign out"
            className="p-1.5 rounded-md text-[var(--ink-faint)] hover:text-[var(--flare)] hover:bg-[var(--flare)]/10 transition-colors cursor-pointer"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop persistent sidebar */}
      <aside className="hidden lg:flex h-full shrink-0">{sidebarContent}</aside>

      {/* Mobile slide-over drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/40 backdrop-blur-[1px]"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <aside className="relative z-10 flex h-full">{sidebarContent}</aside>
        </div>
      )}
    </>
  );
}
