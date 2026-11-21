import { Menu } from 'lucide-react';
import { useEffect, useState } from 'react';
import { ChatView } from './components/ChatView.tsx';
import { FilesystemView } from './components/FilesystemView.tsx';
import { LoginGate } from './components/LoginGate.tsx';
import { NemofyLogo } from './components/NemofyLogo.tsx';
import { SettingsView } from './components/SettingsView.tsx';
import { Sidebar } from './components/Sidebar.tsx';
import { checkAuthSession, fetchLLMSettings, logoutSession, saveLLMSettings } from './lib/apiClient.ts';
import type { NavSection, PersistedLLMConfig, SettingsTab, ThemeMode } from './types/agent.ts';

function getInitialTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'light';
  const saved = window.localStorage.getItem('nemofy_theme');
  if (saved === 'dark' || saved === 'light') return saved;
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

export function App() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [authChecked, setAuthChecked] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const [username, setUsername] = useState('admin');

  const [activeSection, setActiveSection] = useState<NavSection>('chat');
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('llm');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [llmConfig, setLlmConfig] = useState<PersistedLLMConfig | null>(null);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    try {
      window.localStorage.setItem('nemofy_theme', theme);
    } catch {
      // ignore storage errors
    }
  }, [theme]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      const session = await checkAuthSession();
      if (!mounted) return;
      setAuthenticated(Boolean(session.authenticated));
      if (session.username) setUsername(session.username);
      setAuthChecked(true);
    })();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!authenticated) return;
    let mounted = true;
    (async () => {
      try {
        const res = await fetchLLMSettings();
        if (mounted) setLlmConfig(res.config);
      } catch {
        // non-blocking
      }
    })();
    return () => {
      mounted = false;
    };
  }, [authenticated]);

  function toggleTheme() {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }

  async function handleLogout() {
    await logoutSession();
    setAuthenticated(false);
    setLlmConfig(null);
    setActiveSection('chat');
  }

  async function handlePrimaryThinkingChange(nextEffort: string) {
    if (!llmConfig) return;
    const updated: PersistedLLMConfig = {
      ...llmConfig,
      primary: {
        ...llmConfig.primary,
        reasoning_effort: nextEffort,
      },
    };
    setLlmConfig(updated);
    try {
      const saved = await saveLLMSettings(updated);
      setLlmConfig(saved.config);
    } catch {
      // Keep local optimistic selection; sendChatMessage still sends explicit reasoningEffort
    }
  }

  if (!authChecked) {
    return (
      <div className="h-screen flex items-center justify-center bg-[var(--paper)] text-[var(--ink-soft)]">
        <div className="flex items-center gap-3">
          <NemofyLogo size={26} />
          <span className="font-display text-sm font-medium">Loading Nemofy Agent...</span>
        </div>
      </div>
    );
  }

  if (!authenticated) {
    return (
      <LoginGate
        theme={theme}
        onToggleTheme={toggleTheme}
        onAuthenticated={(user) => {
          setUsername(user);
          setAuthenticated(true);
        }}
      />
    );
  }

  return (
    <div className="h-screen flex flex-col lg:flex-row bg-[var(--paper)] text-[var(--ink)] overflow-hidden">
      {/* Mobile Header Bar */}
      <header className="lg:hidden h-14 px-4 border-b border-[var(--rule)] bg-[var(--paper-sunk)] flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={() => setMobileMenuOpen(true)}
            aria-label="Open navigation menu"
            className="p-1.5 rounded-lg border border-[var(--rule)] bg-[var(--paper)] text-[var(--ink)] cursor-pointer"
          >
            <Menu className="w-4 h-4" />
          </button>
          <NemofyLogo size={22} />
          <span className="font-display font-semibold text-sm text-[var(--ink)]">Nemofy Agent</span>
        </div>
        <span className="text-xs font-medium capitalize text-[var(--ink-soft)]">{activeSection}</span>
      </header>

      {/* Left Sidebar */}
      <Sidebar
        activeSection={activeSection}
        onSelectSection={(section) => {
          setActiveSection(section);
          if (section === 'settings') setSettingsTab('llm');
        }}
        theme={theme}
        onToggleTheme={toggleTheme}
        username={username}
        onLogout={() => void handleLogout()}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      {/* Active Workspace View */}
      <main className="flex-1 min-w-0 min-h-0 overflow-hidden">
        {activeSection === 'chat' && (
          <ChatView
            llmConfig={llmConfig}
            onPrimaryThinkingChange={(lvl) => void handlePrimaryThinkingChange(lvl)}
            onOpenLLMSettings={() => {
              setSettingsTab('llm');
              setActiveSection('settings');
            }}
          />
        )}
        {activeSection === 'filesystem' && <FilesystemView />}
        {activeSection === 'settings' && (
          <SettingsView
            theme={theme}
            onSetTheme={setTheme}
            initialTab={settingsTab}
            onLLMConfigSaved={(next) => setLlmConfig(next)}
          />
        )}
      </main>
    </div>
  );
}

export default App;
