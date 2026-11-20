import { Eye, EyeOff, Lock, Moon, Sun, ArrowRight } from 'lucide-react';
import { type FormEvent, useState } from 'react';
import { loginWithCredentials } from '../lib/apiClient.ts';
import { validateLoginInput } from '../lib/authValidation.ts';
import type { ThemeMode } from '../types/agent.ts';
import { NemofyLogo } from './NemofyLogo.tsx';

interface LoginGateProps {
  theme: ThemeMode;
  onToggleTheme: () => void;
  onAuthenticated: (username: string) => void;
}

export function LoginGate({ theme, onToggleTheme, onAuthenticated }: LoginGateProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const check = validateLoginInput(username, password);
    if (!check.valid) {
      setError(check.error || 'Please check your credentials.');
      return;
    }

    setSubmitting(true);
    try {
      const result = await loginWithCredentials(username, password);
      if (!result.ok) {
        setError(result.error || 'Invalid username or password.');
        return;
      }
      onAuthenticated(result.username || username.trim());
    } catch {
      setError('Unable to sign in right now. Please try again.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-[var(--paper)] text-[var(--ink)]">
      {/* Top Bar synced with nemofy.tech minimal header */}
      <header className="w-full max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <NemofyLogo size={28} />
          <span className="font-display font-semibold text-lg tracking-tight text-[var(--ink)]">
            Nemofy Agent
          </span>
        </div>

        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[var(--rule)] bg-[var(--paper-raised)] text-sm font-medium text-[var(--ink-soft)] hover:text-[var(--ink)] hover:border-[var(--rule-strong)] transition-colors cursor-pointer"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
        </button>
      </header>

      {/* Main Centered Protected Entry Card */}
      <main className="flex-1 flex items-center justify-center px-6 py-10">
        <div className="w-full max-w-[420px]">
          <div className="mb-8">
            <div className="inline-flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-[var(--signal)] mb-3">
              <Lock className="w-3.5 h-3.5" />
              <span>Protected Workspace</span>
            </div>
            <h1 className="font-display text-3xl font-semibold tracking-tight text-[var(--ink)]">
              Sign in to Nemofy Agent
            </h1>
            <p className="mt-2 text-sm text-[var(--ink-soft)] leading-relaxed">
              Enter your administrator credentials to access autonomous chat, workspace files, and model configuration.
            </p>
          </div>

          <form
            onSubmit={handleSubmit}
            className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-6 sm:p-7 space-y-5 shadow-[0_1px_2px_rgba(22,28,45,0.04)]"
            noValidate
          >
            <div>
              <label
                htmlFor="login-username"
                className="block text-sm font-medium text-[var(--ink)] mb-1.5"
              >
                Username
              </label>
              <input
                id="login-username"
                name="username"
                type="text"
                autoComplete="username"
                autoFocus
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                className="w-full h-11 px-3.5 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-sm text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none focus:border-[var(--signal)] focus:ring-2 focus:ring-[var(--signal)]/20 transition-colors"
              />
            </div>

            <div>
              <label
                htmlFor="login-password"
                className="block text-sm font-medium text-[var(--ink)] mb-1.5"
              >
                Password
              </label>
              <div className="relative">
                <input
                  id="login-password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                  className="w-full h-11 pl-3.5 pr-11 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] text-sm text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none focus:border-[var(--signal)] focus:ring-2 focus:ring-[var(--signal)]/20 transition-colors"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((prev) => !prev)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1.5 rounded-md text-[var(--ink-faint)] hover:text-[var(--ink)] transition-colors cursor-pointer"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {error && (
              <div
                role="alert"
                className="rounded-lg border border-[var(--flare)]/30 bg-[var(--flare)]/10 px-3.5 py-2.5 text-sm text-[var(--flare)]"
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="w-full h-11 inline-flex items-center justify-center gap-2 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-60 text-white text-sm font-semibold transition-colors cursor-pointer"
            >
              <span>{submitting ? 'Verifying...' : 'Sign in'}</span>
              {!submitting && <ArrowRight className="w-4 h-4" />}
            </button>
          </form>

          <p className="mt-6 text-center text-xs text-[var(--ink-faint)]">
            Protected by server-enforced session authentication.
          </p>
        </div>
      </main>
    </div>
  );
}
