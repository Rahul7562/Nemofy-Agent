import { useCallback, useEffect, useState } from 'react';
import { ArrowRight, RefreshCw, AlertTriangle } from 'lucide-react';

interface ProxyStatusResponse {
  phase: 'installing' | 'starting' | 'ready' | 'error';
  n8nPort: number;
  proxyPort: number;
  startedAt: number | null;
  readyAt: number | null;
  error: string | null;
  attachedExisting: boolean;
}

const POLL_INTERVAL_MS = 1200;

export default function App() {
  const isBootPreview =
    typeof window !== 'undefined' &&
    window.location.pathname.startsWith('/_n8n_proxy/boot');

  const [status, setStatus] = useState<ProxyStatusResponse>({
    phase: 'starting',
    n8nPort: 5678,
    proxyPort: 3000,
    startedAt: null,
    readyAt: null,
    error: null,
    attachedExisting: false,
  });
  const [restarting, setRestarting] = useState(false);

  const pollStatus = useCallback(async () => {
    try {
      const response = await fetch('/_n8n_proxy/status', {
        cache: 'no-store',
      });
      if (!response.ok) return;
      const data = (await response.json()) as ProxyStatusResponse;
      setStatus(data);

      if (data.phase === 'ready' && !isBootPreview) {
        window.location.reload();
      }
    } catch {
      // Devserver restarting; retry on next poll tick
    }
  }, [isBootPreview]);

  useEffect(() => {
    void pollStatus();
    const timer = window.setInterval(() => {
      void pollStatus();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [pollStatus]);

  const handleRestart = async () => {
    setRestarting(true);
    try {
      await fetch('/_n8n_proxy/restart', { method: 'POST' });
      await pollStatus();
    } finally {
      setRestarting(false);
    }
  };

  const handleOpenWorkspace = () => {
    window.location.assign('/');
  };

  const headingText =
    status.phase === 'installing'
      ? 'Installing n8n runtime'
      : status.phase === 'ready'
        ? 'n8n is connected'
        : status.phase === 'error'
          ? 'n8n failed to start'
          : 'Launching n8n';

  const subtitleText =
    status.phase === 'installing'
      ? 'Preparing the isolated n8n runtime packages before starting the workflow engine.'
      : status.phase === 'ready'
        ? 'The n8n web server is live and bridged directly to this devserver port.'
        : status.phase === 'error'
          ? status.error || 'The n8n process stopped unexpectedly before becoming ready.'
          : 'Initializing database migrations and connecting the n8n web interface.';

  return (
    <main className="min-h-screen bg-[#141418] text-[#F5F5F7] flex items-center justify-center px-6 py-12 antialiased">
      <section className="w-full max-w-md rounded-2xl border border-white/10 bg-[#1B1B21] p-8 shadow-2xl">
        <div className="flex items-center gap-3 mb-7">
          <div
            aria-hidden="true"
            className="h-10 w-10 rounded-xl bg-[#FF6D5A] flex items-center justify-center text-white font-bold tracking-tight text-sm shadow-sm"
          >
            n8n
          </div>
          <div>
            <p className="text-xs font-medium uppercase tracking-widest text-[#A1A1AA]">
              Workflow Automation
            </p>
            <p className="font-mono text-xs text-[#D4D4D8] mt-0.5">
              127.0.0.1:{status.n8nPort} → :{status.proxyPort}
            </p>
          </div>
        </div>

        <h1 className="text-2xl font-semibold tracking-tight text-white">
          {headingText}
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#A1A1AA]">
          {subtitleText}
        </p>

        {(status.phase === 'starting' || status.phase === 'installing') && (
          <div className="mt-7" role="status" aria-live="polite">
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
              <div className="h-full w-2/5 rounded-full bg-[#FF6D5A] animate-pulse" />
            </div>
            <p className="mt-3 text-xs text-[#A1A1AA]">
              You will enter the n8n editor automatically once startup completes.
            </p>
          </div>
        )}

        {status.phase === 'error' && (
          <div className="mt-7 space-y-4">
            <div
              role="alert"
              className="flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 p-3.5 text-xs text-red-200"
            >
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-red-400" />
              <span className="break-words">
                {status.error || 'Unable to reach n8n on its internal port.'}
              </span>
            </div>
            <button
              type="button"
              onClick={handleRestart}
              disabled={restarting}
              className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-[#FF6D5A] px-4 py-2.5 text-sm font-medium text-white transition hover:bg-[#ff5a45] disabled:opacity-60 cursor-pointer"
            >
              <RefreshCw className={`h-4 w-4 ${restarting ? 'animate-spin' : ''}`} />
              {restarting ? 'Restarting n8n…' : 'Retry launch'}
            </button>
          </div>
        )}

        {status.phase === 'ready' && (
          <div className="mt-7 flex items-center gap-3">
            <button
              type="button"
              onClick={handleOpenWorkspace}
              className="flex-1 inline-flex items-center justify-center gap-2 rounded-xl bg-[#FF6D5A] px-4 py-2.5 text-sm font-medium text-white transition hover:bg-[#ff5a45] cursor-pointer"
            >
              Open n8n Editor
              <ArrowRight className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={handleRestart}
              disabled={restarting}
              aria-label="Restart n8n engine"
              className="inline-flex items-center justify-center rounded-xl border border-white/15 bg-white/5 px-3.5 py-2.5 text-sm font-medium text-[#E4E4E7] transition hover:bg-white/10 disabled:opacity-60 cursor-pointer"
            >
              <RefreshCw className={`h-4 w-4 ${restarting ? 'animate-spin' : ''}`} />
            </button>
          </div>
        )}
      </section>
    </main>
  );
}
