import { execFile, spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export type FirecrawlLifecycleState = 'idle' | 'starting' | 'ready' | 'unavailable' | 'disabled' | 'error';

export function discoverChromeBinary(
  env: Record<string, string | undefined> = process.env,
  cwd: string = process.cwd(),
): string | null {
  const explicit = (env.FIRECRAWL_CHROME_PATH || env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || '').trim();
  if (explicit && fs.existsSync(explicit)) {
    return explicit;
  }

  const staticCandidates = [
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ];
  for (const cand of staticCandidates) {
    if (fs.existsSync(cand)) return cand;
  }

  // Search local project cache and Playwright browser caches (prefer full chromium over chrome-headless-shell)
  const cacheRoots = [
    path.join(cwd, '.firecrawl', 'browsers'),
    '/ms-playwright',
    path.join(os.homedir(), '.cache', 'ms-playwright'),
  ];
  const fullChromeMatches: string[] = [];
  const shellMatches: string[] = [];
  for (const root of cacheRoots) {
    if (!fs.existsSync(root)) continue;
    try {
      for (const entry of fs.readdirSync(root)) {
        const base = path.join(root, entry);
        for (const rel of [
          path.join('chrome-linux', 'chrome'),
          path.join('chrome-linux64', 'chrome'),
          path.join('chrome-mac', 'Chromium.app', 'Contents', 'MacOS', 'Chromium'),
          path.join('chrome-mac-arm64', 'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing'),
          path.join('chrome-mac-x64', 'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing'),
        ]) {
          const full = path.join(base, rel);
          if (fs.existsSync(full)) fullChromeMatches.push(full);
        }
        for (const rel of [
          path.join('chrome-linux', 'headless_shell'),
          path.join('chrome-headless-shell-linux64', 'chrome-headless-shell'),
        ]) {
          const full = path.join(base, rel);
          if (fs.existsSync(full)) shellMatches.push(full);
        }
      }
    } catch {
      // ignore unreadable cache directories
    }
  }
  if (fullChromeMatches.length > 0) return fullChromeMatches.sort().at(-1) || null;
  if (shellMatches.length > 0) return shellMatches.sort().at(-1) || null;
  return null;
}

/**
 * When running as root inside a Linux container (uid === 0), Chromium refuses to start
 * without `--no-sandbox`. Emit a minimal wrapper script under `.firecrawl/bin/` that
 * forwards all arguments from `firecrawl-local` (chromiumoxide) to the discovered Chrome binary.
 * On non-root hosts, returns the Chrome binary directly so Chrome's OS sandbox stays enabled.
 */
export function prepareChromeExecutable(
  cwd: string,
  env: Record<string, string | undefined> = process.env,
): string | null {
  const rawChrome = discoverChromeBinary(env, cwd);
  if (!rawChrome) return null;

  const isRootLinux =
    process.platform === 'linux' && typeof process.getuid === 'function' && process.getuid() === 0;
  if (!isRootLinux) {
    return rawChrome;
  }

  try {
    const binDir = path.join(cwd, '.firecrawl', 'bin');
    fs.mkdirSync(binDir, { recursive: true });
    const wrapperPath = path.join(binDir, 'chrome-launcher.sh');
    const escapedChrome = rawChrome.replace(/'/g, `'\\''`);
    const script = `#!/usr/bin/env bash\nexec '${escapedChrome}' --no-sandbox --disable-dev-shm-usage "$@"\n`;
    fs.writeFileSync(wrapperPath, script, { mode: 0o755 });
    fs.chmodSync(wrapperPath, 0o755);
    return wrapperPath;
  } catch {
    return rawChrome;
  }
}

export class FirecrawlLocalProcessManager {
  private child: ChildProcess | null = null;
  private readonly cwd: string;
  private state: FirecrawlLifecycleState = 'idle';
  private lastError: string | null = null;
  private startPromise: Promise<void> | null = null;
  private readonly getExtraEnv?: () => Record<string, string>;
  readonly host: string;
  readonly port: number;
  readonly baseUrl: string;

  constructor(
    cwd?: string,
    port?: number,
    getExtraEnv?: () => Record<string, string>,
  ) {
    this.cwd = path.resolve(cwd || process.cwd());
    this.host = (process.env.FIRECRAWL_LOCAL_HOST || '127.0.0.1').trim() || '127.0.0.1';
    this.port = port || Number(process.env.FIRECRAWL_PORT || '3002');
    this.baseUrl = `http://${this.host}:${this.port}`;
    this.getExtraEnv = getExtraEnv;
  }

  getStatus(): {
    state: FirecrawlLifecycleState;
    baseUrl: string;
    binaryPath: string | null;
    chromePath: string | null;
    error: string | null;
  } {
    return {
      state: this.state,
      baseUrl: this.baseUrl,
      binaryPath: this.findBinarySync(),
      chromePath: discoverChromeBinary(process.env, this.cwd),
      error: this.lastError,
    };
  }

  findBinarySync(): string | null {
    const candidates = [
      (process.env.FIRECRAWL_LOCAL_BIN || '').trim(),
      path.join(this.cwd, '.firecrawl', 'bin', 'firecrawl-local'),
      path.join(os.homedir(), '.cargo', 'bin', 'firecrawl-local'),
      '/usr/local/cargo/bin/firecrawl-local',
      '/usr/local/bin/firecrawl-local',
      '/usr/bin/firecrawl-local',
    ].filter(Boolean);

    for (const candidate of candidates) {
      try {
        if (fs.existsSync(candidate)) {
          fs.accessSync(candidate, fs.constants.X_OK);
          return candidate;
        }
      } catch {
        // try next
      }
    }
    return null;
  }

  private async resolveBinary(): Promise<string | null> {
    const direct = this.findBinarySync();
    if (direct) return direct;
    try {
      const { stdout } = await execFileAsync('which', ['firecrawl-local'], { timeout: 3000 });
      const resolved = stdout.trim();
      if (resolved && fs.existsSync(resolved)) return resolved;
    } catch {
      // not on PATH
    }
    return null;
  }

  async isHealthy(): Promise<boolean> {
    try {
      const token = (process.env.FIRECRAWL_LOCAL_TOKEN || '').trim();
      const headers: Record<string, string> = {};
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`${this.baseUrl}/health`, {
        headers,
        signal: AbortSignal.timeout(1500),
      });
      if (res.ok) {
        this.state = 'ready';
        this.lastError = null;
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  private async waitForHealth(timeoutMs = 15_000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (await this.isHealthy()) return true;
      if (!this.child && this.state === 'error') return false;
      await new Promise((r) => setTimeout(r, 300));
    }
    return false;
  }

  private shouldAutostart(extraEnv: Record<string, string>): boolean {
    const flag = (process.env.FIRECRAWL_LOCAL_AUTOSTART || 'true').trim().toLowerCase();
    if (flag === 'false' || flag === '0' || flag === 'no' || flag === 'off') {
      return false;
    }
    const configuredUrl = (extraEnv.FIRECRAWL_API_URL || process.env.FIRECRAWL_API_URL || '').trim();
    if (!configuredUrl) return true;
    try {
      const parsed = new URL(configuredUrl);
      const host = parsed.hostname.toLowerCase();
      return host === '127.0.0.1' || host === 'localhost' || host === '::1';
    } catch {
      return true;
    }
  }

  async start(): Promise<void> {
    if (this.startPromise) {
      return this.startPromise;
    }

    this.startPromise = (async () => {
      try {
        const extraEnv = this.getExtraEnv ? this.getExtraEnv() : {};
        if (!this.shouldAutostart(extraEnv)) {
          this.state = 'disabled';
          this.lastError = null;
          return;
        }

        if (await this.isHealthy()) {
          console.log(`[firecrawl-local] Connected to running Firecrawl Local at ${this.baseUrl}`);
          return;
        }

        const bin = await this.resolveBinary();
        if (!bin) {
          this.state = 'unavailable';
          this.lastError =
            'firecrawl-local binary not installed yet. Run `npm run firecrawl:install` (or `cargo install firecrawl-local`).';
          console.log(`[firecrawl-local] ${this.lastError}`);
          return;
        }

        const dataHome = (
          process.env.FIRECRAWL_LOCAL_HOME || path.join(this.cwd, 'data', 'firecrawl-local')
        ).trim();
        fs.mkdirSync(dataHome, { recursive: true });

        const chromePath = prepareChromeExecutable(this.cwd, { ...process.env, ...extraEnv });
        const token = (extraEnv.FIRECRAWL_LOCAL_TOKEN || process.env.FIRECRAWL_LOCAL_TOKEN || '').trim();
        const allowPrivate =
          (process.env.FIRECRAWL_LOCAL_ALLOW_PRIVATE || '').trim().toLowerCase() === 'true';

        const args = ['serve', '--host', this.host, '--port', String(this.port)];
        if (token) {
          args.push('--token', token);
        }
        if (allowPrivate) {
          args.push('--allow-private');
        }

        this.state = 'starting';
        this.lastError = null;
        console.log(
          `[firecrawl-local] Launching official Firecrawl Local (${bin}) on ${this.baseUrl}` +
            (chromePath ? ` with Chrome=${chromePath}` : ' (no Chrome detected; static mode)') +
            '...',
        );

        this.child = spawn(bin, args, {
          cwd: this.cwd,
          env: {
            ...process.env,
            ...extraEnv,
            FIRECRAWL_LOCAL_HOME: dataHome,
            FIRECRAWL_CLOUD: (process.env.FIRECRAWL_CLOUD || 'auto').trim() || 'auto',
            FIRECRAWL_LOCAL_BROWSER: (process.env.FIRECRAWL_LOCAL_BROWSER || 'auto').trim() || 'auto',
            ...(chromePath ? { FIRECRAWL_CHROME_PATH: chromePath } : {}),
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        });

        this.child.stdout?.on('data', (chunk: Buffer) => {
          const line = chunk.toString().trim();
          if (line) console.log(`[firecrawl-local] ${line}`);
        });
        this.child.stderr?.on('data', (chunk: Buffer) => {
          const line = chunk.toString().trim();
          if (line) console.log(`[firecrawl-local] ${line}`);
        });
        this.child.on('exit', (code) => {
          console.log(`[firecrawl-local] Process exited (code=${code ?? 'null'})`);
          this.child = null;
          if (this.state !== 'idle') {
            this.state = 'error';
            this.lastError = `firecrawl-local exited unexpectedly (code=${code ?? 'null'}).`;
          }
        });

        const ok = await this.waitForHealth(15_000);
        if (ok) {
          this.state = 'ready';
          this.lastError = null;
          console.log(`[firecrawl-local] Firecrawl Local v2 API is ready at ${this.baseUrl}`);
        } else {
          this.state = 'error';
          this.lastError = this.lastError || 'Timed out waiting for firecrawl-local /health.';
        }
      } catch (err) {
        this.state = 'error';
        this.lastError = err instanceof Error ? err.message : String(err);
        console.error(`[firecrawl-local] Startup failed: ${this.lastError}`);
      } finally {
        this.startPromise = null;
      }
    })();

    return this.startPromise;
  }

  async stop(): Promise<void> {
    this.state = 'idle';
    if (!this.child) return;
    const proc = this.child;
    this.child = null;
    proc.kill('SIGTERM');
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        try {
          proc.kill('SIGKILL');
        } catch {
          // ignore
        }
        resolve();
      }, 2500);
      proc.once('exit', () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
}
