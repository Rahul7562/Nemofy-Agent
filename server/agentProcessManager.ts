import { execFile, spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export type AgentCoreLifecycleState = 'idle' | 'bootstrapping' | 'starting' | 'ready' | 'error';

export class AgentCoreProcessManager {
  private child: ChildProcess | null = null;
  private readonly cwd: string;
  private state: AgentCoreLifecycleState = 'idle';
  private lastError: string | null = null;
  private startPromise: Promise<void> | null = null;
  readonly port: number;
  readonly baseUrl: string;

  constructor(cwd?: string, port?: number) {
    this.cwd = path.resolve(cwd || process.cwd());
    this.port = port || Number(process.env.AGENT_API_PORT || '8100');
    this.baseUrl = `http://127.0.0.1:${this.port}`;
  }

  getStatus(): { state: AgentCoreLifecycleState; error: string | null } {
    return { state: this.state, error: this.lastError };
  }

  private get venvPython(): string {
    return path.join(this.cwd, '.venv', 'bin', 'python');
  }

  private async hasRequiredPackages(pythonBin: string): Promise<boolean> {
    if (pythonBin.includes(path.sep) && !fs.existsSync(pythonBin)) {
      return false;
    }
    try {
      await execFileAsync(
        pythonBin,
        ['-c', 'import uvicorn, fastapi, langgraph, langchain_openai, httpx, dotenv'],
        { cwd: this.cwd, timeout: 8000 },
      );
      return true;
    } catch {
      return false;
    }
  }

  private async resolveUvBinary(): Promise<string | null> {
    const candidates = [
      path.join('/tmp', 'nemofy-uv', 'uv'),
      path.join(this.cwd, '.uv', 'uv'),
      '/root/.local/bin/uv',
      'uv',
    ];
    for (const candidate of candidates) {
      try {
        await execFileAsync(candidate, ['--version'], { timeout: 5000 });
        return candidate;
      } catch {
        // try next
      }
    }

    // Install standalone uv binary into /tmp/nemofy-uv (fast, self-contained, zero system pip needed)
    const installDir = path.join('/tmp', 'nemofy-uv');
    fs.mkdirSync(installDir, { recursive: true });
    console.log('[agent-core] Downloading uv to bootstrap Python .venv...');
    await execFileAsync(
      'bash',
      [
        '-lc',
        `curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${installDir}" sh`,
      ],
      { cwd: this.cwd, timeout: 60_000 },
    );

    const installed = path.join(installDir, 'uv');
    if (fs.existsSync(installed)) {
      return installed;
    }
    return null;
  }

  private async ensurePythonEnvironment(): Promise<string> {
    if (await this.hasRequiredPackages(this.venvPython)) {
      return this.venvPython;
    }
    if (await this.hasRequiredPackages('python3')) {
      return 'python3';
    }

    this.state = 'bootstrapping';
    this.lastError = null;
    console.log('[agent-core] Python dependencies missing (cold container) — bootstrapping .venv automatically...');

    const uvBin = await this.resolveUvBinary();
    if (!uvBin) {
      throw new Error('Unable to obtain uv binary for Python environment bootstrap.');
    }

    const venvDir = path.join(this.cwd, '.venv');
    if (!fs.existsSync(this.venvPython)) {
      await execFileAsync(uvBin, ['venv', venvDir, '--python', 'python3'], {
        cwd: this.cwd,
        timeout: 60_000,
      });
    }

    console.log('[agent-core] Installing pure-Python Agent-Core packages into .venv...');
    await execFileAsync(
      uvBin,
      ['pip', 'install', '--python', this.venvPython, '-e', '.'],
      {
        cwd: this.cwd,
        timeout: 240_000,
        maxBuffer: 10 * 1024 * 1024,
      },
    );

    if (!(await this.hasRequiredPackages(this.venvPython))) {
      throw new Error('Bootstrapped .venv is still missing required Agent-Core packages.');
    }

    console.log('[agent-core] Python .venv bootstrap complete.');
    return this.venvPython;
  }

  async isHealthy(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(1500) });
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

  private async waitForHealth(timeoutMs = 20_000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (await this.isHealthy()) return true;
      if (!this.child && this.state === 'error') return false;
      await new Promise((r) => setTimeout(r, 400));
    }
    return false;
  }

  async ensureRunning(waitUpToMs = 0): Promise<boolean> {
    if (await this.isHealthy()) return true;
    if (!this.startPromise) {
      void this.start();
    }
    if (waitUpToMs > 0) {
      return this.waitForHealth(waitUpToMs);
    }
    return false;
  }

  async start(): Promise<void> {
    if (this.startPromise) {
      return this.startPromise;
    }

    this.startPromise = (async () => {
      try {
        if (await this.isHealthy()) {
          console.log(`[agent-core] Connected to running Python Agent-Core at ${this.baseUrl}`);
          return;
        }

        const pythonBin = await this.ensurePythonEnvironment();
        this.state = 'starting';
        this.lastError = null;

        const agentCoreDir = path.join(this.cwd, 'Agent-Core');
        const pythonPath = [agentCoreDir, this.cwd, process.env.PYTHONPATH || '']
          .filter(Boolean)
          .join(path.delimiter);

        console.log(`[agent-core] Launching pure-Python Agent-Core FastAPI on ${this.baseUrl}...`);
        this.child = spawn(
          pythonBin,
          ['-m', 'uvicorn', 'agent_core.api:app', '--host', '127.0.0.1', '--port', String(this.port)],
          {
            cwd: this.cwd,
            env: {
              ...process.env,
              PYTHONPATH: pythonPath,
              AGENT_API_HOST: '127.0.0.1',
              AGENT_API_PORT: String(this.port),
              SKILLS_DIR: path.join(this.cwd, 'Skills'),
              MCP_CONFIG_PATH: path.join(this.cwd, 'MCPs', 'servers.json'),
              WORKSPACE_DIR: path.join(this.cwd, 'workspace'),
              LLM_CONFIG_PATH: path.join(this.cwd, 'data', 'llm_config.json'),
              CHECKPOINT_DB_PATH: path.join(this.cwd, 'data', 'checkpoints.db'),
            },
            stdio: ['ignore', 'pipe', 'pipe'],
          },
        );

        this.child.stdout?.on('data', (chunk: Buffer) => {
          const line = chunk.toString().trim();
          if (line) console.log(`[agent-core] ${line}`);
        });
        this.child.stderr?.on('data', (chunk: Buffer) => {
          const line = chunk.toString().trim();
          if (line) console.log(`[agent-core] ${line}`);
        });
        this.child.on('exit', (code) => {
          console.log(`[agent-core] Python process exited (code=${code ?? 'null'})`);
          this.child = null;
          if (this.state !== 'idle') {
            this.state = 'error';
            this.lastError = `Python Agent-Core exited unexpectedly (code=${code ?? 'null'}).`;
          }
        });

        const ok = await this.waitForHealth(25_000);
        if (ok) {
          this.state = 'ready';
          this.lastError = null;
          console.log(`[agent-core] Python Agent-Core is ready at ${this.baseUrl}`);
        } else {
          this.state = 'error';
          this.lastError = this.lastError || 'Timed out waiting for Python Agent-Core health check.';
        }
      } catch (err) {
        this.state = 'error';
        this.lastError = err instanceof Error ? err.message : String(err);
        console.error(`[agent-core] Startup failed: ${this.lastError}`);
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
      }, 3000);
      proc.once('exit', () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
}
