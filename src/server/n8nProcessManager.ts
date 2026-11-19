import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {
  buildN8nChildEnv,
  type N8nLifecyclePhase,
  type N8nProxyStatus,
  type N8nRuntimeConfig,
} from './n8nConfig.ts';
import { syncOrchestratorWorkflow } from '../orchestrator/workflowSync.ts';

const HEALTH_POLL_INTERVAL_MS = 1000;
const HEALTH_REQUEST_TIMEOUT_MS = 1500;
const AUTO_RESTART_DELAY_MS = 3000;
const MAX_AUTO_RESTARTS = 3;

export class N8nProcessManager {
  private readonly config: N8nRuntimeConfig;
  private readonly cwd: string;
  private child: ChildProcess | null = null;
  private phase: N8nLifecyclePhase = 'starting';
  private startedAt: number | null = null;
  private readyAt: number | null = null;
  private error: string | null = null;
  private attachedExisting = false;
  private stopping = false;
  private restartCount = 0;
  private healthTimer: ReturnType<typeof setInterval> | null = null;
  private restartTimer: ReturnType<typeof setTimeout> | null = null;
  private startPromise: Promise<void> | null = null;
  private postBootSeedChecked = false;

  constructor(config: N8nRuntimeConfig, cwd: string) {
    this.config = config;
    this.cwd = cwd;
  }

  public getStatus(): N8nProxyStatus {
    return {
      phase: this.phase,
      n8nPort: this.config.n8nPort,
      proxyPort: this.config.proxyPort,
      startedAt: this.startedAt,
      readyAt: this.readyAt,
      error: this.error,
      attachedExisting: this.attachedExisting,
    };
  }

  public getPhase(): N8nLifecyclePhase {
    return this.phase;
  }

  public async start(): Promise<void> {
    if (this.startPromise) {
      return this.startPromise;
    }

    this.startPromise = this.bootSequence().finally(() => {
      this.startPromise = null;
    });
    return this.startPromise;
  }

  public async restart(): Promise<void> {
    this.restartCount = 0;
    this.error = null;
    this.attachedExisting = false;
    this.clearTimers();

    if (this.child) {
      const existing = this.child;
      this.child = null;
      await this.terminateProcess(existing);
    }

    this.phase = 'starting';
    this.readyAt = null;
    await this.start();
  }

  public async stop(): Promise<void> {
    this.stopping = true;
    this.clearTimers();
    if (this.child) {
      const existing = this.child;
      this.child = null;
      await this.terminateProcess(existing);
    }
  }

  private clearTimers(): void {
    if (this.healthTimer) {
      clearInterval(this.healthTimer);
      this.healthTimer = null;
    }
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
  }

  private async bootSequence(): Promise<void> {
    this.stopping = false;
    this.startedAt = Date.now();
    this.error = null;

    fs.mkdirSync(this.config.userFolder, { recursive: true });

    const alreadyAlive = await this.probeHealth();
    if (alreadyAlive) {
      this.attachedExisting = true;
      this.phase = 'ready';
      this.readyAt = Date.now();
      console.log(
        `[n8n-proxy] Connected to existing n8n instance on ${this.config.targetOrigin}`,
      );
      this.beginHealthMonitor();
      return;
    }

    let n8nBinary = this.findN8nBinary();
    if (!n8nBinary) {
      this.phase = 'installing';
      console.log('[n8n-proxy] n8n binary not found; installing isolated runtime...');
      try {
        await this.installIsolatedRuntime();
        n8nBinary = this.findN8nBinary();
      } catch (err) {
        this.phase = 'error';
        this.error =
          err instanceof Error
            ? `Failed to install n8n runtime: ${err.message}`
            : 'Failed to install n8n runtime.';
        console.error(`[n8n-proxy] ${this.error}`);
        return;
      }
    }

    if (!n8nBinary) {
      this.phase = 'error';
      this.error = 'n8n binary could not be located after runtime setup.';
      console.error(`[n8n-proxy] ${this.error}`);
      return;
    }

    try {
      const syncResult = syncOrchestratorWorkflow({
        cwd: this.cwd,
        userFolder: this.config.userFolder,
        overwriteExisting: false,
      });
      if (syncResult.imported) {
        console.log(
          `[n8n-proxy] Seeded workflow "${syncResult.workflowName}" (${syncResult.workflowId}, published=${syncResult.published})`,
        );
      }
    } catch (err) {
      console.warn(
        `[n8n-proxy] Skipping automatic workflow seed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    this.spawnN8n(n8nBinary);
  }

  private findN8nBinary(): string | null {
    const candidates = [
      path.resolve(this.cwd, '.n8n-runtime/node_modules/.bin/n8n'),
      path.resolve(this.cwd, 'node_modules/.bin/n8n'),
      '/usr/local/bin/n8n',
      '/usr/bin/n8n',
    ];

    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }
    return null;
  }

  private installIsolatedRuntime(): Promise<void> {
    const runtimeDir = path.resolve(this.cwd, '.n8n-runtime');
    const pkgFile = path.join(runtimeDir, 'package.json');
    if (!fs.existsSync(pkgFile)) {
      fs.mkdirSync(runtimeDir, { recursive: true });
      fs.writeFileSync(
        pkgFile,
        JSON.stringify(
          {
            name: 'n8n-runtime',
            private: true,
            dependencies: { n8n: '2.35.7' },
            trustedDependencies: [
              'sqlite3',
              'better-sqlite3',
              'esbuild',
              'isolated-vm',
              'msgpackr-extract',
              're2',
              'sharp',
            ],
          },
          null,
          2,
        ),
      );
    }

    const useBun = fs.existsSync('/usr/local/bin/bun');
    const cmd = useBun ? '/usr/local/bin/bun' : 'npm';
    const args = useBun
      ? ['install']
      : ['install', '--no-audit', '--no-fund'];

    return new Promise<void>((resolve, reject) => {
      const installer = spawn(cmd, args, {
        cwd: runtimeDir,
        stdio: 'inherit',
      });
      installer.on('error', reject);
      installer.on('exit', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`${cmd} exited with status ${code}`));
        }
      });
    });
  }

  private spawnN8n(binaryPath: string): void {
    this.phase = 'starting';
    this.readyAt = null;
    this.error = null;

    const childEnv = buildN8nChildEnv(process.env, this.config);

    console.log(
      `[n8n-proxy] Launching n8n (${binaryPath}) on ${this.config.targetOrigin}`,
    );

    const child = spawn(binaryPath, ['start'], {
      cwd: this.cwd,
      env: childEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    this.child = child;
    let lastErrorLine = '';

    child.stdout?.on('data', (chunk: Buffer) => {
      const text = chunk.toString('utf8').trimEnd();
      if (text) {
        console.log(`[n8n] ${text}`);
      }
    });

    child.stderr?.on('data', (chunk: Buffer) => {
      const text = chunk.toString('utf8').trimEnd();
      if (text) {
        lastErrorLine = text.split('\n').pop() || text;
        console.error(`[n8n] ${text}`);
      }
    });

    child.on('error', (err) => {
      if (this.child !== child) return;
      this.phase = 'error';
      this.error = `Failed to spawn n8n: ${err.message}`;
      console.error(`[n8n-proxy] ${this.error}`);
    });

    child.on('exit', (code, signal) => {
      if (this.child !== child) return;
      this.child = null;

      if (this.stopping) {
        return;
      }

      const reason = signal
        ? `terminated by signal ${signal}`
        : `exited with code ${code}`;

      if (this.restartCount < MAX_AUTO_RESTARTS) {
        this.restartCount += 1;
        this.phase = 'starting';
        console.warn(
          `[n8n-proxy] n8n ${reason}; restarting (${this.restartCount}/${MAX_AUTO_RESTARTS})...`,
        );
        this.restartTimer = setTimeout(() => {
          this.spawnN8n(binaryPath);
        }, AUTO_RESTART_DELAY_MS);
      } else {
        this.phase = 'error';
        this.error = lastErrorLine
          ? `n8n ${reason}: ${lastErrorLine}`
          : `n8n ${reason}`;
        console.error(`[n8n-proxy] ${this.error}`);
      }
    });

    this.beginHealthMonitor();
  }

  private beginHealthMonitor(): void {
    if (this.healthTimer) {
      clearInterval(this.healthTimer);
    }

    const check = async () => {
      if (this.stopping) return;
      const healthy = await this.probeHealth();
      if (healthy) {
        if (this.phase !== 'ready') {
          if (!this.postBootSeedChecked) {
            this.postBootSeedChecked = true;
            try {
              const postSync = syncOrchestratorWorkflow({
                cwd: this.cwd,
                userFolder: this.config.userFolder,
                overwriteExisting: false,
              });
              if (postSync.imported) {
                console.log(
                  `[n8n-proxy] Post-init seeded workflow "${postSync.workflowName}" (${postSync.workflowId}, published=${postSync.published}); reloading n8n...`,
                );
                void this.restart();
                return;
              }
            } catch (err) {
              console.warn(
                `[n8n-proxy] Post-init workflow check warning: ${err instanceof Error ? err.message : String(err)}`,
              );
            }
          }

          this.phase = 'ready';
          this.readyAt = Date.now();
          this.error = null;
          this.restartCount = 0;
          console.log(
            `[n8n-proxy] n8n is ready on ${this.config.targetOrigin} -> proxied on http://localhost:${this.config.proxyPort}`,
          );
        }
      } else if (this.phase === 'ready' && this.attachedExisting) {
        // Attached external process went down; switch to starting and boot our own
        this.attachedExisting = false;
        this.phase = 'starting';
        this.readyAt = null;
        void this.start();
      }
    };

    void check();
    this.healthTimer = setInterval(() => {
      void check();
    }, HEALTH_POLL_INTERVAL_MS);
  }

  private probeHealth(): Promise<boolean> {
    return new Promise((resolve) => {
      const req = http.get(
        {
          host: this.config.n8nHost,
          port: this.config.n8nPort,
          path: '/healthz/readiness',
          timeout: HEALTH_REQUEST_TIMEOUT_MS,
        },
        (res) => {
          res.resume();
          resolve(
            typeof res.statusCode === 'number' &&
              res.statusCode >= 200 &&
              res.statusCode < 400,
          );
        },
      );
      req.on('error', () => resolve(false));
      req.on('timeout', () => {
        req.destroy();
        resolve(false);
      });
    });
  }

  private terminateProcess(proc: ChildProcess): Promise<void> {
    return new Promise((resolve) => {
      if (proc.exitCode !== null || proc.killed) {
        resolve();
        return;
      }

      const forceKill = setTimeout(() => {
        try {
          proc.kill('SIGKILL');
        } catch {
          // ignore
        }
        resolve();
      }, 5000);

      proc.once('exit', () => {
        clearTimeout(forceKill);
        resolve();
      });

      try {
        proc.kill('SIGTERM');
      } catch {
        clearTimeout(forceKill);
        resolve();
      }
    });
  }
}
