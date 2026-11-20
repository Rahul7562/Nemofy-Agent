import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

export class AgentCoreProcessManager {
  private child: ChildProcess | null = null;
  private readonly cwd: string;
  readonly port: number;
  readonly baseUrl: string;

  constructor(cwd?: string, port?: number) {
    this.cwd = path.resolve(cwd || process.cwd());
    this.port = port || Number(process.env.AGENT_API_PORT || '8100');
    this.baseUrl = `http://127.0.0.1:${this.port}`;
  }

  private resolvePythonBin(): string {
    const venvPy = path.join(this.cwd, '.venv', 'bin', 'python');
    if (fs.existsSync(venvPy)) return venvPy;
    return 'python3';
  }

  async isHealthy(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(1500) });
      return res.ok;
    } catch {
      return false;
    }
  }

  async start(): Promise<void> {
    if (await this.isHealthy()) {
      console.log(`[agent-core] Connected to running Python Agent-Core at ${this.baseUrl}`);
      return;
    }

    const pythonBin = this.resolvePythonBin();
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
    });
  }

  async stop(): Promise<void> {
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
