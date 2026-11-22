export interface ServerListenConfig {
  port: number;
  host: string;
}

function parseValidPort(raw: string | undefined): number | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const parsed = Number.parseInt(trimmed, 10);
  if (!Number.isFinite(parsed) || parsed < 1 || parsed > 65535) return null;
  return parsed;
}

function parseCliFlag(argv: readonly string[], longFlag: string, shortFlag?: string): string | undefined {
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === longFlag || (shortFlag && arg === shortFlag)) {
      const next = argv[i + 1];
      if (next && !next.startsWith('-')) {
        return next;
      }
    } else if (arg.startsWith(`${longFlag}=`)) {
      return arg.slice(longFlag.length + 1);
    }
  }
  return undefined;
}

/**
 * Resolves the HTTP listen host and port for server.ts.
 *
 * Precedence:
 * 1. Explicit CLI flags (`--port`, `-p`, `--host`), which platforms such as
 *    AI Studio pass via `npm run dev --port 3000 --host 0.0.0.0` even when a
 *    container-level `PORT=8080` env var is present for the reverse proxy.
 * 2. Environment variables (`PORT`, `HOST`).
 * 3. Defaults (`3000`, `'0.0.0.0'`).
 */
export function resolveServerListenConfig(
  argv: readonly string[] = process.argv.slice(2),
  env: NodeJS.ProcessEnv = process.env,
): ServerListenConfig {
  const cliPort = parseValidPort(parseCliFlag(argv, '--port', '-p'));
  const envPort = parseValidPort(env.PORT);
  const port = cliPort ?? envPort ?? 3000;

  const cliHost = parseCliFlag(argv, '--host', '-H')?.trim();
  const envHost = env.HOST?.trim();
  const host = cliHost || envHost || '0.0.0.0';

  return { port, host };
}
