import path from 'node:path';

export interface N8nRuntimeConfig {
  proxyPort: number;
  n8nHost: string;
  n8nPort: number;
  userFolder: string;
  publicBaseUrl: string;
  targetOrigin: string;
}

export type N8nLifecyclePhase =
  | 'installing'
  | 'starting'
  | 'ready'
  | 'error';

export interface N8nProxyStatus {
  phase: N8nLifecyclePhase;
  n8nPort: number;
  proxyPort: number;
  startedAt: number | null;
  readyAt: number | null;
  error: string | null;
  attachedExisting: boolean;
}

const DEFAULT_PROXY_PORT = 3000;
const DEFAULT_N8N_PORT = 5678;
const DEFAULT_N8N_HOST = '127.0.0.1';

function parsePort(raw: string | undefined, fallback: number): number {
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 65535) {
    return fallback;
  }
  return parsed;
}

export function normalizePublicBaseUrl(
  appUrl: string | undefined,
  proxyPort: number,
): string {
  const trimmed = appUrl?.trim();
  if (
    !trimmed ||
    trimmed === 'MY_APP_URL' ||
    trimmed === '"MY_APP_URL"' ||
    !/^https?:\/\//i.test(trimmed)
  ) {
    return `http://localhost:${proxyPort}/`;
  }
  return trimmed.endsWith('/') ? trimmed : `${trimmed}/`;
}

export function resolveN8nRuntimeConfig(
  env: Record<string, string | undefined>,
  cwd: string,
): N8nRuntimeConfig {
  const proxyPort = parsePort(env.PORT, DEFAULT_PROXY_PORT);
  const n8nPort = parsePort(env.N8N_PORT, DEFAULT_N8N_PORT);
  const n8nHost = env.N8N_HOST?.trim() || DEFAULT_N8N_HOST;
  const rawUserFolder = env.N8N_USER_FOLDER?.trim() || './.n8n-data';
  const userFolder = path.isAbsolute(rawUserFolder)
    ? rawUserFolder
    : path.resolve(cwd, rawUserFolder);
  const publicBaseUrl = normalizePublicBaseUrl(env.APP_URL, proxyPort);
  const targetOrigin = `http://${n8nHost}:${n8nPort}`;

  return {
    proxyPort,
    n8nHost,
    n8nPort,
    userFolder,
    publicBaseUrl,
    targetOrigin,
  };
}

export function buildN8nChildEnv(
  baseEnv: Record<string, string | undefined>,
  config: N8nRuntimeConfig,
): Record<string, string> {
  const childEnv: Record<string, string> = {};
  for (const [key, value] of Object.entries(baseEnv)) {
    if (typeof value === 'string') {
      childEnv[key] = value;
    }
  }

  const parsedPublic = new URL(config.publicBaseUrl);
  const publicProtocol = parsedPublic.protocol.replace(':', '');

  childEnv.N8N_HOST = config.n8nHost;
  childEnv.N8N_PORT = String(config.n8nPort);
  childEnv.N8N_LISTEN_ADDRESS = config.n8nHost;
  childEnv.N8N_PROTOCOL = 'http';
  childEnv.N8N_USER_FOLDER = config.userFolder;
  childEnv.N8N_EDITOR_BASE_URL = config.publicBaseUrl;
  childEnv.WEBHOOK_URL = config.publicBaseUrl;
  childEnv.N8N_PROXY_HOPS = baseEnv.N8N_PROXY_HOPS || '1';
  childEnv.N8N_SECURE_COOKIE = 'false';
  childEnv.N8N_SAMESITE_COOKIE = publicProtocol === 'https' ? 'none' : 'lax';
  childEnv.N8N_DIAGNOSTICS_ENABLED = 'false';
  childEnv.N8N_VERSION_NOTIFICATIONS_ENABLED = 'false';
  childEnv.N8N_PERSONALIZATION_ENABLED = 'false';
  childEnv.N8N_HIRING_BANNER_ENABLED = 'false';
  childEnv.N8N_TEMPLATES_ENABLED = baseEnv.N8N_TEMPLATES_ENABLED || 'true';
  childEnv.N8N_PUSH_BACKEND = baseEnv.N8N_PUSH_BACKEND || 'websocket';
  childEnv.N8N_DEFAULT_BINARY_DATA_MODE =
    baseEnv.N8N_DEFAULT_BINARY_DATA_MODE || 'filesystem';

  return childEnv;
}

export function isForwardedHttps(
  headers: Record<string, string | string[] | undefined>,
): boolean {
  const rawProto = headers['x-forwarded-proto'] ?? headers['x-forwarded-scheme'];
  if (rawProto) {
    const value = Array.isArray(rawProto) ? rawProto[0] : rawProto;
    if (value.split(',')[0].trim().toLowerCase() === 'https') {
      return true;
    }
  }

  const rawForwarded = headers.forwarded;
  if (rawForwarded) {
    const value = Array.isArray(rawForwarded) ? rawForwarded[0] : rawForwarded;
    if (/proto=https\b/i.test(value)) {
      return true;
    }
  }

  return false;
}

function isLocalHostname(hostHeader: string): boolean {
  const normalized = hostHeader.trim().toLowerCase();
  return (
    normalized.startsWith('localhost') ||
    normalized.startsWith('127.0.0.1') ||
    normalized.startsWith('[::1]') ||
    normalized.startsWith('0.0.0.0')
  );
}

export function isHttpsProxyRequest(
  headers: Record<string, string | string[] | undefined>,
  publicBaseUrl?: string,
): boolean {
  if (isForwardedHttps(headers)) {
    return true;
  }

  const origin = headers.origin;
  const originStr = Array.isArray(origin) ? origin[0] : origin;
  if (originStr && originStr.toLowerCase().startsWith('https://')) {
    return true;
  }

  const referer = headers.referer;
  const refererStr = Array.isArray(referer) ? referer[0] : referer;
  if (refererStr && refererStr.toLowerCase().startsWith('https://')) {
    return true;
  }

  const host = headers['x-forwarded-host'] ?? headers.host;
  const hostStr = Array.isArray(host) ? host[0] : host;
  if (
    hostStr &&
    !isLocalHostname(hostStr) &&
    publicBaseUrl?.toLowerCase().startsWith('https://')
  ) {
    return true;
  }

  return false;
}

export function rewriteSetCookieValue(
  cookieHeader: string,
  httpsRequest: boolean,
): string {
  const parts = cookieHeader
    .split(';')
    .map((segment) => segment.trim())
    .filter(Boolean);

  if (parts.length === 0) return cookieHeader;

  const [nameValue, ...rawAttributes] = parts;
  const filteredAttributes = rawAttributes.filter((attr) => {
    const lower = attr.toLowerCase();
    return (
      lower !== 'secure' &&
      lower !== 'partitioned' &&
      !lower.startsWith('samesite=')
    );
  });

  if (httpsRequest) {
    filteredAttributes.push('SameSite=None');
    filteredAttributes.push('Secure');
    filteredAttributes.push('Partitioned');
  } else {
    filteredAttributes.push('SameSite=Lax');
  }

  return [nameValue, ...filteredAttributes].join('; ');
}

export function stripFrameAncestorsFromCsp(csp: string): string {
  return csp
    .split(';')
    .map((directive) => directive.trim())
    .filter((directive) => {
      if (!directive) return false;
      return !directive.toLowerCase().startsWith('frame-ancestors');
    })
    .join('; ');
}

export function sanitizeProxyResponseHeaders(
  headers: Record<string, string | string[] | undefined>,
  options: {
    httpsRequest: boolean;
    targetOrigin: string;
  },
): Record<string, string | string[] | undefined> {
  const next: Record<string, string | string[] | undefined> = { ...headers };

  delete next['x-frame-options'];
  delete next['cross-origin-resource-policy'];

  if (typeof next['content-security-policy'] === 'string') {
    const cleaned = stripFrameAncestorsFromCsp(next['content-security-policy']);
    if (cleaned) {
      next['content-security-policy'] = cleaned;
    } else {
      delete next['content-security-policy'];
    }
  }

  if (typeof next.location === 'string') {
    if (next.location.startsWith(options.targetOrigin)) {
      const stripped = next.location.slice(options.targetOrigin.length);
      next.location = stripped.startsWith('/') ? stripped : `/${stripped}`;
    }
  }

  const rawCookies = next['set-cookie'];
  if (typeof rawCookies === 'string') {
    next['set-cookie'] = rewriteSetCookieValue(rawCookies, options.httpsRequest);
  } else if (Array.isArray(rawCookies)) {
    next['set-cookie'] = rawCookies.map((cookie) =>
      rewriteSetCookieValue(cookie, options.httpsRequest),
    );
  }

  return next;
}

export function isInternalControlPath(urlPath: string): boolean {
  const pathname = urlPath.split('?')[0];
  return (
    pathname === '/_n8n_proxy/status' ||
    pathname === '/_n8n_proxy/restart' ||
    pathname === '/_n8n_proxy/boot' ||
    pathname.startsWith('/_n8n_proxy/boot/')
  );
}

export function shouldProxyRequestToN8n(
  urlPath: string,
  phase: N8nLifecyclePhase,
): boolean {
  if (isInternalControlPath(urlPath)) {
    return false;
  }
  return phase === 'ready';
}
