export const N8N_AUTH_COOKIE_NAME = 'n8n-auth';

export type AuthCookieUpdate =
  | { kind: 'set'; token: string; maxAgeMs: number }
  | { kind: 'clear' };

interface StoredAuthSession {
  token: string;
  expiresAt: number;
  fingerprint: string;
}

const DEFAULT_COOKIE_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const MAX_STORED_SESSIONS = 256;

export function getHeaderString(
  headers: Record<string, string | string[] | undefined>,
  name: string,
): string | null {
  const raw = headers[name.toLowerCase()];
  if (!raw) return null;
  const value = Array.isArray(raw) ? raw[0] : raw;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function hasAuthCookie(cookieHeader: string | undefined): boolean {
  if (!cookieHeader) return false;
  return cookieHeader
    .split(';')
    .some((part) => {
      const trimmed = part.trim();
      if (!trimmed.startsWith(`${N8N_AUTH_COOKIE_NAME}=`)) return false;
      const value = trimmed.slice(N8N_AUTH_COOKIE_NAME.length + 1).trim();
      return value.length > 0;
    });
}

export function appendAuthCookieHeader(
  existingCookieHeader: string | undefined,
  token: string,
): string {
  const authPair = `${N8N_AUTH_COOKIE_NAME}=${token}`;
  if (!existingCookieHeader || existingCookieHeader.trim().length === 0) {
    return authPair;
  }
  const filtered = existingCookieHeader
    .split(';')
    .map((part) => part.trim())
    .filter((part) => part.length > 0 && !part.startsWith(`${N8N_AUTH_COOKIE_NAME}=`));
  filtered.push(authPair);
  return filtered.join('; ');
}

export function extractAuthCookieUpdate(
  setCookieHeader: string | string[] | undefined,
  nowMs: number = Date.now(),
): AuthCookieUpdate | null {
  if (!setCookieHeader) return null;
  const cookies = Array.isArray(setCookieHeader) ? setCookieHeader : [setCookieHeader];

  let result: AuthCookieUpdate | null = null;
  for (const cookie of cookies) {
    const segments = cookie
      .split(';')
      .map((part) => part.trim())
      .filter(Boolean);
    if (segments.length === 0) continue;

    const [nameValue, ...attrs] = segments;
    const eqIndex = nameValue.indexOf('=');
    if (eqIndex <= 0) continue;

    const name = nameValue.slice(0, eqIndex).trim();
    if (name !== N8N_AUTH_COOKIE_NAME) continue;

    const value = nameValue.slice(eqIndex + 1).trim();
    if (!value) {
      result = { kind: 'clear' };
      continue;
    }

    let maxAgeMs = DEFAULT_COOKIE_TTL_MS;
    let cleared = false;

    for (const attr of attrs) {
      const lower = attr.toLowerCase();
      if (lower.startsWith('max-age=')) {
        const seconds = Number.parseInt(attr.slice('max-age='.length).trim(), 10);
        if (Number.isFinite(seconds)) {
          if (seconds <= 0) {
            cleared = true;
          } else {
            maxAgeMs = seconds * 1000;
          }
        }
      } else if (lower.startsWith('expires=')) {
        const parsedTime = Date.parse(attr.slice('expires='.length).trim());
        if (Number.isFinite(parsedTime)) {
          if (parsedTime <= nowMs) {
            cleared = true;
          } else {
            maxAgeMs = parsedTime - nowMs;
          }
        }
      }
    }

    result = cleared ? { kind: 'clear' } : { kind: 'set', token: value, maxAgeMs };
  }

  return result;
}

export function buildClientFingerprint(
  headers: Record<string, string | string[] | undefined>,
  remoteAddress?: string,
): string {
  const ua = getHeaderString(headers, 'user-agent') || 'unknown-ua';
  const forwardedFor = getHeaderString(headers, 'x-forwarded-for');
  const clientIp = forwardedFor
    ? forwardedFor.split(',')[0].trim()
    : remoteAddress || 'local';
  return `${clientIp}|${ua}`;
}

export class AuthCookieBridge {
  private readonly byBrowserId = new Map<string, StoredAuthSession>();
  private readonly byFingerprint = new Map<string, { token: string; expiresAt: number }>();

  public recordResponseCookies(
    setCookieHeader: string | string[] | undefined,
    reqHeaders: Record<string, string | string[] | undefined>,
    remoteAddress?: string,
    nowMs: number = Date.now(),
  ): void {
    const update = extractAuthCookieUpdate(setCookieHeader, nowMs);
    if (!update) return;

    const browserId = getHeaderString(reqHeaders, 'browser-id');
    const fingerprint = buildClientFingerprint(reqHeaders, remoteAddress);

    if (update.kind === 'clear') {
      if (browserId) {
        this.byBrowserId.delete(browserId);
      }
      this.byFingerprint.delete(fingerprint);
      return;
    }

    const expiresAt = nowMs + update.maxAgeMs;
    if (browserId) {
      this.byBrowserId.set(browserId, {
        token: update.token,
        expiresAt,
        fingerprint,
      });
    }
    this.byFingerprint.set(fingerprint, {
      token: update.token,
      expiresAt,
    });

    this.prune(nowMs);
  }

  public resolveFallbackTokenForHttp(
    reqHeaders: Record<string, string | string[] | undefined>,
    nowMs: number = Date.now(),
  ): string | null {
    const cookieHeader = getHeaderString(reqHeaders, 'cookie') ?? undefined;
    if (hasAuthCookie(cookieHeader)) {
      return null;
    }

    const browserId = getHeaderString(reqHeaders, 'browser-id');
    if (!browserId) {
      return null;
    }

    const session = this.byBrowserId.get(browserId);
    if (!session) return null;
    if (session.expiresAt <= nowMs) {
      this.byBrowserId.delete(browserId);
      return null;
    }
    return session.token;
  }

  public resolveFallbackTokenForWebSocket(
    reqHeaders: Record<string, string | string[] | undefined>,
    remoteAddress?: string,
    nowMs: number = Date.now(),
  ): string | null {
    const cookieHeader = getHeaderString(reqHeaders, 'cookie') ?? undefined;
    if (hasAuthCookie(cookieHeader)) {
      return null;
    }

    const browserId = getHeaderString(reqHeaders, 'browser-id');
    if (browserId) {
      const byId = this.byBrowserId.get(browserId);
      if (byId && byId.expiresAt > nowMs) {
        return byId.token;
      }
    }

    const fingerprint = buildClientFingerprint(reqHeaders, remoteAddress);
    const entry = this.byFingerprint.get(fingerprint);
    if (!entry) return null;
    if (entry.expiresAt <= nowMs) {
      this.byFingerprint.delete(fingerprint);
      return null;
    }
    return entry.token;
  }

  private prune(nowMs: number): void {
    for (const [key, session] of this.byBrowserId.entries()) {
      if (session.expiresAt <= nowMs) {
        this.byBrowserId.delete(key);
      }
    }
    for (const [key, entry] of this.byFingerprint.entries()) {
      if (entry.expiresAt <= nowMs) {
        this.byFingerprint.delete(key);
      }
    }

    while (this.byBrowserId.size > MAX_STORED_SESSIONS) {
      const oldestKey = this.byBrowserId.keys().next().value;
      if (!oldestKey) break;
      this.byBrowserId.delete(oldestKey);
    }
    while (this.byFingerprint.size > MAX_STORED_SESSIONS) {
      const oldestKey = this.byFingerprint.keys().next().value;
      if (!oldestKey) break;
      this.byFingerprint.delete(oldestKey);
    }
  }
}
