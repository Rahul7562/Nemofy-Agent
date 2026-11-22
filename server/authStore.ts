import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

export interface StoredAuthRecord {
  username: string;
  passwordSalt: string;
  passwordHash: string;
  sessionSecret: string;
  vaultMasterSecret: string;
  agentApiToken?: string;
  updatedAt: string;
  setupComplete?: boolean;
  revokedTokens?: Record<string, number>;
}

export interface SessionPayload {
  username: string;
  issuedAt: number;
  expiresAt: number;
  version: string;
  jti?: string;
}

const DEFAULT_USERNAME = 'admin';
const SESSION_TTL_MS = 24 * 60 * 60 * 1000;

function getEnvAdminPassword(): string {
  return (process.env.NEMOFY_ADMIN_PASSWORD || '').trim();
}

function hashPassword(password: string, salt: string): string {
  return crypto.scryptSync(password, salt, 64).toString('hex');
}

function timingSafeEqualHex(a: string, b: string): boolean {
  const bufA = Buffer.from(a, 'hex');
  const bufB = Buffer.from(b, 'hex');
  if (bufA.length !== bufB.length) return false;
  return crypto.timingSafeEqual(bufA, bufB);
}

function timingSafeEqualUtf8(a: string, b: string): boolean {
  const bufA = Buffer.from(a, 'utf-8');
  const bufB = Buffer.from(b, 'utf-8');
  if (bufA.length !== bufB.length) return false;
  return crypto.timingSafeEqual(bufA, bufB);
}

function pruneRevokedTokens(tokens: Record<string, number> | undefined, now = Date.now()): Record<string, number> {
  const cleaned: Record<string, number> = {};
  if (!tokens || typeof tokens !== 'object') return cleaned;
  for (const [jti, exp] of Object.entries(tokens)) {
    if (typeof exp === 'number' && exp > now) {
      cleaned[jti] = exp;
    }
  }
  return cleaned;
}

export class AuthStore {
  private readonly filePath: string;
  private record: StoredAuthRecord;

  constructor(filePath?: string) {
    this.filePath = path.resolve(filePath || path.join(process.cwd(), 'data', 'auth.json'));
    this.record = this.loadOrSeed();
  }

  private loadOrSeed(): StoredAuthRecord {
    const envPassword = getEnvAdminPassword();
    if (fs.existsSync(this.filePath)) {
      try {
        const parsed = JSON.parse(fs.readFileSync(this.filePath, 'utf-8')) as Partial<StoredAuthRecord>;
        if (
          parsed &&
          typeof parsed.username === 'string' &&
          typeof parsed.passwordSalt === 'string' &&
          typeof parsed.passwordHash === 'string' &&
          typeof parsed.sessionSecret === 'string'
        ) {
          const vaultMasterSecret =
            typeof parsed.vaultMasterSecret === 'string' && parsed.vaultMasterSecret.length >= 32
              ? parsed.vaultMasterSecret
              : crypto.randomBytes(32).toString('hex');
          const agentApiToken =
            typeof parsed.agentApiToken === 'string' && parsed.agentApiToken.length >= 32
              ? parsed.agentApiToken
              : crypto.randomBytes(32).toString('hex');
          const hasStoredHash = Boolean(parsed.passwordSalt && parsed.passwordHash);
          const setupComplete =
            typeof parsed.setupComplete === 'boolean' ? parsed.setupComplete : hasStoredHash;
          const loaded: StoredAuthRecord = {
            username: parsed.username,
            passwordSalt: parsed.passwordSalt,
            passwordHash: parsed.passwordHash,
            sessionSecret: parsed.sessionSecret,
            vaultMasterSecret,
            agentApiToken,
            updatedAt: parsed.updatedAt || new Date().toISOString(),
            setupComplete,
            revokedTokens: pruneRevokedTokens(parsed.revokedTokens),
          };
          if (!parsed.vaultMasterSecret || !parsed.agentApiToken) {
            this.save(loaded);
          }
          return loaded;
        }
      } catch {
        // Re-seed if corrupted
      }
    }

    const salt = envPassword ? crypto.randomBytes(16).toString('hex') : '';
    const seeded: StoredAuthRecord = {
      username: DEFAULT_USERNAME,
      passwordSalt: salt,
      passwordHash: envPassword ? hashPassword(envPassword, salt) : '',
      sessionSecret: crypto.randomBytes(32).toString('hex'),
      vaultMasterSecret: crypto.randomBytes(32).toString('hex'),
      agentApiToken: crypto.randomBytes(32).toString('hex'),
      updatedAt: new Date().toISOString(),
      setupComplete: Boolean(envPassword),
      revokedTokens: {},
    };
    this.save(seeded);
    return seeded;
  }

  private save(rec: StoredAuthRecord): void {
    rec.revokedTokens = pruneRevokedTokens(rec.revokedTokens);
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(rec, null, 2) + '\n', {
      encoding: 'utf-8',
      mode: 0o600,
    });
  }

  needsSetup(): boolean {
    if (getEnvAdminPassword()) return false;
    return !this.record.setupComplete || !this.record.passwordHash || !this.record.passwordSalt;
  }

  setupInitialPassword(
    newPassword: string,
    username?: string,
  ): { ok: boolean; error?: string; token?: string; username?: string } {
    if (!this.needsSetup()) {
      return { ok: false, error: 'Administrator account is already configured.' };
    }
    if (typeof newPassword !== 'string' || newPassword.length < 8) {
      return { ok: false, error: 'Password must be at least 8 characters long.' };
    }
    const cleanUser = (username || DEFAULT_USERNAME).trim() || DEFAULT_USERNAME;
    if (!/^[a-zA-Z0-9._-]{2,32}$/.test(cleanUser)) {
      return { ok: false, error: 'Username must be 2–32 characters (letters, numbers, ., _, -).' };
    }

    const newSalt = crypto.randomBytes(16).toString('hex');
    this.record = {
      username: cleanUser,
      passwordSalt: newSalt,
      passwordHash: hashPassword(newPassword, newSalt),
      sessionSecret: crypto.randomBytes(32).toString('hex'),
      vaultMasterSecret: this.record.vaultMasterSecret,
      agentApiToken: this.record.agentApiToken || crypto.randomBytes(32).toString('hex'),
      updatedAt: new Date().toISOString(),
      setupComplete: true,
      revokedTokens: {},
    };
    this.save(this.record);
    return { ok: true, username: this.record.username, token: this.createSessionToken() };
  }

  getUsername(): string {
    return this.record.username;
  }

  getUpdatedAt(): string {
    return this.record.updatedAt;
  }

  getVaultKeyMaterial(): string {
    const envMaster = (process.env.VAULT_MASTER_KEY || '').trim();
    if (envMaster.length >= 16) return envMaster;
    return this.record.vaultMasterSecret;
  }

  getAgentApiToken(): string {
    const envToken = (process.env.AGENT_API_TOKEN || '').trim();
    if (envToken) return envToken;
    if (this.record.agentApiToken && this.record.agentApiToken.length >= 32) {
      return this.record.agentApiToken;
    }
    const nextToken = crypto.randomBytes(32).toString('hex');
    this.record = {
      ...this.record,
      agentApiToken: nextToken,
    };
    this.save(this.record);
    return nextToken;
  }

  verifyCredentials(username: string, password: string): boolean {
    const cleanUser = username.trim();
    if (!cleanUser || !password) return false;
    if (cleanUser !== this.record.username) return false;

    // A configured NEMOFY_ADMIN_PASSWORD env secret always overrides any claimed fallback hash
    const envPassword = getEnvAdminPassword();
    if (envPassword) {
      return timingSafeEqualUtf8(password, envPassword);
    }

    if (this.needsSetup()) return false;
    const candidateHash = hashPassword(password, this.record.passwordSalt);
    return timingSafeEqualHex(candidateHash, this.record.passwordHash);
  }

  createSessionToken(): string {
    const now = Date.now();
    const payload: SessionPayload = {
      username: this.record.username,
      issuedAt: now,
      expiresAt: now + SESSION_TTL_MS,
      version: this.record.updatedAt,
      jti: crypto.randomBytes(16).toString('hex'),
    };
    const encoded = Buffer.from(JSON.stringify(payload), 'utf-8').toString('base64url');
    const sig = crypto
      .createHmac('sha256', this.record.sessionSecret)
      .update(encoded)
      .digest('base64url');
    return `${encoded}.${sig}`;
  }

  verifySessionToken(token: string | undefined | null): SessionPayload | null {
    if (!token || typeof token !== 'string') return null;
    const dot = token.indexOf('.');
    if (dot <= 0) return null;
    const encoded = token.slice(0, dot);
    const providedSig = token.slice(dot + 1);
    if (!encoded || !providedSig) return null;

    const expectedSig = crypto
      .createHmac('sha256', this.record.sessionSecret)
      .update(encoded)
      .digest('base64url');

    const bufA = Buffer.from(providedSig, 'utf-8');
    const bufB = Buffer.from(expectedSig, 'utf-8');
    if (bufA.length !== bufB.length || !crypto.timingSafeEqual(bufA, bufB)) {
      return null;
    }

    try {
      const payload = JSON.parse(
        Buffer.from(encoded, 'base64url').toString('utf-8'),
      ) as SessionPayload;
      if (!payload || payload.username !== this.record.username) return null;
      if (typeof payload.expiresAt !== 'number' || payload.expiresAt <= Date.now()) return null;
      if (payload.version !== this.record.updatedAt) return null;
      if (payload.jti && this.record.revokedTokens && this.record.revokedTokens[payload.jti]) {
        return null;
      }
      return payload;
    } catch {
      return null;
    }
  }

  revokeSessionToken(token: string | undefined | null): boolean {
    const payload = this.verifySessionToken(token);
    if (!payload) return false;
    const revoked = pruneRevokedTokens(this.record.revokedTokens);
    if (payload.jti) {
      revoked[payload.jti] = payload.expiresAt;
      this.record = {
        ...this.record,
        revokedTokens: revoked,
      };
    } else {
      // Legacy token without jti: rotate sessionSecret to invalidate immediately
      this.record = {
        ...this.record,
        sessionSecret: crypto.randomBytes(32).toString('hex'),
        updatedAt: new Date().toISOString(),
        revokedTokens: revoked,
      };
    }
    this.save(this.record);
    return true;
  }

  changePassword(currentPassword: string, newPassword: string): { ok: boolean; error?: string; token?: string } {
    if (getEnvAdminPassword()) {
      return {
        ok: false,
        error: 'Password is managed by the NEMOFY_ADMIN_PASSWORD environment variable.',
      };
    }
    if (!this.verifyCredentials(this.record.username, currentPassword)) {
      return { ok: false, error: 'Current password is incorrect.' };
    }
    if (newPassword.length < 8) {
      return { ok: false, error: 'New password must be at least 8 characters long.' };
    }
    if (newPassword === currentPassword) {
      return { ok: false, error: 'New password must differ from the current password.' };
    }

    const newSalt = crypto.randomBytes(16).toString('hex');
    this.record = {
      username: this.record.username,
      passwordSalt: newSalt,
      passwordHash: hashPassword(newPassword, newSalt),
      sessionSecret: crypto.randomBytes(32).toString('hex'),
      vaultMasterSecret: this.record.vaultMasterSecret,
      agentApiToken: this.record.agentApiToken || crypto.randomBytes(32).toString('hex'),
      updatedAt: new Date().toISOString(),
      setupComplete: true,
      revokedTokens: {},
    };
    this.save(this.record);
    return { ok: true, token: this.createSessionToken() };
  }
}

export function parseCookieHeader(cookieHeader: string | undefined): Record<string, string> {
  const result: Record<string, string> = {};
  if (!cookieHeader) return result;
  for (const part of cookieHeader.split(';')) {
    const idx = part.indexOf('=');
    if (idx <= 0) continue;
    const key = part.slice(0, idx).trim();
    const rawVal = part.slice(idx + 1).trim();
    let val = rawVal;
    try {
      val = decodeURIComponent(rawVal);
    } catch {
      val = rawVal;
    }
    if (key) result[key] = val;
  }
  return result;
}
