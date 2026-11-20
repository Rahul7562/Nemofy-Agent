import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

export interface StoredAuthRecord {
  username: string;
  passwordSalt: string;
  passwordHash: string;
  sessionSecret: string;
  updatedAt: string;
}

export interface SessionPayload {
  username: string;
  issuedAt: number;
  expiresAt: number;
  version: string;
}

const DEFAULT_USERNAME = 'admin';
const DEFAULT_PASSWORD = process.env.NEMOFY_ADMIN_PASSWORD || 'Rahul1234$';
const SESSION_TTL_MS = 24 * 60 * 60 * 1000;

function hashPassword(password: string, salt: string): string {
  return crypto.scryptSync(password, salt, 64).toString('hex');
}

function timingSafeEqualHex(a: string, b: string): boolean {
  const bufA = Buffer.from(a, 'hex');
  const bufB = Buffer.from(b, 'hex');
  if (bufA.length !== bufB.length) return false;
  return crypto.timingSafeEqual(bufA, bufB);
}

export class AuthStore {
  private readonly filePath: string;
  private record: StoredAuthRecord;

  constructor(filePath?: string) {
    this.filePath = path.resolve(filePath || path.join(process.cwd(), 'data', 'auth.json'));
    this.record = this.loadOrSeed();
  }

  private loadOrSeed(): StoredAuthRecord {
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
          return {
            username: parsed.username,
            passwordSalt: parsed.passwordSalt,
            passwordHash: parsed.passwordHash,
            sessionSecret: parsed.sessionSecret,
            updatedAt: parsed.updatedAt || new Date().toISOString(),
          };
        }
      } catch {
        // Re-seed if corrupted
      }
    }

    const salt = crypto.randomBytes(16).toString('hex');
    const seeded: StoredAuthRecord = {
      username: DEFAULT_USERNAME,
      passwordSalt: salt,
      passwordHash: hashPassword(DEFAULT_PASSWORD, salt),
      sessionSecret: crypto.randomBytes(32).toString('hex'),
      updatedAt: new Date().toISOString(),
    };
    this.save(seeded);
    return seeded;
  }

  private save(rec: StoredAuthRecord): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(rec, null, 2) + '\n', {
      encoding: 'utf-8',
      mode: 0o600,
    });
  }

  getUsername(): string {
    return this.record.username;
  }

  getUpdatedAt(): string {
    return this.record.updatedAt;
  }

  verifyCredentials(username: string, password: string): boolean {
    const cleanUser = username.trim();
    if (!cleanUser || !password) return false;
    if (cleanUser !== this.record.username) return false;
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
      return payload;
    } catch {
      return null;
    }
  }

  changePassword(currentPassword: string, newPassword: string): { ok: boolean; error?: string; token?: string } {
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
      updatedAt: new Date().toISOString(),
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
    const val = decodeURIComponent(part.slice(idx + 1).trim());
    if (key) result[key] = val;
  }
  return result;
}
