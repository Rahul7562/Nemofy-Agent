import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { maskEnvSecret } from './configStore.ts';

export const ALLOWED_PROFILE_SECRET_KEYS = new Set<string>([
  'RIVER_AI_API',
  'NVIDIA_API_KEY',
  'OPENAI_API_KEY',
  'OPENROUTER_API_KEY',
  'GEMINI_API_KEY',
  'GOOGLE_API_KEY',
  'DEEPSEEK_API_KEY',
  'GROQ_API_KEY',
  'ANTHROPIC_API_KEY',
  'TOGETHER_API_KEY',
  'FIREWORKS_API_KEY',
  'OLLAMA_API_KEY',
  'VLLM_API_KEY',
  'PRIMARY_API_KEY',
  'WORKER_API_KEY',
]);

export const ENV_VAR_TO_PROVIDER_META: Record<string, { provider: string; label: string }> = {
  RIVER_AI_API: { provider: 'river_ai', label: 'River AI' },
  NVIDIA_API_KEY: { provider: 'nvidia', label: 'NVIDIA NIM' },
  OPENAI_API_KEY: { provider: 'openai', label: 'OpenAI' },
  OPENROUTER_API_KEY: { provider: 'openrouter', label: 'OpenRouter' },
  GEMINI_API_KEY: { provider: 'gemini', label: 'Google Gemini' },
  GOOGLE_API_KEY: { provider: 'gemini', label: 'Google Gemini' },
  DEEPSEEK_API_KEY: { provider: 'deepseek', label: 'DeepSeek' },
  GROQ_API_KEY: { provider: 'groq', label: 'Groq' },
  ANTHROPIC_API_KEY: { provider: 'anthropic', label: 'Anthropic' },
  TOGETHER_API_KEY: { provider: 'together', label: 'Together AI' },
  FIREWORKS_API_KEY: { provider: 'fireworks', label: 'Fireworks AI' },
  OLLAMA_API_KEY: { provider: 'ollama', label: 'Ollama' },
  VLLM_API_KEY: { provider: 'vllm', label: 'vLLM' },
  PRIMARY_API_KEY: { provider: 'custom', label: 'Custom (Main Agent)' },
  WORKER_API_KEY: { provider: 'custom', label: 'Custom (Sub-Agent)' },
};

export interface ProfileSecretEntryMeta {
  env_var: string;
  provider: string;
  provider_label: string;
  masked_key: string;
  updated_at: string;
}

interface StoredSecretRecord {
  value: string;
  updatedAt: string;
}

interface DecryptedVaultPayload {
  version: 1;
  username: string;
  secrets: Record<string, StoredSecretRecord>;
}

interface EncryptedVaultEnvelope {
  version: 1;
  algorithm: 'aes-256-gcm';
  kdf: 'scrypt';
  username: string;
  salt: string;
  iv: string;
  tag: string;
  ciphertext: string;
  updatedAt: string;
}

const USERNAME_PATTERN = /^[a-z0-9_-]{1,32}$/i;

function normalizeUsername(username: string): string {
  const clean = (username || '').trim().toLowerCase();
  if (!USERNAME_PATTERN.test(clean)) {
    throw new Error('Invalid profile username.');
  }
  return clean;
}

function buildAad(username: string): Buffer {
  return Buffer.from(`nemofy-profile-vault:v1:${username}`, 'utf-8');
}

function deriveEncryptionKey(masterMaterial: string, username: string, saltHex: string): Buffer {
  const salt = Buffer.from(saltHex, 'hex');
  return crypto.scryptSync(`${masterMaterial}:${username}`, salt, 32, {
    N: 16384,
    r: 8,
    p: 1,
  });
}

export function stripSecretKeysFromEnvContent(
  existingContent: string,
  keysToStrip: Set<string>,
): { content: string; stripped: string[] } {
  if (!existingContent || keysToStrip.size === 0) {
    return { content: existingContent, stripped: [] };
  }
  const lines = existingContent.split(/\r?\n/);
  const nextLines: string[] = [];
  const stripped: string[] = [];

  for (const line of lines) {
    const match = line.match(/^\s*([A-Z][A-Z0-9_]*)\s*=(.*)$/);
    if (match && keysToStrip.has(match[1])) {
      stripped.push(match[1]);
      continue;
    }
    nextLines.push(line);
  }

  return {
    content: nextLines.join('\n').replace(/\n*$/, '\n'),
    stripped,
  };
}

export class ProfileSecretVaultStore {
  private readonly vaultsDir: string;
  private readonly getMasterMaterial: () => string;

  constructor(vaultsDir: string, getMasterMaterial: () => string) {
    this.vaultsDir = path.resolve(vaultsDir);
    this.getMasterMaterial = getMasterMaterial;
    fs.mkdirSync(this.vaultsDir, { recursive: true, mode: 0o700 });
  }

  private resolveVaultPath(username: string): { safeUser: string; filePath: string } {
    const safeUser = normalizeUsername(username);
    const filePath = path.resolve(this.vaultsDir, `${safeUser}.vault.json`);
    if (!filePath.startsWith(this.vaultsDir + path.sep)) {
      throw new Error('Vault path escapes storage directory.');
    }
    return { safeUser, filePath };
  }

  private readVaultPayload(username: string): DecryptedVaultPayload {
    const { safeUser, filePath } = this.resolveVaultPath(username);
    if (!fs.existsSync(filePath)) {
      return { version: 1, username: safeUser, secrets: {} };
    }

    const raw = fs.readFileSync(filePath, 'utf-8');
    const envl = JSON.parse(raw) as EncryptedVaultEnvelope;
    if (
      !envl ||
      envl.version !== 1 ||
      envl.algorithm !== 'aes-256-gcm' ||
      envl.username !== safeUser ||
      typeof envl.salt !== 'string' ||
      typeof envl.iv !== 'string' ||
      typeof envl.tag !== 'string' ||
      typeof envl.ciphertext !== 'string'
    ) {
      throw new Error('Encrypted profile vault envelope is invalid.');
    }

    const key = deriveEncryptionKey(this.getMasterMaterial(), safeUser, envl.salt);
    const decipher = crypto.createDecipheriv(
      'aes-256-gcm',
      key,
      Buffer.from(envl.iv, 'hex'),
    );
    decipher.setAAD(buildAad(safeUser));
    decipher.setAuthTag(Buffer.from(envl.tag, 'hex'));

    const decrypted = Buffer.concat([
      decipher.update(Buffer.from(envl.ciphertext, 'base64')),
      decipher.final(),
    ]).toString('utf-8');

    const parsed = JSON.parse(decrypted) as DecryptedVaultPayload;
    if (!parsed || parsed.username !== safeUser || typeof parsed.secrets !== 'object') {
      throw new Error('Decrypted profile vault payload is malformed.');
    }
    return {
      version: 1,
      username: safeUser,
      secrets: parsed.secrets || {},
    };
  }

  private writeVaultPayload(payload: DecryptedVaultPayload): void {
    const { safeUser, filePath } = this.resolveVaultPath(payload.username);
    const saltHex = crypto.randomBytes(32).toString('hex');
    const iv = crypto.randomBytes(12);
    const key = deriveEncryptionKey(this.getMasterMaterial(), safeUser, saltHex);

    const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
    cipher.setAAD(buildAad(safeUser));
    const plaintext = JSON.stringify({
      version: 1,
      username: safeUser,
      secrets: payload.secrets,
    });
    const ciphertext = Buffer.concat([cipher.update(plaintext, 'utf-8'), cipher.final()]);
    const tag = cipher.getAuthTag();

    const envelope: EncryptedVaultEnvelope = {
      version: 1,
      algorithm: 'aes-256-gcm',
      kdf: 'scrypt',
      username: safeUser,
      salt: saltHex,
      iv: iv.toString('hex'),
      tag: tag.toString('hex'),
      ciphertext: ciphertext.toString('base64'),
      updatedAt: new Date().toISOString(),
    };

    fs.mkdirSync(this.vaultsDir, { recursive: true, mode: 0o700 });
    const tmpPath = `${filePath}.${crypto.randomBytes(6).toString('hex')}.tmp`;
    fs.writeFileSync(tmpPath, JSON.stringify(envelope, null, 2) + '\n', {
      encoding: 'utf-8',
      mode: 0o600,
    });
    fs.renameSync(tmpPath, filePath);
  }

  getDecryptedSecrets(username: string): Record<string, string> {
    const payload = this.readVaultPayload(username);
    const out: Record<string, string> = {};
    for (const [key, entry] of Object.entries(payload.secrets)) {
      if (ALLOWED_PROFILE_SECRET_KEYS.has(key) && entry && typeof entry.value === 'string' && entry.value.trim()) {
        out[key] = entry.value.trim();
      }
    }
    return out;
  }

  listSecretMetadata(username: string): ProfileSecretEntryMeta[] {
    const payload = this.readVaultPayload(username);
    const items: ProfileSecretEntryMeta[] = [];
    for (const [envVar, entry] of Object.entries(payload.secrets)) {
      if (!ALLOWED_PROFILE_SECRET_KEYS.has(envVar) || !entry?.value?.trim()) continue;
      const meta = ENV_VAR_TO_PROVIDER_META[envVar] || { provider: 'custom', label: envVar };
      items.push({
        env_var: envVar,
        provider: meta.provider,
        provider_label: meta.label,
        masked_key: maskEnvSecret(entry.value) || '****',
        updated_at: entry.updatedAt || new Date().toISOString(),
      });
    }
    return items.sort((a, b) => a.provider_label.localeCompare(b.provider_label));
  }

  setSecrets(username: string, updates: Record<string, string>): string[] {
    const cleanUpdates: Record<string, string> = {};
    for (const [rawKey, rawVal] of Object.entries(updates)) {
      const envVar = rawKey.trim().toUpperCase();
      const val = (rawVal || '').trim();
      if (!ALLOWED_PROFILE_SECRET_KEYS.has(envVar)) {
        throw new Error(`Unsupported API key variable: ${envVar}`);
      }
      if (!val) continue;
      cleanUpdates[envVar] = val;
    }

    const keys = Object.keys(cleanUpdates);
    if (keys.length === 0) return [];

    const payload = this.readVaultPayload(username);
    const now = new Date().toISOString();
    for (const [envVar, val] of Object.entries(cleanUpdates)) {
      payload.secrets[envVar] = { value: val, updatedAt: now };
    }
    this.writeVaultPayload(payload);
    return keys;
  }

  deleteSecret(username: string, envVarRaw: string): boolean {
    const envVar = (envVarRaw || '').trim().toUpperCase();
    if (!ALLOWED_PROFILE_SECRET_KEYS.has(envVar)) {
      throw new Error(`Unsupported API key variable: ${envVar}`);
    }
    const payload = this.readVaultPayload(username);
    if (!(envVar in payload.secrets)) {
      return false;
    }
    delete payload.secrets[envVar];
    this.writeVaultPayload(payload);
    return true;
  }

  /**
   * One-time automatic migration of any existing plaintext provider keys in .env
   * or process.env into the user's encrypted vault, then strips those plaintext
   * provider keys from .env while preserving non-provider vars like AGENT_API_TOKEN.
   */
  migrateLegacyEnvSecrets(
    username: string,
    envFilePath: string,
    envSource: Record<string, string | undefined> = process.env,
  ): string[] {
    const existingPayload = this.readVaultPayload(username);
    const discovered: Record<string, string> = {};

    // 1. Read any legacy provider keys from .env file on disk
    let rawEnvContent = '';
    if (envFilePath && fs.existsSync(envFilePath)) {
      try {
        rawEnvContent = fs.readFileSync(envFilePath, 'utf-8');
        for (const rawLine of rawEnvContent.split(/\r?\n/)) {
          const line = rawLine.trim();
          if (!line || line.startsWith('#')) continue;
          const eqIdx = line.indexOf('=');
          if (eqIdx <= 0) continue;
          const key = line.slice(0, eqIdx).trim();
          if (!ALLOWED_PROFILE_SECRET_KEYS.has(key)) continue;
          let val = line.slice(eqIdx + 1).trim();
          if (
            (val.startsWith('"') && val.endsWith('"')) ||
            (val.startsWith("'") && val.endsWith("'"))
          ) {
            val = val.slice(1, -1);
          }
          if (val && !existingPayload.secrets[key]?.value) {
            discovered[key] = val;
          }
        }
      } catch {
        // ignore read error
      }
    }

    // 2. Also pick up any provider keys loaded into process.env that aren't in the vault yet
    for (const key of ALLOWED_PROFILE_SECRET_KEYS) {
      const val = (envSource[key] || '').trim();
      if (val && !existingPayload.secrets[key]?.value && !discovered[key]) {
        discovered[key] = val;
      }
    }

    const migrated = this.setSecrets(username, discovered);

    // 3. Scrub plaintext provider keys from .env so they only live encrypted in the vault
    if (rawEnvContent) {
      const keysToScrub = new Set<string>([
        ...Object.keys(existingPayload.secrets),
        ...migrated,
      ]);
      const { content: cleaned, stripped } = stripSecretKeysFromEnvContent(rawEnvContent, keysToScrub);
      if (stripped.length > 0) {
        fs.writeFileSync(envFilePath, cleaned, { encoding: 'utf-8', mode: 0o600 });
      }
    }

    return migrated;
  }
}
