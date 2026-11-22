export const DEFAULT_ADMIN_USERNAME = 'admin';

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

export function validateLoginInput(username: string, password: string): ValidationResult {
  if (!username.trim()) {
    return { valid: false, error: 'Please enter your username.' };
  }
  if (!password) {
    return { valid: false, error: 'Please enter your password.' };
  }
  return { valid: true };
}

export function validateInitialSetup(
  username: string,
  password: string,
  confirmPassword: string,
): ValidationResult {
  const cleanUser = username.trim();
  if (!cleanUser) {
    return { valid: false, error: 'Please enter an administrator username.' };
  }
  if (!/^[a-zA-Z0-9._-]{2,32}$/.test(cleanUser)) {
    return { valid: false, error: 'Username must be 2–32 characters (letters, numbers, ., _, -).' };
  }
  if (password.length < 8) {
    return { valid: false, error: 'Password must be at least 8 characters.' };
  }
  if (password !== confirmPassword) {
    return { valid: false, error: 'Password and confirmation do not match.' };
  }
  return { valid: true };
}

export function validatePasswordChange(
  currentPassword: string,
  newPassword: string,
  confirmPassword: string,
): ValidationResult {
  if (!currentPassword) {
    return { valid: false, error: 'Enter your current password.' };
  }
  if (newPassword.length < 8) {
    return { valid: false, error: 'New password must be at least 8 characters.' };
  }
  if (newPassword === currentPassword) {
    return { valid: false, error: 'New password must differ from your current password.' };
  }
  if (newPassword !== confirmPassword) {
    return { valid: false, error: 'New password and confirmation do not match.' };
  }
  return { valid: true };
}

export async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export async function hashStandaloneCredential(username: string, password: string): Promise<string> {
  return sha256Hex(`nemofy:v1:${username.trim().toLowerCase()}:${password}`);
}
