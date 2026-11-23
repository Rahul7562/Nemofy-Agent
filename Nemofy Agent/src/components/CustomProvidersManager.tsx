import {
  Check,
  Edit3,
  Eye,
  EyeOff,
  KeyRound,
  Plus,
  Server,
  Trash2,
  X,
} from 'lucide-react';
import { useState } from 'react';
import {
  createCustomProvider,
  deleteCustomProvider,
  updateCustomProvider,
} from '../lib/apiClient.ts';
import type {
  CustomProviderConfig,
  CustomProviderModelConfig,
  CustomProviderProtocol,
  ProfileVaultSummary,
  ProviderAvailability,
} from '../types/agent.ts';

interface CustomProvidersManagerProps {
  customProviders: CustomProviderConfig[];
  onUpdated: (payload: {
    custom_providers: CustomProviderConfig[];
    providers: Record<string, ProviderAvailability>;
    vault?: ProfileVaultSummary;
  }) => void;
}

interface DraftModelRow {
  id: string;
  name: string;
  context_window: string;
  max_output_tokens: string;
}

function emptyModelRow(): DraftModelRow {
  return { id: '', name: '', context_window: '', max_output_tokens: '' };
}

function parseOptionalInt(raw: string): number | null {
  const clean = raw.trim().replace(/[_,]/g, '');
  if (!clean) return null;
  const num = Number(clean);
  if (!Number.isFinite(num) || num <= 0) return null;
  return Math.round(num);
}

export function CustomProvidersManager({
  customProviders,
  onUpdated,
}: CustomProvidersManagerProps) {
  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [protocol, setProtocol] = useState<CustomProviderProtocol>('openai_compatible');
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [timeoutSeconds, setTimeoutSeconds] = useState('');
  const [enabled, setEnabled] = useState(true);
  const [models, setModels] = useState<DraftModelRow[]>([emptyModelRow()]);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'ok' | 'error'; text: string } | null>(null);

  function openNewForm() {
    setEditingId(null);
    setName('');
    setBaseUrl('');
    setProtocol('openai_compatible');
    setApiKey('');
    setShowApiKey(false);
    setTimeoutSeconds('');
    setEnabled(true);
    setModels([emptyModelRow()]);
    setFeedback(null);
    setFormOpen(true);
  }

  function openEditForm(cp: CustomProviderConfig) {
    setEditingId(cp.id);
    setName(cp.name);
    setBaseUrl(cp.base_url);
    setProtocol(cp.protocol || 'openai_compatible');
    setApiKey('');
    setShowApiKey(false);
    setTimeoutSeconds(
      typeof cp.timeout_seconds === 'number' && cp.timeout_seconds > 0
        ? String(cp.timeout_seconds)
        : '',
    );
    setEnabled(cp.enabled !== false);
    setModels(
      cp.models.length > 0
        ? cp.models.map((m) => ({
            id: m.id,
            name: m.name || '',
            context_window:
              typeof m.context_window === 'number' && m.context_window > 0
                ? String(m.context_window)
                : '',
            max_output_tokens:
              typeof m.max_output_tokens === 'number' && m.max_output_tokens > 0
                ? String(m.max_output_tokens)
                : '',
          }))
        : [emptyModelRow()],
    );
    setFeedback(null);
    setFormOpen(true);
  }

  function closeForm() {
    setFormOpen(false);
    setEditingId(null);
    setApiKey('');
  }

  async function handleSaveProvider() {
    setFeedback(null);
    const cleanName = name.trim();
    const cleanUrl = baseUrl.trim();
    if (!cleanName) {
      setFeedback({ type: 'error', text: 'Provider display name is required.' });
      return;
    }
    if (!cleanUrl || !/^https?:\/\/.+/.test(cleanUrl)) {
      setFeedback({ type: 'error', text: 'Base URL must start with http:// or https://.' });
      return;
    }

    const parsedModels: CustomProviderModelConfig[] = [];
    const seen = new Set<string>();
    for (const row of models) {
      const id = row.id.trim();
      if (!id || seen.has(id.toLowerCase())) continue;
      seen.add(id.toLowerCase());
      parsedModels.push({
        id,
        name: row.name.trim() || undefined,
        context_window: parseOptionalInt(row.context_window),
        max_output_tokens: parseOptionalInt(row.max_output_tokens),
      });
    }

    const parsedTimeout = parseOptionalInt(timeoutSeconds);
    const payload = {
      name: cleanName,
      base_url: cleanUrl,
      protocol,
      enabled,
      models: parsedModels,
      timeout_seconds: parsedTimeout,
      ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
    };

    setBusy(true);
    try {
      const res = editingId
        ? await updateCustomProvider(editingId, payload)
        : await createCustomProvider(payload);
      onUpdated({
        custom_providers: res.custom_providers,
        providers: res.providers,
        vault: res.vault,
      });
      setFeedback({
        type: 'ok',
        text: editingId
          ? `Updated custom provider "${res.provider.name}".`
          : `Created custom provider "${res.provider.name}" (${res.provider.id}).`,
      });
      closeForm();
    } catch (err) {
      setFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to save custom provider.',
      });
    } finally {
      setBusy(false);
    }
  }

  async function handleToggleEnabled(cp: CustomProviderConfig) {
    setBusy(true);
    setFeedback(null);
    try {
      const res = await updateCustomProvider(cp.id, { enabled: !cp.enabled });
      onUpdated({
        custom_providers: res.custom_providers,
        providers: res.providers,
        vault: res.vault,
      });
      setFeedback({
        type: 'ok',
        text: `${res.provider.name} is now ${res.provider.enabled ? 'enabled' : 'disabled'}.`,
      });
    } catch (err) {
      setFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to update custom provider.',
      });
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(cp: CustomProviderConfig) {
    setBusy(true);
    setFeedback(null);
    try {
      const res = await deleteCustomProvider(cp.id);
      onUpdated({
        custom_providers: res.custom_providers,
        providers: res.providers,
        vault: res.vault,
      });
      if (editingId === cp.id) {
        closeForm();
      }
      setFeedback({
        type: 'ok',
        text: `Deleted custom provider "${cp.name}" and cleared its vault key.`,
      });
    } catch (err) {
      setFeedback({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to delete custom provider.',
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section
      data-testid="custom-providers-section"
      aria-label="Custom Model Providers"
      className="rounded-xl border border-[var(--rule)] bg-[var(--paper-raised)] p-6 sm:p-7 space-y-5"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="font-display text-base font-semibold text-[var(--ink)] flex items-center gap-2">
            <Server className="w-4 h-4 text-[var(--signal)]" />
            <span>Custom Model Providers</span>
          </h2>
          <p className="text-xs text-[var(--ink-soft)] mt-1 leading-relaxed">
            Configure multiple independent OpenAI-compatible or Anthropic-compatible endpoints with isolated encrypted API keys, model lists, and verified context windows.
          </p>
        </div>
        <button
          type="button"
          data-testid="add-custom-provider-button"
          onClick={openNewForm}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper)] hover:border-[var(--signal)] hover:text-[var(--signal)] text-xs font-medium text-[var(--ink)] transition-colors cursor-pointer shrink-0 whitespace-nowrap"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Add Custom Provider</span>
        </button>
      </div>

      {feedback && (
        <div
          role={feedback.type === 'error' ? 'alert' : 'status'}
          data-testid="custom-provider-feedback"
          className={`rounded-lg px-4 py-2.5 text-xs border ${
            feedback.type === 'ok'
              ? 'border-[var(--jade)]/35 bg-[var(--jade)]/10 text-[var(--jade)]'
              : 'border-[var(--flare)]/35 bg-[var(--flare)]/10 text-[var(--flare)]'
          }`}
        >
          {feedback.text}
        </div>
      )}

      {/* Existing Custom Providers List */}
      {customProviders.length > 0 ? (
        <div className="divide-y divide-[var(--rule)] border border-[var(--rule)] rounded-lg overflow-hidden">
          {customProviders.map((cp) => (
            <div
              key={cp.id}
              data-testid={`custom-provider-row-${cp.id}`}
              className={`px-5 py-4 flex flex-wrap items-center justify-between gap-4 bg-[var(--paper)] ${
                !cp.enabled ? 'opacity-65' : ''
              }`}
            >
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-semibold text-[var(--ink)]">{cp.name}</span>
                  <span className="font-mono text-[11px] text-[var(--ink-faint)]">{cp.id}</span>
                  <span className="text-[var(--ink-faint)]">·</span>
                  <span className="text-[11px] text-[var(--ink-soft)]">
                    {cp.protocol === 'anthropic_compatible'
                      ? 'Anthropic-compatible'
                      : 'OpenAI-compatible'}
                  </span>
                  <span className="text-[var(--ink-faint)]">·</span>
                  <span
                    className={`text-[11px] font-medium ${
                      cp.api_key_configured ? 'text-[var(--jade)]' : 'text-[var(--honey)]'
                    }`}
                  >
                    {cp.api_key_configured
                      ? cp.masked_key
                        ? `Key saved (${cp.masked_key})`
                        : 'Key saved'
                      : 'No API key'}
                  </span>
                </div>
                <div className="font-mono text-[11px] text-[var(--ink-faint)] truncate">
                  {cp.base_url}
                  {cp.models.length > 0
                    ? ` · ${cp.models.length} ${cp.models.length === 1 ? 'model' : 'models'}`
                    : ' · Live discovery / custom model IDs'}
                  {typeof cp.timeout_seconds === 'number' && cp.timeout_seconds > 0
                    ? ` · ${cp.timeout_seconds}s timeout`
                    : ''}
                </div>
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  type="button"
                  disabled={busy}
                  data-testid={`toggle-custom-provider-${cp.id}`}
                  onClick={() => void handleToggleEnabled(cp)}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                    cp.enabled
                      ? 'border-[var(--jade)]/40 bg-[var(--jade)]/10 text-[var(--jade)]'
                      : 'border-[var(--rule)] text-[var(--ink-faint)]'
                  }`}
                >
                  {cp.enabled ? 'Enabled' : 'Disabled'}
                </button>
                <button
                  type="button"
                  disabled={busy}
                  data-testid={`edit-custom-provider-${cp.id}`}
                  onClick={() => openEditForm(cp)}
                  aria-label={`Edit ${cp.name}`}
                  className="p-1.5 rounded-md border border-[var(--rule)] text-[var(--ink-soft)] hover:text-[var(--ink)] hover:border-[var(--rule-strong)] cursor-pointer"
                >
                  <Edit3 className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  disabled={busy}
                  data-testid={`delete-custom-provider-${cp.id}`}
                  onClick={() => void handleDelete(cp)}
                  aria-label={`Delete ${cp.name}`}
                  className="p-1.5 rounded-md border border-[var(--rule)] text-[var(--ink-faint)] hover:text-[var(--flare)] hover:border-[var(--flare)]/40 cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        !formOpen && (
          <p className="text-xs text-[var(--ink-faint)]">
            No custom providers configured yet. Add endpoints such as self-hosted vLLM clusters, LiteLLM proxies, Azure gateways, or Anthropic-compatible relays.
          </p>
        )
      )}

      {/* Create / Edit Custom Provider Form */}
      {formOpen && (
        <div
          data-testid="custom-provider-form"
          className="rounded-xl border border-[var(--rule-strong)] bg-[var(--paper)] p-5 sm:p-6 space-y-5"
        >
          <div className="flex items-center justify-between gap-3">
            <h3 className="font-display text-sm font-semibold text-[var(--ink)]">
              {editingId ? `Edit Custom Provider (${editingId})` : 'New Custom Provider'}
            </h3>
            <button
              type="button"
              onClick={closeForm}
              aria-label="Close custom provider form"
              className="p-1.5 rounded-md text-[var(--ink-faint)] hover:text-[var(--ink)] cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="custom-provider-name"
                className="block text-xs font-medium text-[var(--ink)] mb-1.5"
              >
                Display Name
              </label>
              <input
                id="custom-provider-name"
                data-testid="custom-provider-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Internal vLLM Gateway"
                className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper-raised)] text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
              />
            </div>

            <div>
              <label
                htmlFor="custom-provider-protocol"
                className="block text-xs font-medium text-[var(--ink)] mb-1.5"
              >
                API Protocol
              </label>
              <select
                id="custom-provider-protocol"
                data-testid="custom-provider-protocol"
                value={protocol}
                onChange={(e) => setProtocol(e.target.value as CustomProviderProtocol)}
                className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper-raised)] text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
              >
                <option value="openai_compatible">OpenAI-Compatible (/v1/chat/completions)</option>
                <option value="anthropic_compatible">Anthropic-Compatible (/v1/messages)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="sm:col-span-2">
              <label
                htmlFor="custom-provider-base-url"
                className="block text-xs font-medium text-[var(--ink)] mb-1.5"
              >
                Base URL
              </label>
              <input
                id="custom-provider-base-url"
                data-testid="custom-provider-base-url"
                type="url"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://llm-gateway.example.com/v1"
                className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper-raised)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
              />
            </div>

            <div>
              <label
                htmlFor="custom-provider-timeout"
                className="block text-xs font-medium text-[var(--ink)] mb-1.5"
              >
                Read Timeout (sec)
              </label>
              <input
                id="custom-provider-timeout"
                data-testid="custom-provider-timeout"
                type="number"
                min={10}
                max={900}
                value={timeoutSeconds}
                onChange={(e) => setTimeoutSeconds(e.target.value)}
                placeholder="Default (180)"
                className="w-full h-10 px-3 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper-raised)] font-mono text-xs tabular-nums text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
              />
            </div>
          </div>

          <div>
            <label
              htmlFor="custom-provider-api-key"
              className="block text-xs font-medium text-[var(--ink)] mb-1.5"
            >
              API Key <span className="text-[var(--ink-faint)] font-normal">(encrypted in profile vault)</span>
            </label>
            <div className="relative">
              <input
                id="custom-provider-api-key"
                data-testid="custom-provider-api-key"
                type={showApiKey ? 'text' : 'password'}
                autoComplete="off"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={
                  editingId
                    ? 'Leave blank to keep existing vault key, or paste new key'
                    : 'Optional for local/open endpoints; paste bearer key to encrypt'
                }
                className="w-full h-10 pl-3 pr-10 rounded-lg border border-[var(--rule-strong)] bg-[var(--paper-raised)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
              />
              <button
                type="button"
                onClick={() => setShowApiKey((v) => !v)}
                aria-label={showApiKey ? 'Hide API key' : 'Show API key'}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-[var(--ink-faint)] hover:text-[var(--ink)] cursor-pointer"
              >
                {showApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          {/* Models & Verified Context Window Metadata */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-[var(--ink)]">
                Configured Models &amp; Context Limits
              </span>
              <button
                type="button"
                onClick={() => setModels((prev) => [...prev, emptyModelRow()])}
                className="inline-flex items-center gap-1 text-xs font-medium text-[var(--signal)] hover:underline cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add model</span>
              </button>
            </div>

            <div className="space-y-2.5">
              {models.map((row, idx) => (
                <div
                  key={idx}
                  className="grid grid-cols-1 sm:grid-cols-12 gap-2.5 items-center"
                >
                  <input
                    type="text"
                    value={row.id}
                    data-testid={`custom-provider-model-id-${idx}`}
                    onChange={(e) =>
                      setModels((prev) =>
                        prev.map((m, i) => (i === idx ? { ...m, id: e.target.value } : m)),
                      )
                    }
                    placeholder="Model ID (e.g. llama-3.3-70b)"
                    aria-label={`Model ID row ${idx + 1}`}
                    className="sm:col-span-5 h-9 px-3 rounded-md border border-[var(--rule-strong)] bg-[var(--paper-raised)] font-mono text-xs text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                  />
                  <input
                    type="text"
                    inputMode="numeric"
                    value={row.context_window}
                    data-testid={`custom-provider-model-ctx-${idx}`}
                    onChange={(e) =>
                      setModels((prev) =>
                        prev.map((m, i) =>
                          i === idx ? { ...m, context_window: e.target.value } : m,
                        ),
                      )
                    }
                    placeholder="Context (e.g. 131072)"
                    aria-label={`Context window row ${idx + 1}`}
                    className="sm:col-span-3 h-9 px-3 rounded-md border border-[var(--rule-strong)] bg-[var(--paper-raised)] font-mono text-xs tabular-nums text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                  />
                  <input
                    type="text"
                    inputMode="numeric"
                    value={row.max_output_tokens}
                    onChange={(e) =>
                      setModels((prev) =>
                        prev.map((m, i) =>
                          i === idx ? { ...m, max_output_tokens: e.target.value } : m,
                        ),
                      )
                    }
                    placeholder="Max out (e.g. 16384)"
                    aria-label={`Max output tokens row ${idx + 1}`}
                    className="sm:col-span-3 h-9 px-3 rounded-md border border-[var(--rule-strong)] bg-[var(--paper-raised)] font-mono text-xs tabular-nums text-[var(--ink)] focus:outline-none focus:border-[var(--signal)]"
                  />
                  <button
                    type="button"
                    disabled={models.length <= 1}
                    onClick={() =>
                      setModels((prev) => prev.filter((_, i) => i !== idx))
                    }
                    aria-label={`Remove model row ${idx + 1}`}
                    className="sm:col-span-1 h-9 inline-flex items-center justify-center rounded-md text-[var(--ink-faint)] hover:text-[var(--flare)] disabled:opacity-30 cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-[var(--rule)]">
            <label className="inline-flex items-center gap-2 text-xs text-[var(--ink-soft)] cursor-pointer select-none">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
                className="accent-[var(--signal)] w-4 h-4 cursor-pointer"
              />
              <span>Enabled in chat provider selector</span>
            </label>

            <div className="flex items-center gap-2.5">
              <button
                type="button"
                onClick={closeForm}
                className="h-10 px-4 rounded-lg border border-[var(--rule)] text-xs text-[var(--ink-soft)] hover:text-[var(--ink)] cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                data-testid="save-custom-provider-button"
                disabled={busy}
                onClick={() => void handleSaveProvider()}
                className="h-10 px-5 inline-flex items-center gap-1.5 rounded-lg bg-[var(--signal)] hover:bg-[var(--signal-hover)] disabled:opacity-50 text-white text-xs font-semibold cursor-pointer whitespace-nowrap"
              >
                {busy ? <KeyRound className="w-3.5 h-3.5" /> : <Check className="w-3.5 h-3.5" />}
                <span>{busy ? 'Saving...' : editingId ? 'Save Provider' : 'Create Provider'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
