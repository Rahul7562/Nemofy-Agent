import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatThread, ContextUsageSnapshot, PersistedLLMConfig } from '../types/agent.ts';
import {
  AUTO_COMPACT_THRESHOLD_PERCENT,
  classifyContextUsage,
  computeModelContextUsage,
  formatTokenCount,
  rescaleSnapshotForModel,
  resolveModelContextWindow,
  resolveModelLimits,
  resolveModelMaxOutputTokens,
} from './contextUtils.ts';

function makeConfig(model: string, provider = 'river_ai', contextWindow?: number | null): PersistedLLMConfig {
  return {
    primary: {
      provider,
      model,
      base_url: 'https://api.river.ai/v1',
      temperature: 0.2,
      max_tokens: 16384,
      reasoning_effort: 'off',
      context_window: contextWindow ?? undefined,
    },
    worker: {
      provider: 'river_ai',
      model: 'deepseek-ai/DeepSeek-V4.1-Flash',
      base_url: 'https://api.river.ai/v1',
      temperature: 0.1,
      max_tokens: 8192,
      reasoning_effort: 'off',
    },
    max_turns: 30,
  };
}

test('resolveModelContextWindow returns model-specific context limits and prefers explicit catalog values', () => {
  assert.deepEqual(resolveModelContextWindow('gemini', 'gemini-2.5-pro'), {
    contextWindow: 1_048_576,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('anthropic', 'claude-opus-4-6'), {
    contextWindow: 1_000_000,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('anthropic', 'claude-opus-4-5'), {
    contextWindow: 200_000,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('openai', 'gpt-5.2'), {
    contextWindow: 400_000,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('openai', 'gpt-5.6-luna'), {
    contextWindow: 1_050_000,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('river_ai', 'deepseek-ai/DeepSeek-V4.1-Flash'), {
    contextWindow: 1_048_576,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('nvidia', 'minimaxai/minimax-m3.1-flash'), {
    contextWindow: 1_048_576,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('nvidia', 'mistralai/mistral-large-4-1t-a52b'), {
    contextWindow: 1_048_576,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('nvidia', 'google/gemma-4-31b-it'), {
    contextWindow: 262_144,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('river_ai', 'GLM-5.3'), {
    contextWindow: 1_048_576,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('river_ai', 'Qwen/Qwen3.8-27B-FP8'), {
    contextWindow: 1_024_000,
    source: 'model_registry',
  });
  assert.deepEqual(resolveModelContextWindow('ollama', 'my-custom-local-model'), {
    contextWindow: 32_768,
    source: 'provider_default',
  });
  assert.deepEqual(resolveModelContextWindow('openrouter', 'some/custom-model', 262_144), {
    contextWindow: 262_144,
    source: 'provider_catalog',
  });
});

test('resolveModelMaxOutputTokens and resolveModelLimits return exact model-specific input, output, and total limits', () => {
  // 1M Gemini 2.5 Pro -> 1,048,576 total / 65,536 out
  const gemini = resolveModelLimits('gemini', 'gemini-2.5-pro');
  assert.equal(gemini.contextWindow, 1_048_576);
  assert.equal(gemini.maxOutputTokens, 65_536);

  // GPT-5.2 -> 400k total / 128k out
  const gpt52 = resolveModelLimits('openai', 'gpt-5.2');
  assert.equal(gpt52.contextWindow, 400_000);
  assert.equal(gpt52.maxOutputTokens, 128_000);

  // Claude Opus 4.6 -> 1M total / 128k out
  const opus46 = resolveModelLimits('anthropic', 'claude-opus-4-6');
  assert.equal(opus46.contextWindow, 1_000_000);
  assert.equal(opus46.maxOutputTokens, 128_000);

  // Kimi K3 -> 1M total / 131,072 out
  const kimi = resolveModelLimits('river_ai', 'moonshotai/Kimi-K3.1-Pro');
  assert.equal(kimi.contextWindow, 1_048_576);
  assert.equal(kimi.maxOutputTokens, 131_072);

  // DeepSeek V4 -> 1M total / 131,072 out vs DeepSeek Chat -> 128k total / 8,192 out
  const dsv4 = resolveModelLimits('river_ai', 'deepseek-ai/DeepSeek-V4.1-Flash');
  assert.equal(dsv4.contextWindow, 1_048_576);
  assert.equal(dsv4.maxOutputTokens, 131_072);
  assert.deepEqual(resolveModelMaxOutputTokens('deepseek', 'deepseek-chat'), {
    maxOutputTokens: 8_192,
    source: 'model_registry',
  });

  // Live provider catalog values override registry cleanly (no artificial cap, e.g. 2M ctx / 256k out)
  const liveCustom = resolveModelLimits('openrouter', 'custom/long-context', 2_000_000, 256_000);
  assert.equal(liveCustom.contextWindow, 2_000_000);
  assert.equal(liveCustom.contextWindowSource, 'provider_catalog');
  assert.equal(liveCustom.maxOutputTokens, 256_000);
  assert.equal(liveCustom.maxOutputTokensSource, 'provider_catalog');

  // Output limit can never exceed total context window
  const clamped = resolveModelLimits('custom', 'tiny-model', 4_096, 65_536);
  assert.equal(clamped.maxOutputTokens, 4_096);
});

test('classifyContextUsage enforces strict 80% Auto-Compacting threshold and tier boundaries', () => {
  assert.equal(AUTO_COMPACT_THRESHOLD_PERCENT, 80);
  assert.equal(classifyContextUsage(0).level, 'optimal');
  assert.equal(classifyContextUsage(49.9).label, 'Optimal');
  assert.equal(classifyContextUsage(50.0).level, 'moderate');
  assert.equal(classifyContextUsage(69.9).label, 'Moderate');
  assert.equal(classifyContextUsage(70.0).level, 'high');
  assert.equal(classifyContextUsage(79.9).level, 'high');
  assert.equal(classifyContextUsage(79.9).label, 'High');
  assert.equal(classifyContextUsage(80.0).level, 'critical');
  assert.equal(classifyContextUsage(80.0).label, 'Auto-Compacting');
  assert.equal(classifyContextUsage(95.4).level, 'critical');
});

test('rescaleSnapshotForModel adapts saved thread usage dynamically when user switches Primary model', () => {
  const baseSnapshot: ContextUsageSnapshot = {
    model: 'deepseek-chat',
    provider: 'deepseek',
    context_window: 128_000,
    context_window_source: 'model_registry',
    used_tokens: 102_400, // 80.0% of 128k
    free_tokens: 25_600,
    usage_percent: 80.0,
    auto_compact_threshold_percent: 80,
    classification: classifyContextUsage(80.0),
    categories: [
      { key: 'system_prompt', label: 'System Prompt', description: '', tokens: 1_000, percent: 0.8, share_of_used: 1.0 },
      { key: 'user_messages', label: 'User Messages', description: '', tokens: 50_000, percent: 39.1, share_of_used: 48.8 },
      { key: 'assistant_messages', label: 'Assistant & Reasoning', description: '', tokens: 51_400, percent: 40.2, share_of_used: 50.2 },
    ],
    compacted_count: 1,
    last_compacted_at: '2026-11-21T00:00:00Z',
    tokens_saved_last_compaction: 40_000,
    calibrated_from_provider: true,
  };

  // Switch to 1M Gemini -> same 102.4k tokens is now ~9.8% Optimal, max output updates to 65,536!
  const onGemini = rescaleSnapshotForModel(baseSnapshot, 'gemini', 'gemini-2.5-pro');
  assert.equal(onGemini.context_window, 1_048_576);
  assert.equal(onGemini.max_output_tokens, 65_536);
  assert.equal(onGemini.usage_percent, 9.8);
  assert.equal(onGemini.classification.level, 'optimal');
  assert.equal(onGemini.categories.length, 7);

  // Switch to 200k Claude Haiku 4.5 -> 102.4k tokens is 51.2% Moderate, max output updates to 64,000
  const onClaude = rescaleSnapshotForModel(baseSnapshot, 'anthropic', 'claude-haiku-4-5');
  assert.equal(onClaude.context_window, 200_000);
  assert.equal(onClaude.max_output_tokens, 64_000);
  assert.equal(onClaude.usage_percent, 51.2);
  assert.equal(onClaude.classification.level, 'moderate');
});

test('computeModelContextUsage estimates all 7 categories for threads without backend snapshots', () => {
  const thread: ChatThread = {
    id: 't-1',
    title: 'Test Thread',
    createdAt: '2026-11-21T00:00:00Z',
    updatedAt: '2026-11-21T00:00:00Z',
    messages: [
      {
        id: 'm-1',
        role: 'user',
        content: 'Analyze this architecture diagram and summarize key modules.',
        timestamp: '2026-11-21T00:00:00Z',
        attachments: [
          {
            id: 'att-1',
            name: 'diagram.png',
            kind: 'image',
            mimeType: 'image/png',
            size: 4096,
            url: '/api/media/t-1/diagram.png',
          },
        ],
      },
      {
        id: 'm-2',
        role: 'assistant',
        content: 'Here is a breakdown of the key modules.',
        thinking: 'Checking each layer carefully.',
        timestamp: '2026-11-21T00:00:01Z',
        toolEvents: [
          {
            id: 'ev-1',
            tool: 'read_file',
            args_summary: 'path=README.md',
            status: 'ok',
            output_summary: 'Read 120 lines from README.md',
          },
        ],
      },
    ],
  };

  const snap = computeModelContextUsage({
    thread,
    llmConfig: makeConfig('claude-opus-4-6', 'anthropic'),
  });

  assert.equal(snap.context_window, 1_000_000);
  assert.equal(snap.max_output_tokens, 128_000);
  assert.equal(snap.configured_max_tokens, 16_384);
  assert.equal(snap.categories.length, 7);
  const byKey = Object.fromEntries(snap.categories.map((c) => [c.key, c.tokens]));
  assert.ok(byKey.system_prompt > 0);
  assert.ok(byKey.tools > 0);
  assert.ok(byKey.skills > 0);
  assert.ok(byKey.user_messages > 1_000); // includes image attachment overhead
  assert.ok(byKey.assistant_messages > 0);
  assert.ok(byKey.tool_outputs > 0);
  assert.equal(byKey.compacted_summary, 0);
  assert.equal(snap.used_tokens + snap.free_tokens, snap.context_window);
});

test('formatTokenCount formats tokens cleanly across scales', () => {
  assert.equal(formatTokenCount(0), '0');
  assert.equal(formatTokenCount(842), '842');
  assert.equal(formatTokenCount(1_540), '1.5k');
  assert.equal(formatTokenCount(128_000), '128k');
  assert.equal(formatTokenCount(1_048_576), '1M');
  assert.equal(formatTokenCount(2_000_000), '2M');
  assert.equal(formatTokenCount(10_000_000), '10M');
});
