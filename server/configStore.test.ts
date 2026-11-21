import assert from 'node:assert/strict';
import test from 'node:test';
import { maskEnvSecret, upsertEnvContent } from './configStore.ts';

test('upsertEnvContent preserves comments, updates existing keys, and appends new keys', () => {
  const initial = [
    '# Nemofy Agent env',
    'AGENT_API_TOKEN=secret-tok-123',
    'RIVER_AI_API=old-river-key',
    '',
  ].join('\n');

  const updated = upsertEnvContent(initial, {
    RIVER_AI_API: 'new-river-key-999',
    OPENAI_API_KEY: 'sk-proj-test-1234',
  });

  assert.match(updated, /# Nemofy Agent env/);
  assert.match(updated, /AGENT_API_TOKEN=secret-tok-123/);
  assert.match(updated, /RIVER_AI_API=new-river-key-999/);
  assert.match(updated, /OPENAI_API_KEY=sk-proj-test-1234/);
  assert.equal(updated.includes('old-river-key'), false);
});

test('maskEnvSecret redacts secrets safely', () => {
  assert.equal(maskEnvSecret(''), null);
  assert.equal(maskEnvSecret('short'), '****');
  assert.equal(maskEnvSecret('sk-proj-1234567890abcd'), 'sk-p...abcd');
});
