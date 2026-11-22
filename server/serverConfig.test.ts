import assert from 'node:assert/strict';
import test from 'node:test';
import { resolveServerListenConfig } from './serverConfig.ts';

test('prefers CLI --port and --host flags over container PORT env', () => {
  const cfg = resolveServerListenConfig(
    ['--port', '3000', '--host', '0.0.0.0'],
    { PORT: '8080', HOST: '127.0.0.1' },
  );
  assert.deepEqual(cfg, { port: 3000, host: '0.0.0.0' });
});

test('supports --port=<n>, -p <n>, and --host=<h> flag forms', () => {
  assert.deepEqual(
    resolveServerListenConfig(['--port=4123', '--host=127.0.0.1'], { PORT: '8080' }),
    { port: 4123, host: '127.0.0.1' },
  );
  assert.deepEqual(
    resolveServerListenConfig(['-p', '5050'], {}),
    { port: 5050, host: '0.0.0.0' },
  );
});

test('falls back to PORT/HOST env variables and safe defaults', () => {
  assert.deepEqual(resolveServerListenConfig([], { PORT: '3099', HOST: '127.0.0.1' }), {
    port: 3099,
    host: '127.0.0.1',
  });
  assert.deepEqual(resolveServerListenConfig([], {}), {
    port: 3000,
    host: '0.0.0.0',
  });
});

test('ignores invalid or out-of-range port values', () => {
  assert.deepEqual(resolveServerListenConfig(['--port', '99999'], { PORT: 'abc' }), {
    port: 3000,
    host: '0.0.0.0',
  });
  assert.deepEqual(resolveServerListenConfig(['--port', '0'], { PORT: '3001' }), {
    port: 3001,
    host: '0.0.0.0',
  });
});
