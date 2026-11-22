import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { WorkspaceAndConfigStore } from './configStore.ts';
import { ChatJobOrchestrator } from './jobOrchestrator.ts';

test('ChatJobOrchestrator completes and persists background jobs even when zero browser clients poll', async () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-orch-bg-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);
    const now = new Date().toISOString();
    store.updateThreads((threads) => {
      threads.push({
        id: 'thread-bg-1',
        title: 'Background Task',
        createdAt: now,
        updatedAt: now,
        messages: [{ id: 'u1', role: 'user', content: 'Run background check', timestamp: now }],
      });
    });

    let pollCount = 0;
    const orchestrator = new ChatJobOrchestrator({
      store,
      pollIntervalMs: 50,
      callAgentCore: async (endpoint) => {
        if (endpoint.includes('/cancel')) {
          return { ok: true, status: 200, data: { status: 'cancelled' } };
        }
        pollCount += 1;
        if (pollCount < 3) {
          return {
            ok: true,
            status: 200,
            data: {
              job_id: 'job-bg-1',
              status: 'running',
              current_phase: 'thinking',
              last_event: `Step ${pollCount}`,
              elapsed_seconds: pollCount * 0.2,
              turn_count: pollCount,
            },
          };
        }
        return {
          ok: true,
          status: 200,
          data: {
            job_id: 'job-bg-1',
            status: 'completed',
            current_phase: 'done',
            last_event: 'Done',
            elapsed_seconds: 0.8,
            turn_count: 3,
            result: {
              final_response: 'Background task completed cleanly!',
              status: 'completed',
              turn_count: 3,
            },
          },
        };
      },
    });

    orchestrator.registerJob({
      jobId: 'job-bg-1',
      threadId: 'thread-bg-1',
      initialPayload: {
        job_id: 'job-bg-1',
        status: 'running',
        current_phase: 'thinking',
        last_event: 'Starting',
      },
    });

    assert.equal(orchestrator.getActiveJobs().length, 1);

    // Wait for background poller to finish WITHOUT any client calling pollJobForClient
    await new Promise((r) => setTimeout(r, 320));

    assert.equal(orchestrator.getActiveJobs().length, 0);
    const persisted = store.listThreads().find((t) => t.id === 'thread-bg-1');
    assert.ok(persisted);
    assert.equal(persisted.messages.length, 2);
    assert.equal(persisted.messages[1].role, 'assistant');
    assert.equal(persisted.messages[1].content, 'Background task completed cleanly!');

    orchestrator.shutdown();
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});

test('ChatJobOrchestrator finalizes idempotently under concurrent client polls and tracks multi-thread runs', async () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-orch-idemp-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);
    const now = new Date().toISOString();
    store.updateThreads((threads) => {
      threads.push({
        id: 'thread-a',
        title: 'Session A',
        createdAt: now,
        updatedAt: now,
        messages: [],
      });
      threads.push({
        id: 'thread-b',
        title: 'Session B',
        createdAt: now,
        updatedAt: now,
        messages: [],
      });
    });

    const orchestrator = new ChatJobOrchestrator({
      store,
      pollIntervalMs: 500,
      callAgentCore: async (endpoint) => {
        const jobId = endpoint.includes('job-a') ? 'job-a' : 'job-b';
        return {
          ok: true,
          status: 200,
          data: {
            job_id: jobId,
            status: 'completed',
            current_phase: 'done',
            result: {
              final_response: `Final reply for ${jobId}`,
              status: 'completed',
              turn_count: 1,
            },
          },
        };
      },
    });

    orchestrator.registerJob({
      jobId: 'job-a',
      threadId: 'thread-a',
      initialPayload: { status: 'running', current_phase: 'thinking' },
    });
    orchestrator.registerJob({
      jobId: 'job-b',
      threadId: 'thread-b',
      initialPayload: { status: 'running', current_phase: 'thinking' },
    });
    assert.equal(orchestrator.getActiveJobForThread('thread-a')?.jobId, 'job-a');
    assert.equal(orchestrator.getActiveJobForThread('thread-b')?.jobId, 'job-b');

    // Race two concurrent polls for job-a
    const [r1, r2] = await Promise.all([
      orchestrator.pollJobForClient('job-a'),
      orchestrator.pollJobForClient('job-a'),
    ]);

    assert.equal(r1.ok && r1.done, true);
    assert.equal(r2.ok && r2.done, true);

    // Exactly ONE assistant message must be persisted (no duplication!)
    const threadA = store.listThreads().find((t) => t.id === 'thread-a');
    assert.ok(threadA);
    assert.equal(threadA.messages.length, 1);
    assert.equal(threadA.messages[0].content, 'Final reply for job-a');

    orchestrator.shutdown();
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});

test('ChatJobOrchestrator does not resurrect deleted threads when a background job finishes', async () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemofy-orch-del-'));
  try {
    const store = new WorkspaceAndConfigStore(tmpRoot);
    const now = new Date().toISOString();
    store.updateThreads((threads) => {
      threads.push({
        id: 'thread-del',
        title: 'To Delete',
        createdAt: now,
        updatedAt: now,
        messages: [],
      });
    });

    const orchestrator = new ChatJobOrchestrator({
      store,
      pollIntervalMs: 500,
      callAgentCore: async () => ({
        ok: true,
        status: 200,
        data: {
          job_id: 'job-del',
          status: 'completed',
          current_phase: 'done',
          result: { final_response: 'Late reply', status: 'completed', turn_count: 1 },
        },
      }),
    });

    orchestrator.registerJob({
      jobId: 'job-del',
      threadId: 'thread-del',
      initialPayload: { status: 'running', current_phase: 'thinking' },
    });

    // Delete the thread from store before finalization
    store.updateThreads((threads) => {
      const idx = threads.findIndex((t) => t.id === 'thread-del');
      if (idx >= 0) threads.splice(idx, 1);
    });

    const res = await orchestrator.pollJobForClient('job-del');
    assert.equal(res.ok && res.done, true);
    assert.equal(store.listThreads().some((t) => t.id === 'thread-del'), false);

    orchestrator.shutdown();
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});
