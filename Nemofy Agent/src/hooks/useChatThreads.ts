import { useCallback, useEffect, useState } from 'react';
import { createThread, deleteThread, listThreads } from '../lib/apiClient.ts';
import type { ChatThread } from '../types/agent.ts';

export function useChatThreads(authenticated: boolean) {
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [loadingThreads, setLoadingThreads] = useState(true);
  const [threadsError, setThreadsError] = useState<string | null>(null);

  useEffect(() => {
    if (!authenticated) {
      setThreads([]);
      setActiveThreadId(null);
      setLoadingThreads(false);
      setThreadsError(null);
      return;
    }

    let mounted = true;
    setLoadingThreads(true);
    (async () => {
      try {
        const loaded = await listThreads();
        if (!mounted) return;
        setThreads(loaded);
        setActiveThreadId((prev) => {
          if (prev && loaded.some((t) => t.id === prev)) return prev;
          return loaded[0]?.id || null;
        });
        setThreadsError(null);
      } catch (err) {
        if (mounted) {
          setThreadsError(err instanceof Error ? err.message : 'Unable to load sessions.');
        }
      } finally {
        if (mounted) setLoadingThreads(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [authenticated]);

  const upsertThread = useCallback((thread: ChatThread, activate = false) => {
    setThreads((prev) => {
      const exists = prev.some((t) => t.id === thread.id);
      if (exists) {
        return prev.map((t) => (t.id === thread.id ? thread : t));
      }
      return [thread, ...prev];
    });
    if (activate) {
      setActiveThreadId(thread.id);
    }
  }, []);

  const updateThreadById = useCallback(
    (id: string, updater: (thread: ChatThread) => ChatThread) => {
      setThreads((prev) => prev.map((t) => (t.id === id ? updater(t) : t)));
    },
    [],
  );

  const createNewThread = useCallback(async (title = 'New Agent Session'): Promise<ChatThread | null> => {
    setThreadsError(null);
    try {
      const created = await createThread(title);
      setThreads((prev) => [created, ...prev]);
      setActiveThreadId(created.id);
      return created;
    } catch (err) {
      setThreadsError(err instanceof Error ? err.message : 'Unable to create session.');
      return null;
    }
  }, []);

  const removeThread = useCallback(async (id: string) => {
    setThreadsError(null);
    try {
      await deleteThread(id);
      setThreads((prev) => {
        const remaining = prev.filter((t) => t.id !== id);
        setActiveThreadId((current) => (current === id ? remaining[0]?.id || null : current));
        return remaining;
      });
    } catch (err) {
      setThreadsError(err instanceof Error ? err.message : 'Unable to delete session.');
    }
  }, []);

  const activeThread = threads.find((t) => t.id === activeThreadId) || null;

  return {
    threads,
    activeThread,
    activeThreadId,
    setActiveThreadId,
    loadingThreads,
    threadsError,
    setThreadsError,
    createNewThread,
    removeThread,
    upsertThread,
    updateThreadById,
  };
}
