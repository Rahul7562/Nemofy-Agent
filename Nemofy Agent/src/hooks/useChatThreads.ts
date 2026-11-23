import { useCallback, useEffect, useRef, useState } from 'react';
import { createThread, deleteThread, listThreads } from '../lib/apiClient.ts';
import type { ChatThread, ThreadSessionConfig } from '../types/agent.ts';

export function useChatThreads(authenticated: boolean) {
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [loadingThreads, setLoadingThreads] = useState(true);
  const [threadsError, setThreadsError] = useState<string | null>(null);
  const deletedIdsRef = useRef<Set<string>>(new Set());
  const threadsRef = useRef<ChatThread[]>([]);
  threadsRef.current = threads;

  useEffect(() => {
    if (!authenticated) {
      deletedIdsRef.current.clear();
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
        const filteredLoaded = loaded.filter((t) => !deletedIdsRef.current.has(t.id));
        setThreads((prev) => {
          // Preserve any threads created locally while initial load was in-flight
          const loadedIds = new Set(filteredLoaded.map((t) => t.id));
          const localOnly = prev.filter(
            (t) => !loadedIds.has(t.id) && !deletedIdsRef.current.has(t.id),
          );
          return [...localOnly, ...filteredLoaded];
        });
        setActiveThreadId((prev) => {
          if (prev && !deletedIdsRef.current.has(prev)) {
            if (
              filteredLoaded.some((t) => t.id === prev) ||
              threadsRef.current.some((t) => t.id === prev)
            ) {
              return prev;
            }
          }
          return filteredLoaded[0]?.id || null;
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
    if (deletedIdsRef.current.has(thread.id)) {
      return;
    }
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
      if (deletedIdsRef.current.has(id)) return;
      setThreads((prev) => prev.map((t) => (t.id === id ? updater(t) : t)));
    },
    [],
  );

  const createNewThread = useCallback(
    async (
      title = 'New Agent Session',
      sessionConfig?: Partial<ThreadSessionConfig>,
    ): Promise<ChatThread | null> => {
      setThreadsError(null);
      try {
        const created = await createThread(title, sessionConfig);
        deletedIdsRef.current.delete(created.id);
        setThreads((prev) => [created, ...prev.filter((t) => t.id !== created.id)]);
        setActiveThreadId(created.id);
        return created;
      } catch (err) {
        setThreadsError(err instanceof Error ? err.message : 'Unable to create session.');
        return null;
      }
    },
    [],
  );

  const removeThread = useCallback(async (id: string) => {
    setThreadsError(null);
    deletedIdsRef.current.add(id);
    try {
      await deleteThread(id);
      const nextRemaining = threadsRef.current.filter((t) => t.id !== id);
      setThreads((prev) => prev.filter((t) => t.id !== id));
      setActiveThreadId((current) => (current === id ? nextRemaining[0]?.id || null : current));
    } catch (err) {
      deletedIdsRef.current.delete(id);
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
