import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

export function useDatabases() {
  return useQuery({
    queryKey: ['databases'],
    queryFn: api.listDatabases,
    staleTime: 5000, // Consider data stale after 5 seconds
  });
}

export function usePrograms(dbPath: string | null) {
  return useQuery({
    queryKey: ['programs', dbPath],
    queryFn: () => (dbPath ? api.getPrograms(dbPath) : Promise.resolve([])),
    enabled: !!dbPath,
    staleTime: 5000,
    refetchInterval: false, // Manual control via auto-refresh
  });
}

export function useMetaFiles(dbPath: string | null) {
  return useQuery({
    queryKey: ['metaFiles', dbPath],
    queryFn: () => (dbPath ? api.getMetaFiles(dbPath) : Promise.resolve([])),
    enabled: !!dbPath,
  });
}

export function useMetaContent(dbPath: string | null, generation: number | null) {
  return useQuery({
    queryKey: ['metaContent', dbPath, generation],
    queryFn: () =>
      dbPath && generation !== null
        ? api.getMetaContent(dbPath, generation)
        : Promise.resolve(null),
    enabled: !!dbPath && generation !== null,
  });
}

// Hook to manually refresh programs data
export function useRefreshPrograms() {
  const queryClient = useQueryClient();

  return (dbPath: string) => {
    return queryClient.invalidateQueries({ queryKey: ['programs', dbPath] });
  };
}

// Hook for auto-refresh functionality
export function useAutoRefresh(
  dbPath: string | null,
  intervalMs: number,
  enabled: boolean
) {
  return useQuery({
    queryKey: ['programs', dbPath],
    queryFn: () => (dbPath ? api.getPrograms(dbPath) : Promise.resolve([])),
    enabled: enabled && !!dbPath,
    refetchInterval: enabled ? intervalMs : false,
  });
}
