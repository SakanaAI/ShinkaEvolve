import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';

export function MetricsView() {
  const { selectedDbPath } = useStore();
  const { data: programs } = usePrograms(selectedDbPath);

  if (!programs || programs.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        No data to display. Select a database to view metrics over generations.
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto p-4">
      <h3 className="mb-4 text-lg font-semibold">Metrics Over Generations</h3>
      <div className="text-gray-600">
        Metrics visualization will be implemented with Recharts.
        <br />
        Will show: combined_score, complexity, API cost trends over generations.
      </div>
    </div>
  );
}
