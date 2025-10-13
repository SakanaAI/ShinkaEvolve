import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';

export function CodeDiff() {
  const { selectedDbPath, selectedNodeId } = useStore();
  const { data: programs } = usePrograms(selectedDbPath);

  const selectedProgram = programs?.find((p) => p.id === selectedNodeId);

  if (!selectedProgram) {
    return (
      <div className="text-gray-500">
        Select a node from the tree to view code diff.
      </div>
    );
  }

  if (!selectedProgram.code_diff) {
    return (
      <div className="text-gray-500">
        No diff available for this program.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h3 className="text-lg font-semibold">Code Diff</h3>
      <pre className="rounded bg-gray-100 p-4 text-xs overflow-auto max-h-[calc(100vh-200px)]">
        {selectedProgram.code_diff}
      </pre>
    </div>
  );
}
