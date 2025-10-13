import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';

export function NodeDetails() {
  const { selectedDbPath, selectedNodeId } = useStore();
  const { data: programs } = usePrograms(selectedDbPath);

  const selectedProgram = programs?.find((p) => p.id === selectedNodeId);

  if (!selectedProgram) {
    return (
      <div className="text-gray-500">
        Select a node from the tree to view details.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Node Details</h3>
      <pre className="rounded bg-gray-100 p-4 text-xs overflow-auto">
        {JSON.stringify(selectedProgram, null, 2)}
      </pre>
    </div>
  );
}
