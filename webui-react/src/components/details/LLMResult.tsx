import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';

export function LLMResult() {
  const { selectedDbPath, selectedNodeId } = useStore();
  const { data: programs } = usePrograms(selectedDbPath);

  const selectedProgram = programs?.find((p) => p.id === selectedNodeId);

  if (!selectedProgram) {
    return (
      <div className="text-gray-500">
        Select a node from the tree to view LLM results.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">LLM Generation Result</h3>

      <div>
        <h4 className="mb-2 font-semibold">Model Information</h4>
        <div className="rounded bg-gray-100 p-3 text-sm">
          <div className="flex justify-between">
            <span className="font-medium">Model:</span>
            <span className="font-mono">{selectedProgram.metadata.llm_model || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="font-medium">API Cost:</span>
            <span className="font-mono">
              ${(selectedProgram.metadata.api_cost as number)?.toFixed(6) || 'N/A'}
            </span>
          </div>
        </div>
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Patch Description</h4>
        <div className="rounded bg-gray-100 p-3 text-sm">
          {selectedProgram.metadata.patch_description || 'No description available.'}
        </div>
      </div>

      {selectedProgram.code_diff && (
        <div>
          <h4 className="mb-2 font-semibold">Generated Changes</h4>
          <pre className="rounded bg-gray-900 p-3 text-xs text-gray-100 overflow-auto max-h-60">
            {selectedProgram.code_diff}
          </pre>
        </div>
      )}
    </div>
  );
}
