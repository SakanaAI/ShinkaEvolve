import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';
import { formatNumber, formatTimestamp } from '@/lib/utils';

export function AgentInfo() {
  const { selectedDbPath, selectedNodeId } = useStore();
  const { data: programs } = usePrograms(selectedDbPath);

  const selectedProgram = programs?.find((p) => p.id === selectedNodeId);

  if (!selectedProgram) {
    return (
      <div className="text-gray-500">
        Select a node from the tree to view agent information.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Program Metadata</h3>

      <div className="space-y-2">
        <InfoRow label="ID" value={selectedProgram.id} />
        <InfoRow label="Generation" value={selectedProgram.generation.toString()} />
        <InfoRow label="Timestamp" value={formatTimestamp(selectedProgram.timestamp)} />
        <InfoRow label="Score" value={formatNumber(selectedProgram.combined_score, 6)} />
        <InfoRow label="Correct" value={selectedProgram.correct ? 'Yes' : 'No'} />
        <InfoRow label="Island" value={selectedProgram.island_idx?.toString() || 'N/A'} />
        <InfoRow label="Complexity" value={formatNumber(selectedProgram.complexity, 2)} />
        <InfoRow label="In Archive" value={selectedProgram.in_archive ? 'Yes' : 'No'} />
        <InfoRow label="Children Count" value={selectedProgram.children_count.toString()} />
      </div>

      <div>
        <h4 className="mb-2 font-semibold">Patch Information</h4>
        <div className="space-y-1 text-sm">
          <InfoRow label="Name" value={selectedProgram.metadata.patch_name || 'N/A'} />
          <InfoRow label="Type" value={selectedProgram.metadata.patch_type || 'N/A'} />
          <InfoRow label="LLM Model" value={selectedProgram.metadata.llm_model || 'N/A'} />
          <InfoRow
            label="API Cost"
            value={formatNumber(selectedProgram.metadata.api_cost as number, 6)}
          />
        </div>
      </div>

      {Object.keys(selectedProgram.public_metrics).length > 0 && (
        <div>
          <h4 className="mb-2 font-semibold">Public Metrics</h4>
          <div className="space-y-1 text-sm">
            {Object.entries(selectedProgram.public_metrics).map(([key, value]) => (
              <InfoRow key={key} label={key} value={String(value)} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b pb-1">
      <span className="font-medium text-gray-700">{label}:</span>
      <span className="font-mono text-sm text-gray-900">{value}</span>
    </div>
  );
}
