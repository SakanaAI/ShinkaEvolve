import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';

export function LogOutput() {
  const { selectedDbPath, selectedNodeId } = useStore();
  const { data: programs } = usePrograms(selectedDbPath);

  const selectedProgram = programs?.find((p) => p.id === selectedNodeId);

  if (!selectedProgram) {
    return (
      <div className="text-gray-500">
        Select a node from the tree to view evaluation logs.
      </div>
    );
  }

  const feedback = Array.isArray(selectedProgram.text_feedback)
    ? selectedProgram.text_feedback.join('\n')
    : selectedProgram.text_feedback;

  return (
    <div className="space-y-2">
      <h3 className="text-lg font-semibold">Evaluation Output</h3>
      <pre className="rounded bg-gray-900 p-4 text-sm text-green-400 overflow-auto max-h-[calc(100vh-200px)] font-mono">
        {feedback || 'No evaluation output available.'}
      </pre>
    </div>
  );
}
