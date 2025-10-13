import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';
import { Copy, Download } from 'lucide-react';
import { copyToClipboard, downloadFile } from '@/lib/utils';

export function AgentCode() {
  const { selectedDbPath, selectedNodeId } = useStore();
  const { data: programs } = usePrograms(selectedDbPath);

  const selectedProgram = programs?.find((p) => p.id === selectedNodeId);

  const handleCopy = () => {
    if (selectedProgram) {
      copyToClipboard(selectedProgram.code);
    }
  };

  const handleDownload = () => {
    if (selectedProgram) {
      const filename = `${selectedProgram.metadata.patch_name || selectedProgram.id}.py`;
      downloadFile(selectedProgram.code, filename, 'text/plain');
    }
  };

  if (!selectedProgram) {
    return (
      <div className="text-gray-500">
        Select a node from the tree to view code.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Program Code</h3>
        <div className="flex gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 rounded bg-gray-200 px-3 py-1 text-sm hover:bg-gray-300"
          >
            <Copy className="h-4 w-4" />
            Copy
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1 rounded bg-gray-200 px-3 py-1 text-sm hover:bg-gray-300"
          >
            <Download className="h-4 w-4" />
            Download
          </button>
        </div>
      </div>

      <pre className="rounded bg-gray-900 p-4 text-sm text-gray-100 overflow-auto max-h-[calc(100vh-200px)]">
        <code>{selectedProgram.code}</code>
      </pre>
    </div>
  );
}
