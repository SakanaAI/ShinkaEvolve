import { useStore } from '@/store/useStore';
import { useMetaFiles, useMetaContent } from '@/hooks/usePrograms';
import { useEffect } from 'react';
import { Download } from 'lucide-react';
import { api } from '@/lib/api';

export function MetaAnalysis() {
  const { selectedDbPath, selectedMetaGeneration, setSelectedMetaGeneration } = useStore();
  const { data: metaFiles } = useMetaFiles(selectedDbPath);
  const { data: metaContent } = useMetaContent(selectedDbPath, selectedMetaGeneration);

  // Set initial generation when meta files load
  useEffect(() => {
    if (metaFiles && metaFiles.length > 0 && !selectedMetaGeneration) {
      setSelectedMetaGeneration(metaFiles[metaFiles.length - 1].generation);
    }
  }, [metaFiles, selectedMetaGeneration, setSelectedMetaGeneration]);

  const handleDownloadPdf = () => {
    if (selectedDbPath && selectedMetaGeneration) {
      const url = api.getMetaPdfUrl(selectedDbPath, selectedMetaGeneration);
      window.open(url, '_blank');
    }
  };

  if (!metaFiles || metaFiles.length === 0) {
    return (
      <div className="text-gray-500">
        No meta-analysis files available for this database.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Meta-Analysis Scratchpad</h3>
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium">Generation:</label>
          <select
            value={selectedMetaGeneration || ''}
            onChange={(e) => setSelectedMetaGeneration(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
          >
            {metaFiles.map((file) => (
              <option key={file.generation} value={file.generation}>
                {file.generation}
              </option>
            ))}
          </select>
          <button
            onClick={handleDownloadPdf}
            disabled={!selectedMetaGeneration}
            className="flex items-center gap-1 rounded bg-blue-600 px-3 py-1 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
          >
            <Download className="h-4 w-4" />
            Download PDF
          </button>
        </div>
      </div>

      {metaContent && (
        <div className="rounded border bg-white p-4">
          <pre className="whitespace-pre-wrap text-sm">{metaContent.content}</pre>
        </div>
      )}
    </div>
  );
}
