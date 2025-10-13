import { useMemo, useState } from 'react';
import { useStore } from '@/store/useStore';
import { usePrograms } from '@/hooks/usePrograms';
import { formatNumber } from '@/lib/utils';
import type { Program } from '@/types/program';

type SortField = 'generation' | 'combined_score' | 'complexity' | 'api_cost' | 'island_idx';
type SortDirection = 'asc' | 'desc';

export function TableView() {
  const { selectedDbPath, showIncorrectPrograms, setShowIncorrectPrograms, setSelectedNodeId } = useStore();
  const { data: programs, isLoading, error } = usePrograms(selectedDbPath);

  const [sortField, setSortField] = useState<SortField>('generation');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const filteredAndSortedPrograms = useMemo(() => {
    if (!programs) return [];

    let filtered = programs;

    // Filter out incorrect programs if needed
    if (!showIncorrectPrograms) {
      filtered = programs.filter((p) => p.correct);
    }

    // Sort
    const sorted = [...filtered].sort((a, b) => {
      let aVal: number | string = 0;
      let bVal: number | string = 0;

      switch (sortField) {
        case 'generation':
          aVal = a.generation;
          bVal = b.generation;
          break;
        case 'combined_score':
          aVal = a.combined_score;
          bVal = b.combined_score;
          break;
        case 'complexity':
          aVal = a.complexity;
          bVal = b.complexity;
          break;
        case 'api_cost':
          aVal = a.metadata.api_cost || 0;
          bVal = b.metadata.api_cost || 0;
          break;
        case 'island_idx':
          aVal = a.island_idx || 0;
          bVal = b.island_idx || 0;
          break;
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return sorted;
  }, [programs, showIncorrectPrograms, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const handleRowClick = (program: Program) => {
    setSelectedNodeId(program.id);
  };

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        Loading programs...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center text-red-500">
        Error loading programs: {error.message}
      </div>
    );
  }

  if (!programs || programs.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        No programs to display.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Controls */}
      <div className="border-b bg-gray-50 p-3">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={showIncorrectPrograms}
            onChange={(e) => setShowIncorrectPrograms(e.target.checked)}
            className="rounded"
          />
          Show incorrect programs
        </label>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 bg-gray-100">
            <tr>
              <SortableHeader
                label="Rank"
                field="generation"
                currentField={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                label="Gen"
                field="generation"
                currentField={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <th className="border-b px-3 py-2 text-left font-medium">Archive</th>
              <th className="border-b px-3 py-2 text-left font-medium">Patch Name</th>
              <th className="border-b px-3 py-2 text-left font-medium">Type</th>
              <SortableHeader
                label="Island"
                field="island_idx"
                currentField={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                label="Score"
                field="combined_score"
                currentField={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                label="API Cost"
                field="api_cost"
                currentField={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <SortableHeader
                label="Complexity"
                field="complexity"
                currentField={sortField}
                direction={sortDirection}
                onSort={handleSort}
              />
              <th className="border-b px-3 py-2 text-left font-medium">Model</th>
            </tr>
          </thead>
          <tbody>
            {filteredAndSortedPrograms.map((program, index) => (
              <tr
                key={program.id}
                onClick={() => handleRowClick(program)}
                className={`cursor-pointer hover:bg-blue-50 ${
                  !program.correct ? 'bg-red-50' : ''
                }`}
              >
                <td className="border-b px-3 py-2">{index + 1}</td>
                <td className="border-b px-3 py-2">{program.generation}</td>
                <td className="border-b px-3 py-2">{program.in_archive ? '✓' : ''}</td>
                <td className="border-b px-3 py-2 font-mono text-xs">
                  {program.metadata.patch_name || 'N/A'}
                </td>
                <td className="border-b px-3 py-2">{program.metadata.patch_type || 'N/A'}</td>
                <td className="border-b px-3 py-2">{program.island_idx ?? 'N/A'}</td>
                <td className="border-b px-3 py-2">{formatNumber(program.combined_score, 4)}</td>
                <td className="border-b px-3 py-2">{formatNumber(program.metadata.api_cost as number, 6)}</td>
                <td className="border-b px-3 py-2">{formatNumber(program.complexity, 1)}</td>
                <td className="border-b px-3 py-2 text-xs">{program.metadata.llm_model || 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SortableHeader({
  label,
  field,
  currentField,
  direction,
  onSort,
}: {
  label: string;
  field: SortField;
  currentField: SortField;
  direction: SortDirection;
  onSort: (field: SortField) => void;
}) {
  return (
    <th
      onClick={() => onSort(field)}
      className="cursor-pointer border-b px-3 py-2 text-left font-medium hover:bg-gray-200"
    >
      <div className="flex items-center gap-1">
        {label}
        {currentField === field && (
          <span className="text-xs">{direction === 'asc' ? '↑' : '↓'}</span>
        )}
      </div>
    </th>
  );
}
