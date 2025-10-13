import { RefreshCw, Download, Clock } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { useDatabases, useRefreshPrograms } from '@/hooks/usePrograms';
import { useMemo } from 'react';

export function Header() {
  const { data: databases } = useDatabases();
  const refreshPrograms = useRefreshPrograms();

  const {
    selectedDbPath,
    setSelectedDbPath,
    selectedTask,
    setSelectedTask,
    autoRefreshEnabled,
    setAutoRefreshEnabled,
  } = useStore();

  // Group databases by task
  const taskGroups = useMemo(() => {
    if (!databases) return {};

    const groups: Record<string, typeof databases> = {};
    databases.forEach((db) => {
      const parts = db.path.split('/');
      const task = parts[0] || 'default';
      if (!groups[task]) {
        groups[task] = [];
      }
      groups[task].push(db);
    });

    return groups;
  }, [databases]);

  const tasks = Object.keys(taskGroups);
  const currentResults = selectedTask ? taskGroups[selectedTask] || [] : [];

  const handleRefresh = () => {
    if (selectedDbPath) {
      refreshPrograms(selectedDbPath);
    }
  };

  const handleToggleAutoRefresh = () => {
    setAutoRefreshEnabled(!autoRefreshEnabled);
  };

  return (
    <header className="flex items-center gap-4 border-b bg-white px-5 py-3 shadow-sm">
      <h1 className="text-xl font-semibold flex items-center gap-2">
        <span className="text-2xl">🎏</span>
        <span className="hidden sm:inline">
          <span className="text-blue-600">ShinkaEvolve</span>: Open-Ended Program Evolution
        </span>
        <span className="sm:hidden text-blue-600">ShinkaEvolve</span>
        <span className="text-2xl">🎏</span>
      </h1>

      <div className="ml-auto flex items-center gap-3">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700">Task:</label>
          <select
            value={selectedTask || ''}
            onChange={(e) => {
              setSelectedTask(e.target.value || null);
              setSelectedDbPath(null);
            }}
            className="rounded border border-gray-300 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="">Select a task...</option>
            {tasks.map((task) => (
              <option key={task} value={task}>
                {task}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700">Result:</label>
          <select
            value={selectedDbPath || ''}
            onChange={(e) => setSelectedDbPath(e.target.value || null)}
            disabled={!selectedTask}
            className="rounded border border-gray-300 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            <option value="">Select a result...</option>
            {currentResults.map((db) => (
              <option key={db.path} value={db.path}>
                {db.name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-1 border-l pl-3">
          <button
            onClick={handleRefresh}
            disabled={!selectedDbPath}
            className="rounded p-2 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Refresh data"
          >
            <RefreshCw className="h-4 w-4" />
          </button>

          <button
            onClick={handleToggleAutoRefresh}
            disabled={!selectedDbPath}
            className={`rounded p-2 transition-colors ${
              autoRefreshEnabled
                ? 'bg-blue-100 text-blue-600 hover:bg-blue-200'
                : 'hover:bg-gray-100'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
            title={autoRefreshEnabled ? 'Disable auto-refresh' : 'Enable auto-refresh (3s)'}
          >
            <Clock className="h-4 w-4" />
          </button>

          <button
            disabled={!selectedDbPath}
            className="rounded p-2 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Download top programs"
          >
            <Download className="h-4 w-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
