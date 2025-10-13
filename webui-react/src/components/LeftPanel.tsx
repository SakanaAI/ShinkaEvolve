import { useStore } from '@/store/useStore';
import { TreeView } from './views/TreeView';
import { TableView } from './views/TableView';
import { MetricsView } from './views/MetricsView';
import { EmbeddingsView } from './views/EmbeddingsView';
import { ClustersView } from './views/ClustersView';
import { IslandsView } from './views/IslandsView';
import { ModelPosteriorsView } from './views/ModelPosteriorsView';
import { BestPathView } from './views/BestPathView';
import type { LeftPanelTab } from '@/types/program';

const tabs: { id: LeftPanelTab; label: string }[] = [
  { id: 'tree-view', label: 'Tree' },
  { id: 'table-view', label: 'Programs' },
  { id: 'metrics-view', label: 'Metrics' },
  { id: 'embeddings-view', label: 'Embeddings' },
  { id: 'clusters-view', label: 'Clusters' },
  { id: 'islands-view', label: 'Islands' },
  { id: 'model-posteriors-view', label: 'LLM Posterior' },
  { id: 'best-path-view', label: 'Path → Best' },
];

export function LeftPanel() {
  const { leftPanelTab, setLeftPanelTab } = useStore();

  return (
    <div className="flex h-full flex-col bg-white">
      {/* Tabs */}
      <div className="flex border-b bg-gray-50 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setLeftPanelTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors ${
              leftPanelTab === tab.id
                ? 'border-b-2 border-blue-500 bg-white text-blue-600'
                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {leftPanelTab === 'tree-view' && <TreeView />}
        {leftPanelTab === 'table-view' && <TableView />}
        {leftPanelTab === 'metrics-view' && <MetricsView />}
        {leftPanelTab === 'embeddings-view' && <EmbeddingsView />}
        {leftPanelTab === 'clusters-view' && <ClustersView />}
        {leftPanelTab === 'islands-view' && <IslandsView />}
        {leftPanelTab === 'model-posteriors-view' && <ModelPosteriorsView />}
        {leftPanelTab === 'best-path-view' && <BestPathView />}
      </div>
    </div>
  );
}
