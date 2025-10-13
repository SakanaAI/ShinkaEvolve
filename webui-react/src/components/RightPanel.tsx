import { useStore } from '@/store/useStore';
import { AgentInfo } from './details/AgentInfo';
import { ParetoFront } from './details/ParetoFront';
import { MetaAnalysis } from './details/MetaAnalysis';
import { NodeDetails } from './details/NodeDetails';
import { AgentCode } from './details/AgentCode';
import { CodeDiff } from './details/CodeDiff';
import { LogOutput } from './details/LogOutput';
import { LLMResult } from './details/LLMResult';
import type { RightPanelTab } from '@/types/program';

const tabs: { id: RightPanelTab; label: string }[] = [
  { id: 'agent-info', label: 'Meta' },
  { id: 'pareto-front', label: 'Pareto Front' },
  { id: 'meta-analysis', label: 'Scratchpad' },
  { id: 'node-details', label: 'Node' },
  { id: 'agent-code', label: 'Code' },
  { id: 'code-diff', label: 'Diff' },
  { id: 'log-output', label: 'Evaluation' },
  { id: 'llm-result', label: 'LLM Result' },
];

export function RightPanel() {
  const { rightPanelTab, setRightPanelTab } = useStore();

  return (
    <div className="flex h-full flex-col bg-white">
      {/* Tabs */}
      <div className="flex border-b bg-gray-50 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setRightPanelTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors ${
              rightPanelTab === tab.id
                ? 'border-b-2 border-blue-500 bg-white text-blue-600'
                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {rightPanelTab === 'agent-info' && <AgentInfo />}
        {rightPanelTab === 'pareto-front' && <ParetoFront />}
        {rightPanelTab === 'meta-analysis' && <MetaAnalysis />}
        {rightPanelTab === 'node-details' && <NodeDetails />}
        {rightPanelTab === 'agent-code' && <AgentCode />}
        {rightPanelTab === 'code-diff' && <CodeDiff />}
        {rightPanelTab === 'log-output' && <LogOutput />}
        {rightPanelTab === 'llm-result' && <LLMResult />}
      </div>
    </div>
  );
}
