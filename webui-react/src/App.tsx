import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { Header } from '@/components/Header';
import { LeftPanel } from '@/components/LeftPanel';
import { RightPanel } from '@/components/RightPanel';

function App() {
  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-gray-50">
      <Header />
      <PanelGroup direction="horizontal" className="flex-1">
        <Panel defaultSize={50} minSize={20} maxSize={80}>
          <LeftPanel />
        </Panel>
        <PanelResizeHandle className="w-2 bg-gray-300 hover:bg-blue-500 transition-colors cursor-col-resize relative group">
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-0.5 h-8 bg-gray-600 rounded group-hover:bg-white" />
        </PanelResizeHandle>
        <Panel defaultSize={50} minSize={20}>
          <RightPanel />
        </Panel>
      </PanelGroup>
    </div>
  );
}

export default App;
