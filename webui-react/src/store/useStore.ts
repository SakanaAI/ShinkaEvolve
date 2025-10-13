import { create } from 'zustand';
import type { LeftPanelTab, RightPanelTab } from '@/types/program';

interface AppState {
  // Database selection
  selectedDbPath: string | null;
  selectedTask: string | null;
  setSelectedDbPath: (path: string | null) => void;
  setSelectedTask: (task: string | null) => void;

  // Tab selection
  leftPanelTab: LeftPanelTab;
  rightPanelTab: RightPanelTab;
  setLeftPanelTab: (tab: LeftPanelTab) => void;
  setRightPanelTab: (tab: RightPanelTab) => void;

  // Selected program node
  selectedNodeId: string | null;
  setSelectedNodeId: (id: string | null) => void;

  // Auto-refresh
  autoRefreshEnabled: boolean;
  autoRefreshInterval: number;
  setAutoRefreshEnabled: (enabled: boolean) => void;
  setAutoRefreshInterval: (interval: number) => void;

  // UI preferences
  showIncorrectPrograms: boolean;
  setShowIncorrectPrograms: (show: boolean) => void;

  // Meta analysis
  selectedMetaGeneration: number | null;
  setSelectedMetaGeneration: (generation: number | null) => void;
}

export const useStore = create<AppState>((set) => ({
  // Database selection
  selectedDbPath: null,
  selectedTask: null,
  setSelectedDbPath: (path) => set({ selectedDbPath: path }),
  setSelectedTask: (task) => set({ selectedTask: task }),

  // Tab selection
  leftPanelTab: 'tree-view',
  rightPanelTab: 'agent-info',
  setLeftPanelTab: (tab) => set({ leftPanelTab: tab }),
  setRightPanelTab: (tab) => set({ rightPanelTab: tab }),

  // Selected program node
  selectedNodeId: null,
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  // Auto-refresh
  autoRefreshEnabled: false,
  autoRefreshInterval: 3000,
  setAutoRefreshEnabled: (enabled) => set({ autoRefreshEnabled: enabled }),
  setAutoRefreshInterval: (interval) => set({ autoRefreshInterval: interval }),

  // UI preferences
  showIncorrectPrograms: false,
  setShowIncorrectPrograms: (show) => set({ showIncorrectPrograms: show }),

  // Meta analysis
  selectedMetaGeneration: null,
  setSelectedMetaGeneration: (generation) => set({ selectedMetaGeneration: generation }),
}));
