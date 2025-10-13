// TypeScript types matching Python Program dataclass
export interface Program {
  // Program identification
  id: string;
  code: string;
  language: string;

  // Evolution information
  parent_id: string | null;
  archive_inspiration_ids: string[];
  top_k_inspiration_ids: string[];
  island_idx: number | null;
  generation: number;
  timestamp: number;
  code_diff: string | null;

  // Performance metrics
  combined_score: number;
  public_metrics: Record<string, number | string>;
  private_metrics: Record<string, number | string>;
  text_feedback: string | string[];
  correct: boolean;
  children_count: number;

  // Derived features
  complexity: number;
  embedding: number[];
  embedding_pca_2d: number[];
  embedding_pca_3d: number[];
  embedding_cluster_id: number | null;

  // Migration history
  migration_history: MigrationEvent[];

  // Metadata
  metadata: ProgramMetadata;

  // Archive status
  in_archive: boolean;
}

export interface MigrationEvent {
  generation: number;
  from_island: number;
  to_island: number;
  timestamp: number;
}

export interface ProgramMetadata {
  patch_name?: string;
  patch_description?: string;
  patch_type?: string;
  llm_model?: string;
  api_cost?: number;
  code_analysis_metrics?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DatabaseInfo {
  path: string;
  name: string;
  actual_path: string;
}

export interface MetaFile {
  generation: number;
  filename: string;
  path: string;
}

export interface MetaContent {
  generation: number;
  filename: string;
  content: string;
}

export type LeftPanelTab =
  | 'tree-view'
  | 'table-view'
  | 'metrics-view'
  | 'embeddings-view'
  | 'clusters-view'
  | 'islands-view'
  | 'model-posteriors-view'
  | 'best-path-view';

export type RightPanelTab =
  | 'agent-info'
  | 'pareto-front'
  | 'meta-analysis'
  | 'node-details'
  | 'agent-code'
  | 'code-diff'
  | 'log-output'
  | 'llm-result';

export interface TreeNode {
  id: string;
  parentId: string | null;
  generation: number;
  score: number;
  correct: boolean;
  islandIdx: number | null;
  patchType: string;
  children: TreeNode[];
}
