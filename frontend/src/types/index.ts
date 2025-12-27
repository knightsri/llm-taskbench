/**
 * TypeScript types for LLM TaskBench frontend.
 */

export interface QualityCheck {
  name: string;
  description: string;
  validation_function: string;
  severity: 'critical' | 'warning' | 'info';
}

export interface Task {
  id: string;
  name: string;
  description: string;
  domain?: string;
  input_format: string;
  output_format: string;
  gold_data?: any;
  quality_checks: QualityCheck[];
  constraints: Record<string, any>;
  evaluation_criteria: string[];
  created_at: string;
  updated_at: string;
}

/**
 * Folder-based use case types
 */
export interface DataFile {
  path: string;
  name: string;
  extension: string;
  content_preview: string;
  size_bytes: number;
  line_count: number;
}

export interface GroundTruthFile {
  path: string;
  name: string;
  extension: string;
  format_type: string;
}

export interface DataGroundTruthPair {
  data_file: DataFile;
  ground_truth_file: GroundTruthFile;
  match_pattern: string;
}

export interface UseCase {
  path: string;
  folder_name: string;
  name: string;
  difficulty?: string;
  capability?: string;
  goal?: string;
  llm_notes?: string;
  output_format?: string;
  data_files: number;
  ground_truth_files: number;
  matched_pairs: number;
  error?: string;
}

export interface UseCaseDetail {
  folder_path: string;
  name: string;
  goal: string;
  difficulty: string;
  primary_capability: string;
  token_range: string;
  llm_notes: string;
  expected_output_schema: string;
  output_format: string;
  quality_criteria: string;
  edge_cases: string[];
  data_files: DataFile[];
  ground_truth_files: GroundTruthFile[];
  matched_pairs: DataGroundTruthPair[];
}

export interface GeneratedPrompts {
  analysis: {
    transformation_type: string;
    key_fields: string[];
    quality_indicators: string[];
    comparison_strategy: string;
  };
  task_prompt: string;
  judge_prompt: string;
  rubric: {
    critical_requirements: Array<{
      name: string;
      description: string;
      derived_from: string;
      penalty_per_violation: number;
    }>;
    compliance_checks: Array<{
      check: string;
      condition: string;
      severity: string;
      penalty: number;
    }>;
    weights: {
      accuracy: number;
      format: number;
      compliance: number;
    };
    scoring_instructions: string;
  };
}

export interface ModelConfig {
  id: string;
  endpoint: string;
  api_key?: string;
  provider: 'openrouter' | 'anthropic' | 'openai' | 'custom';
}

export interface ModelResult {
  id: string;
  model_id: string;
  output?: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  accuracy?: number;
  hallucination_rate?: number;
  completeness?: number;
  cost: number;
  consistency_std?: number;
  instruction_following?: number;
  quality_violations: string[];
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  latency_ms?: number;
  created_at: string;
  completed_at?: string;
  error_message?: string;
}

export interface Evaluation {
  id: string;
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  models: ModelConfig[];
  metrics: string[];
  estimated_cost: number;
  actual_cost: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  results: ModelResult[];
}

export interface EvaluationProgress {
  evaluation_id: string;
  status: string;
  progress_percent: number;
  completed_models: number;
  total_models: number;
  current_cost: number;
  estimated_remaining_cost: number;
  message: string;
}
