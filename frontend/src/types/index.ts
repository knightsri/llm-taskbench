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
