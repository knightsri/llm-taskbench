/**
 * API client for LLM TaskBench backend.
 */

import axios from 'axios';
import type { Task, Evaluation, EvaluationProgress, UseCase, UseCaseDetail, GeneratedPrompts } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Use Cases API (folder-based)
export const usecasesApi = {
  async list(): Promise<UseCase[]> {
    const response = await api.get('/usecases/');
    return response.data;
  },

  async get(folderPath: string): Promise<UseCaseDetail> {
    const response = await api.get('/usecases/detail', { params: { path: folderPath } });
    return response.data;
  },

  async generatePrompts(folderPath: string, force: boolean = false): Promise<GeneratedPrompts> {
    const response = await api.post('/usecases/generate-prompts', { path: folderPath, force });
    return response.data;
  },

  async getPrompts(folderPath: string): Promise<GeneratedPrompts | null> {
    try {
      const response = await api.get('/usecases/prompts', { params: { path: folderPath } });
      return response.data;
    } catch (err) {
      return null;
    }
  },

  async runEvaluation(data: {
    usecase_path: string;
    data_file?: string;
    models: string[];
    skip_judge?: boolean;
    regenerate_prompts?: boolean;
  }): Promise<Evaluation> {
    const response = await api.post('/usecases/evaluate', data);
    return response.data;
  },
};

// Tasks API (legacy)
export const tasksApi = {
  async create(data: {
    name: string;
    description: string;
    domain?: string;
    input_format?: string;
    output_format?: string;
    constraints?: Record<string, any>;
    evaluation_criteria?: string[];
  }): Promise<Task> {
    const response = await api.post('/tasks/', data);
    return response.data;
  },

  async list(): Promise<Task[]> {
    const response = await api.get('/tasks/');
    return response.data;
  },

  async get(id: string): Promise<Task> {
    const response = await api.get(`/tasks/${id}`);
    return response.data;
  },

  async update(id: string, data: Partial<Task>): Promise<Task> {
    const response = await api.patch(`/tasks/${id}`, data);
    return response.data;
  },

  async delete(id: string): Promise<void> {
    await api.delete(`/tasks/${id}`);
  },

  async regenerateQualityChecks(id: string): Promise<Task> {
    const response = await api.post(`/tasks/${id}/regenerate-quality-checks`);
    return response.data;
  },
};

// Evaluations API
export const evaluationsApi = {
  async create(data: {
    task_id: string;
    models: Array<{
      id: string;
      endpoint?: string;
      api_key?: string;
      provider?: string;
    }>;
    metrics?: string[];
    consistency_runs?: number;
  }): Promise<Evaluation> {
    const response = await api.post('/evaluations/', data);
    return response.data;
  },

  async list(taskId?: string): Promise<Evaluation[]> {
    const params = taskId ? { task_id: taskId } : {};
    const response = await api.get('/evaluations/', { params });
    return response.data;
  },

  async get(id: string): Promise<Evaluation> {
    const response = await api.get(`/evaluations/${id}`);
    return response.data;
  },

  async getProgress(id: string): Promise<EvaluationProgress> {
    const response = await api.get(`/evaluations/${id}/progress`);
    return response.data;
  },

  async delete(id: string): Promise<void> {
    await api.delete(`/evaluations/${id}`);
  },
};

export default api;
