/**
 * API client for LLM TaskBench backend.
 */

import axios from 'axios';
import type { Task, Evaluation, EvaluationProgress } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Tasks API
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
