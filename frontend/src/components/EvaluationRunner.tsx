import { useState } from 'react';
import { evaluationsApi } from '../api/client';
import type { Task, Evaluation, ModelConfig } from '../types';

interface EvaluationRunnerProps {
  task: Task;
  onEvaluationStarted: (evaluation: Evaluation) => void;
}

const POPULAR_MODELS = [
  { id: 'anthropic/claude-sonnet-4.5', name: 'Claude Sonnet 4.5', provider: 'openrouter' },
  { id: 'openai/gpt-4o', name: 'GPT-4o', provider: 'openrouter' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini', provider: 'openrouter' },
  { id: 'google/gemini-2.0-flash-exp:free', name: 'Gemini 2.0 Flash', provider: 'openrouter' },
  { id: 'qwen/qwen-2.5-72b-instruct', name: 'Qwen 2.5 72B', provider: 'openrouter' },
];

export default function EvaluationRunner({ task, onEvaluationStarted }: EvaluationRunnerProps) {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId]
    );
  };

  const handleStartEvaluation = async () => {
    if (selectedModels.length === 0) {
      setError('Please select at least one model');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const models: ModelConfig[] = selectedModels.map((modelId) => ({
        id: modelId,
        endpoint: 'https://openrouter.ai/api/v1',
        provider: 'openrouter',
      }));

      const evaluation = await evaluationsApi.create({
        task_id: task.id,
        models,
        metrics: ['accuracy', 'hallucination', 'completeness', 'cost', 'instruction_following'],
      });

      onEvaluationStarted(evaluation);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start evaluation');
      console.error('Error starting evaluation:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold">Run Evaluation</h2>
        <p className="text-gray-600 mt-2">Task: {task.name}</p>
      </div>

      {/* Model Selection */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3">Select Models to Evaluate</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {POPULAR_MODELS.map((model) => (
            <label
              key={model.id}
              className="flex items-center p-4 border rounded-lg cursor-pointer hover:bg-gray-50"
            >
              <input
                type="checkbox"
                checked={selectedModels.includes(model.id)}
                onChange={() => toggleModel(model.id)}
                className="mr-3 h-4 w-4 text-blue-600"
              />
              <div>
                <div className="font-medium">{model.name}</div>
                <div className="text-sm text-gray-500">{model.id}</div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Cost Estimate */}
      {selectedModels.length > 0 && (
        <div className="mb-6 p-4 bg-blue-50 rounded-md">
          <h4 className="font-semibold mb-2">Estimated Cost</h4>
          <p className="text-2xl font-bold text-blue-600">
            ~${(selectedModels.length * 0.50).toFixed(2)}
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Based on {selectedModels.length} model{selectedModels.length !== 1 ? 's' : ''}
          </p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Start Button */}
      <button
        onClick={handleStartEvaluation}
        disabled={loading || selectedModels.length === 0}
        className="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
      >
        {loading ? 'Starting Evaluation...' : `Start Evaluation (${selectedModels.length} models)`}
      </button>

      {/* Info */}
      <p className="mt-4 text-sm text-gray-500 text-center">
        Evaluation will run asynchronously. You can view progress in the Results tab.
      </p>
    </div>
  );
}
