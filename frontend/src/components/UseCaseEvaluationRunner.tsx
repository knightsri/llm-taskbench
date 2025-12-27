import { useState } from 'react';
import { usecasesApi } from '../api/client';
import type { UseCaseDetail, GeneratedPrompts, Evaluation, ModelConfig } from '../types';

interface UseCaseEvaluationRunnerProps {
  usecase: UseCaseDetail;
  prompts: GeneratedPrompts | null;
  onEvaluationStarted: (evaluation: Evaluation) => void;
  onBack: () => void;
}

const POPULAR_MODELS = [
  { id: 'anthropic/claude-sonnet-4', name: 'Claude Sonnet 4', provider: 'openrouter' },
  { id: 'openai/gpt-4o', name: 'GPT-4o', provider: 'openrouter' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini', provider: 'openrouter' },
  { id: 'google/gemini-2.0-flash-exp:free', name: 'Gemini 2.0 Flash', provider: 'openrouter' },
  { id: 'qwen/qwen-2.5-72b-instruct', name: 'Qwen 2.5 72B', provider: 'openrouter' },
  { id: 'meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B', provider: 'openrouter' },
];

export default function UseCaseEvaluationRunner({
  usecase,
  prompts,
  onEvaluationStarted,
  onBack,
}: UseCaseEvaluationRunnerProps) {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedDataFile, setSelectedDataFile] = useState<string | undefined>(
    usecase.matched_pairs.length > 0 ? usecase.matched_pairs[0].data_file.path : undefined
  );
  const [skipJudge, setSkipJudge] = useState(false);
  const [regeneratePrompts, setRegeneratePrompts] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId]
    );
  };

  const selectAll = () => {
    setSelectedModels(POPULAR_MODELS.map((m) => m.id));
  };

  const clearAll = () => {
    setSelectedModels([]);
  };

  const handleStartEvaluation = async () => {
    if (selectedModels.length === 0) {
      setError('Please select at least one model');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const evaluation = await usecasesApi.runEvaluation({
        usecase_path: usecase.folder_path,
        data_file: selectedDataFile,
        models: selectedModels,
        skip_judge: skipJudge,
        regenerate_prompts: regeneratePrompts,
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
    <div className="space-y-6">
      {/* Use Case Summary */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-2xl font-bold">{usecase.name}</h2>
            <p className="text-gray-600 mt-1">{usecase.goal.slice(0, 200)}...</p>
          </div>
          <button
            onClick={onBack}
            className="text-gray-500 hover:text-gray-700"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {usecase.difficulty && (
            <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-sm">
              {usecase.difficulty}
            </span>
          )}
          <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm">
            {usecase.output_format.toUpperCase()}
          </span>
          <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">
            {usecase.matched_pairs.length} test pairs
          </span>
        </div>
      </div>

      {/* Data File Selection */}
      {usecase.matched_pairs.length > 1 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-3">Select Data File</h3>
          <p className="text-sm text-gray-600 mb-4">
            Choose which data file to use for evaluation. Each file has a corresponding ground truth.
          </p>
          <div className="space-y-2">
            {usecase.matched_pairs.map((pair) => (
              <label
                key={pair.data_file.path}
                className="flex items-center p-3 border rounded-lg cursor-pointer hover:bg-gray-50"
              >
                <input
                  type="radio"
                  name="dataFile"
                  checked={selectedDataFile === pair.data_file.path}
                  onChange={() => setSelectedDataFile(pair.data_file.path)}
                  className="mr-3 h-4 w-4 text-blue-600"
                />
                <div className="flex-1">
                  <div className="font-medium">{pair.data_file.name}{pair.data_file.extension}</div>
                  <div className="text-sm text-gray-500">
                    {pair.data_file.line_count} lines | Ground truth: {pair.ground_truth_file.name}{pair.ground_truth_file.extension}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Model Selection */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Select Models to Evaluate</h3>
          <div className="flex gap-2">
            <button
              onClick={selectAll}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              Select All
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={clearAll}
              className="text-sm text-gray-600 hover:text-gray-800"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {POPULAR_MODELS.map((model) => (
            <label
              key={model.id}
              className={`flex items-center p-4 border-2 rounded-lg cursor-pointer transition-all ${
                selectedModels.includes(model.id)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <input
                type="checkbox"
                checked={selectedModels.includes(model.id)}
                onChange={() => toggleModel(model.id)}
                className="mr-3 h-4 w-4 text-blue-600"
              />
              <div>
                <div className="font-medium">{model.name}</div>
                <div className="text-xs text-gray-500">{model.id}</div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Options */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Evaluation Options</h3>
        <div className="space-y-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={skipJudge}
              onChange={(e) => setSkipJudge(e.target.checked)}
              className="mr-3 h-4 w-4 text-blue-600"
            />
            <div>
              <span className="font-medium">Skip Judge Evaluation</span>
              <p className="text-sm text-gray-500">Only run models, don't score outputs</p>
            </div>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={regeneratePrompts}
              onChange={(e) => setRegeneratePrompts(e.target.checked)}
              className="mr-3 h-4 w-4 text-blue-600"
            />
            <div>
              <span className="font-medium">Regenerate Prompts</span>
              <p className="text-sm text-gray-500">Force regenerate task and judge prompts</p>
            </div>
          </label>
        </div>
      </div>

      {/* Prompts Preview */}
      {prompts && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Generated Prompts Preview</h3>

          <div className="space-y-4">
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Task Prompt</h4>
              <pre className="bg-gray-50 p-3 rounded text-sm overflow-x-auto max-h-40 overflow-y-auto">
                {prompts.task_prompt.slice(0, 500)}...
              </pre>
            </div>

            <div>
              <h4 className="font-medium text-gray-700 mb-2">Scoring Weights</h4>
              <div className="flex gap-4">
                <span className="px-3 py-1 bg-blue-50 text-blue-700 rounded">
                  Accuracy: {prompts.rubric.weights.accuracy}%
                </span>
                <span className="px-3 py-1 bg-green-50 text-green-700 rounded">
                  Format: {prompts.rubric.weights.format}%
                </span>
                <span className="px-3 py-1 bg-purple-50 text-purple-700 rounded">
                  Compliance: {prompts.rubric.weights.compliance}%
                </span>
              </div>
            </div>

            <div>
              <h4 className="font-medium text-gray-700 mb-2">
                Compliance Checks ({prompts.rubric.compliance_checks.length})
              </h4>
              <div className="space-y-1">
                {prompts.rubric.compliance_checks.slice(0, 3).map((check, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-sm">
                    <span className={`px-1.5 py-0.5 rounded text-xs ${
                      check.severity === 'CRITICAL'
                        ? 'bg-red-100 text-red-700'
                        : check.severity === 'HIGH'
                        ? 'bg-orange-100 text-orange-700'
                        : 'bg-yellow-100 text-yellow-700'
                    }`}>
                      {check.severity}
                    </span>
                    <span>{check.check}</span>
                  </div>
                ))}
                {prompts.rubric.compliance_checks.length > 3 && (
                  <p className="text-sm text-gray-500">
                    +{prompts.rubric.compliance_checks.length - 3} more checks
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Cost Estimate */}
      {selectedModels.length > 0 && (
        <div className="bg-blue-50 rounded-lg p-6">
          <h4 className="font-semibold mb-2">Estimated Cost</h4>
          <p className="text-2xl font-bold text-blue-600">
            ~${(selectedModels.length * 0.25).toFixed(2)}
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Based on {selectedModels.length} model{selectedModels.length !== 1 ? 's' : ''}
            {!skipJudge && ' + judge evaluation'}
          </p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Start Button */}
      <button
        onClick={handleStartEvaluation}
        disabled={loading || selectedModels.length === 0}
        className="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium text-lg"
      >
        {loading ? 'Starting Evaluation...' : `Start Evaluation (${selectedModels.length} models)`}
      </button>

      <p className="text-sm text-gray-500 text-center">
        Evaluation will run asynchronously. You can monitor progress in the Results tab.
      </p>
    </div>
  );
}
