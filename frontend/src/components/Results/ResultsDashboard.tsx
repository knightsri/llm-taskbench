import { useState, useEffect } from 'react';
import { evaluationsApi } from '../../api/client';
import type { Evaluation, EvaluationProgress } from '../../types';

interface ResultsDashboardProps {
  evaluationId: string;
}

export default function ResultsDashboard({ evaluationId }: ResultsDashboardProps) {
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [progress, setProgress] = useState<EvaluationProgress | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadEvaluation();
    const interval = setInterval(loadProgress, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [evaluationId]);

  const loadEvaluation = async () => {
    try {
      const data = await evaluationsApi.get(evaluationId);
      setEvaluation(data);
      setLoading(false);
    } catch (err) {
      console.error('Error loading evaluation:', err);
      setLoading(false);
    }
  };

  const loadProgress = async () => {
    try {
      const data = await evaluationsApi.getProgress(evaluationId);
      setProgress(data);

      // Reload full evaluation if completed
      if (data.status === 'completed' || data.status === 'failed') {
        loadEvaluation();
      }
    } catch (err) {
      console.error('Error loading progress:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading evaluation...</div>
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <p className="text-red-600">Evaluation not found</p>
      </div>
    );
  }

  const sortedResults = [...evaluation.results].sort((a, b) => {
    const scoreA = a.accuracy || 0;
    const scoreB = b.accuracy || 0;
    return scoreB - scoreA;
  });

  return (
    <div className="space-y-6">
      {/* Progress Bar */}
      {evaluation.status === 'running' && progress && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="font-semibold mb-3">Evaluation Progress</h3>
          <div className="mb-2">
            <div className="flex justify-between text-sm mb-1">
              <span>{progress.message}</span>
              <span>{Math.round(progress.progress_percent)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{ width: `${progress.progress_percent}%` }}
              ></div>
            </div>
          </div>
          <p className="text-sm text-gray-600">
            Completed: {progress.completed_models} / {progress.total_models} models
          </p>
        </div>
      )}

      {/* Results Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="p-6">
          <h2 className="text-2xl font-bold mb-4">Evaluation Results</h2>
          <div className="mb-4 flex items-center justify-between">
            <div>
              <span className="inline-block px-3 py-1 rounded-full text-sm font-medium">
                {evaluation.status === 'completed' && (
                  <span className="bg-green-100 text-green-800">Completed</span>
                )}
                {evaluation.status === 'running' && (
                  <span className="bg-blue-100 text-blue-800">Running</span>
                )}
                {evaluation.status === 'pending' && (
                  <span className="bg-yellow-100 text-yellow-800">Pending</span>
                )}
                {evaluation.status === 'failed' && (
                  <span className="bg-red-100 text-red-800">Failed</span>
                )}
              </span>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-600">Total Cost</p>
              <p className="text-2xl font-bold text-blue-600">
                ${evaluation.actual_cost.toFixed(4)}
              </p>
            </div>
          </div>
        </div>

        {/* Results Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rank</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Accuracy</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Completeness</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Hallucination</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Cost</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Violations</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {sortedResults.map((result, index) => (
                <tr key={result.id} className={index === 0 ? 'bg-green-50' : ''}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {index === 0 && <span className="text-2xl">🏆</span>}
                    {index === 1 && <span className="text-xl">🥈</span>}
                    {index === 2 && <span className="text-xl">🥉</span>}
                    {index > 2 && <span className="text-gray-500">#{index + 1}</span>}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">{result.model_id}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {result.accuracy !== null && result.accuracy !== undefined ? (
                      <span className="text-green-600 font-semibold">
                        {(result.accuracy * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {result.completeness !== null && result.completeness !== undefined ? (
                      <span>{(result.completeness * 100).toFixed(1)}%</span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {result.hallucination_rate !== null && result.hallucination_rate !== undefined ? (
                      <span className="text-red-600">
                        {(result.hallucination_rate * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-blue-600 font-medium">
                      ${result.cost.toFixed(4)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-block px-2 py-1 rounded text-xs ${
                      result.quality_violations.length === 0
                        ? 'bg-green-100 text-green-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {result.quality_violations.length}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-block px-2 py-1 rounded text-xs ${
                      result.status === 'completed'
                        ? 'bg-green-100 text-green-800'
                        : result.status === 'failed'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-blue-100 text-blue-800'
                    }`}>
                      {result.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
