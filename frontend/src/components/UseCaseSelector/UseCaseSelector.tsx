import { useState, useEffect } from 'react';
import { usecasesApi } from '../../api/client';
import type { UseCase, UseCaseDetail, GeneratedPrompts } from '../../types';

interface UseCaseSelectorProps {
  onUseCaseSelected: (usecase: UseCaseDetail, prompts: GeneratedPrompts | null) => void;
}

export default function UseCaseSelector({ onUseCaseSelected }: UseCaseSelectorProps) {
  const [usecases, setUsecases] = useState<UseCase[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<UseCaseDetail | null>(null);
  const [prompts, setPrompts] = useState<GeneratedPrompts | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [generatingPrompts, setGeneratingPrompts] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    loadUsecases();
  }, []);

  const loadUsecases = async () => {
    try {
      const data = await usecasesApi.list();
      setUsecases(data);
      setLoading(false);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load use cases');
      setLoading(false);
    }
  };

  const handleSelectUsecase = async (path: string) => {
    setSelectedPath(path);
    setLoadingDetail(true);
    setError('');

    try {
      const detail = await usecasesApi.get(path);
      setSelectedDetail(detail);

      // Try to load existing prompts
      const existingPrompts = await usecasesApi.getPrompts(path);
      setPrompts(existingPrompts);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load use case details');
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleGeneratePrompts = async () => {
    if (!selectedPath) return;

    setGeneratingPrompts(true);
    setError('');

    try {
      const newPrompts = await usecasesApi.generatePrompts(selectedPath, true);
      setPrompts(newPrompts);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate prompts');
    } finally {
      setGeneratingPrompts(false);
    }
  };

  const handleContinue = () => {
    if (selectedDetail) {
      onUseCaseSelected(selectedDetail, prompts);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading use cases...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Use Case List */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Select Use Case</h2>
        <p className="text-gray-600 mb-6">
          Choose a use case from the sample-usecases folder to evaluate models against.
        </p>

        <div className="grid grid-cols-1 gap-4">
          {usecases.map((usecase) => (
            <div
              key={usecase.path}
              onClick={() => handleSelectUsecase(usecase.path)}
              className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                selectedPath === usecase.path
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-semibold text-lg">{usecase.name}</h3>
                  <p className="text-sm text-gray-500 mt-1">{usecase.folder_name}</p>
                  {usecase.error ? (
                    <p className="text-sm text-red-600 mt-2">{usecase.error}</p>
                  ) : (
                    <div className="flex gap-4 mt-2 text-sm text-gray-600">
                      <span className="flex items-center gap-1">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        {usecase.data_files} data files
                      </span>
                      <span className="flex items-center gap-1">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {usecase.ground_truth_files} ground truth
                      </span>
                      <span className="flex items-center gap-1">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                        </svg>
                        {usecase.matched_pairs} pairs
                      </span>
                    </div>
                  )}
                </div>
                <div className="flex flex-col items-end gap-2">
                  {usecase.difficulty && (
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      usecase.difficulty.toLowerCase().includes('hard')
                        ? 'bg-red-100 text-red-800'
                        : usecase.difficulty.toLowerCase().includes('moderate')
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {usecase.difficulty}
                    </span>
                  )}
                  {usecase.output_format && (
                    <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                      {usecase.output_format.toUpperCase()}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Selected Use Case Details */}
      {loadingDetail && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-center h-32">
            <div className="text-gray-500">Loading use case details...</div>
          </div>
        </div>
      )}

      {selectedDetail && !loadingDetail && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-bold mb-4">{selectedDetail.name}</h3>

          {/* Goal */}
          <div className="mb-4">
            <h4 className="font-semibold text-gray-700 mb-2">Goal</h4>
            <p className="text-gray-600 whitespace-pre-line">{selectedDetail.goal}</p>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            {selectedDetail.difficulty && (
              <div>
                <span className="text-sm text-gray-500">Difficulty</span>
                <p className="font-medium">{selectedDetail.difficulty}</p>
              </div>
            )}
            {selectedDetail.primary_capability && (
              <div>
                <span className="text-sm text-gray-500">Capability</span>
                <p className="font-medium">{selectedDetail.primary_capability}</p>
              </div>
            )}
            {selectedDetail.output_format && (
              <div>
                <span className="text-sm text-gray-500">Output Format</span>
                <p className="font-medium">{selectedDetail.output_format.toUpperCase()}</p>
              </div>
            )}
            {selectedDetail.token_range && (
              <div>
                <span className="text-sm text-gray-500">Token Range</span>
                <p className="font-medium">{selectedDetail.token_range}</p>
              </div>
            )}
          </div>

          {/* Data Files */}
          <div className="mb-4">
            <h4 className="font-semibold text-gray-700 mb-2">
              Data Files ({selectedDetail.data_files.length})
            </h4>
            <div className="flex flex-wrap gap-2">
              {selectedDetail.data_files.map((file, idx) => (
                <span key={idx} className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-sm">
                  {file.name}{file.extension}
                </span>
              ))}
            </div>
          </div>

          {/* Ground Truth */}
          <div className="mb-4">
            <h4 className="font-semibold text-gray-700 mb-2">
              Ground Truth Files ({selectedDetail.ground_truth_files.length})
            </h4>
            <div className="flex flex-wrap gap-2">
              {selectedDetail.ground_truth_files.map((file, idx) => (
                <span key={idx} className="px-2 py-1 bg-green-50 text-green-700 rounded text-sm">
                  {file.name}{file.extension}
                </span>
              ))}
            </div>
          </div>

          {/* Edge Cases */}
          {selectedDetail.edge_cases.length > 0 && (
            <div className="mb-4">
              <h4 className="font-semibold text-gray-700 mb-2">Edge Cases</h4>
              <ul className="list-disc list-inside text-gray-600 text-sm">
                {selectedDetail.edge_cases.map((ec, idx) => (
                  <li key={idx}>{ec}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Prompts Status */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold">Generated Prompts</h4>
                {prompts ? (
                  <p className="text-sm text-green-600">Prompts available</p>
                ) : (
                  <p className="text-sm text-yellow-600">Prompts not yet generated</p>
                )}
              </div>
              <button
                onClick={handleGeneratePrompts}
                disabled={generatingPrompts}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50"
              >
                {generatingPrompts ? 'Generating...' : prompts ? 'Regenerate' : 'Generate'}
              </button>
            </div>

            {prompts && (
              <div className="mt-4 space-y-3">
                <div>
                  <span className="text-sm font-medium text-gray-700">Transformation Type:</span>
                  <span className="ml-2 text-sm text-gray-600">{prompts.analysis.transformation_type}</span>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-700">Key Fields:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {prompts.analysis.key_fields.map((field, idx) => (
                      <span key={idx} className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                        {field}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-700">Compliance Checks:</span>
                  <span className="ml-2 text-sm text-gray-600">
                    {prompts.rubric.compliance_checks.length} checks configured
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Continue Button */}
      {selectedDetail && (
        <button
          onClick={handleContinue}
          className="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 font-medium"
        >
          Continue to Model Selection
        </button>
      )}
    </div>
  );
}
