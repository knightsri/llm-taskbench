import { useState } from 'react';
import UseCaseSelector from './components/UseCaseSelector/UseCaseSelector';
import UseCaseEvaluationRunner from './components/UseCaseEvaluationRunner';
import ResultsDashboard from './components/Results/ResultsDashboard';
import HistoryView from './components/History/HistoryView';
import type { UseCaseDetail, GeneratedPrompts, Evaluation } from './types';

type View = 'select' | 'evaluate' | 'results' | 'history';

function App() {
  const [currentView, setCurrentView] = useState<View>('select');
  const [selectedUseCase, setSelectedUseCase] = useState<UseCaseDetail | null>(null);
  const [generatedPrompts, setGeneratedPrompts] = useState<GeneratedPrompts | null>(null);
  const [selectedEvaluation, setSelectedEvaluation] = useState<Evaluation | null>(null);

  const handleUseCaseSelected = (usecase: UseCaseDetail, prompts: GeneratedPrompts | null) => {
    setSelectedUseCase(usecase);
    setGeneratedPrompts(prompts);
    setCurrentView('evaluate');
  };

  const handleEvaluationStarted = (evaluation: Evaluation) => {
    setSelectedEvaluation(evaluation);
    setCurrentView('results');
  };

  const handleBack = () => {
    setCurrentView('select');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">LLM TaskBench</h1>
              <p className="text-sm text-gray-600">Task-specific LLM evaluation framework</p>
            </div>
            <nav className="flex space-x-4">
              <button
                onClick={() => setCurrentView('select')}
                className={`px-4 py-2 rounded-md ${
                  currentView === 'select'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Use Cases
              </button>
              <button
                onClick={() => setCurrentView('evaluate')}
                className={`px-4 py-2 rounded-md ${
                  currentView === 'evaluate'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                disabled={!selectedUseCase}
              >
                Run Evaluation
              </button>
              <button
                onClick={() => setCurrentView('results')}
                className={`px-4 py-2 rounded-md ${
                  currentView === 'results'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                disabled={!selectedEvaluation}
              >
                View Results
              </button>
              <button
                onClick={() => setCurrentView('history')}
                className={`px-4 py-2 rounded-md ${
                  currentView === 'history'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                History
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentView === 'select' && (
          <UseCaseSelector onUseCaseSelected={handleUseCaseSelected} />
        )}
        {currentView === 'evaluate' && selectedUseCase && (
          <UseCaseEvaluationRunner
            usecase={selectedUseCase}
            prompts={generatedPrompts}
            onEvaluationStarted={handleEvaluationStarted}
            onBack={handleBack}
          />
        )}
        {currentView === 'results' && selectedEvaluation && (
          <ResultsDashboard evaluationId={selectedEvaluation.id} />
        )}
        {currentView === 'history' && (
          <HistoryView
            onSelectEvaluation={(evaluation) => {
              setSelectedEvaluation(evaluation);
              setCurrentView('results');
            }}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <p className="text-sm text-gray-500 text-center">
            LLM TaskBench - Evaluate models on your specific use cases
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
