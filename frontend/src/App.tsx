import { useState } from 'react';
import TaskBuilder from './components/TaskBuilder/TaskBuilder';
import EvaluationRunner from './components/EvaluationRunner';
import ResultsDashboard from './components/Results/ResultsDashboard';
import HistoryView from './components/History/HistoryView';
import type { Task, Evaluation } from './types';

type View = 'tasks' | 'evaluate' | 'results' | 'history';

function App() {
  const [currentView, setCurrentView] = useState<View>('tasks');
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [selectedEvaluation, setSelectedEvaluation] = useState<Evaluation | null>(null);

  const handleTaskCreated = (task: Task) => {
    setSelectedTask(task);
    setCurrentView('evaluate');
  };

  const handleEvaluationStarted = (evaluation: Evaluation) => {
    setSelectedEvaluation(evaluation);
    setCurrentView('results');
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
                onClick={() => setCurrentView('tasks')}
                className={`px-4 py-2 rounded-md ${
                  currentView === 'tasks'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Create Task
              </button>
              <button
                onClick={() => setCurrentView('evaluate')}
                className={`px-4 py-2 rounded-md ${
                  currentView === 'evaluate'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                disabled={!selectedTask}
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
        {currentView === 'tasks' && (
          <TaskBuilder onTaskCreated={handleTaskCreated} />
        )}
        {currentView === 'evaluate' && selectedTask && (
          <EvaluationRunner
            task={selectedTask}
            onEvaluationStarted={handleEvaluationStarted}
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
    </div>
  );
}

export default App;
