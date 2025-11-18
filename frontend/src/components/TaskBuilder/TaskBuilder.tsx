import { useState } from 'react';
import { tasksApi } from '../../api/client';
import type { Task, QualityCheck } from '../../types';

interface TaskBuilderProps {
  onTaskCreated: (task: Task) => void;
}

export default function TaskBuilder({ onTaskCreated }: TaskBuilderProps) {
  const [taskName, setTaskName] = useState('');
  const [description, setDescription] = useState('');
  const [domain, setDomain] = useState('');
  const [outputFormat, setOutputFormat] = useState('json');
  const [loading, setLoading] = useState(false);
  const [qualityChecks, setQualityChecks] = useState<QualityCheck[]>([]);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const task = await tasksApi.create({
        name: taskName,
        description,
        domain: domain || undefined,
        output_format: outputFormat,
        input_format: 'text',
        constraints: {},
        evaluation_criteria: [],
      });

      setQualityChecks(task.quality_checks);
      onTaskCreated(task);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create task');
      console.error('Error creating task:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-6">Create Evaluation Task</h2>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Task Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Task Name
          </label>
          <input
            type="text"
            value={taskName}
            onChange={(e) => setTaskName(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="e.g., lecture_concept_extraction"
            required
          />
        </div>

        {/* Description */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Task Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="Describe what you want the LLM to accomplish..."
            required
          />
          <p className="mt-1 text-sm text-gray-500">
            The framework will analyze this and auto-generate quality checks
          </p>
        </div>

        {/* Domain */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Domain (Optional)
          </label>
          <select
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select domain...</option>
            <option value="education">Education</option>
            <option value="healthcare">Healthcare</option>
            <option value="legal">Legal</option>
            <option value="finance">Finance</option>
            <option value="customer-support">Customer Support</option>
            <option value="content-creation">Content Creation</option>
          </select>
        </div>

        {/* Output Format */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Expected Output Format
          </label>
          <select
            value={outputFormat}
            onChange={(e) => setOutputFormat(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
            <option value="markdown">Markdown</option>
          </select>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !taskName || !description}
          className="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
        >
          {loading ? 'Creating Task & Generating Quality Checks...' : 'Create Task'}
        </button>
      </form>

      {/* Quality Checks Preview */}
      {qualityChecks.length > 0 && (
        <div className="mt-8 p-4 bg-blue-50 rounded-md">
          <h3 className="font-semibold text-lg mb-3">
            Generated Quality Checks ({qualityChecks.length})
          </h3>
          <ul className="space-y-2">
            {qualityChecks.map((check, index) => (
              <li key={index} className="flex items-start">
                <span className="inline-block w-2 h-2 bg-blue-600 rounded-full mt-2 mr-3"></span>
                <div>
                  <strong className="text-gray-900">{check.name}</strong>
                  <p className="text-sm text-gray-600">{check.description}</p>
                  <span className="text-xs text-gray-500">[{check.severity}]</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
