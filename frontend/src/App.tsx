import { useState, useEffect } from 'react';
import { Zap, AlertCircle, Database } from 'lucide-react';
import { JobInputPanel } from './components/JobInputPanel';
import { CandidateCard } from './components/CandidateCard';
import { JDInsightsPanel } from './components/JDInsightsPanel';
import { searchCandidates, getCandidateCount } from './services/api';
import type { SearchResponse } from './services/api';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [candidateCount, setCandidateCount] = useState(0);

  useEffect(() => {
    getCandidateCount().then(setCandidateCount).catch(() => {});
  }, []);

  const handleSearch = async (jdText: string, title: string, topK: number) => {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await searchCandidates({
        job_description: jdText,
        job_title: title,
        top_k: topK,
      });
      setResponse(result);
      setCandidateCount(result.total_candidates_evaluated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-brand-500 rounded-lg flex items-center justify-center">
              <Zap size={16} className="text-white" />
            </div>
            <div>
              <h1 className="font-bold text-gray-900 text-base leading-tight">
                Intelligent Candidate Discovery
              </h1>
              <p className="text-xs text-gray-500">AI-powered multi-signal ranking</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <Database size={13} />
            <span>{candidateCount} candidates</span>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Input + Insights */}
          <div className="lg:col-span-1 space-y-4">
            <JobInputPanel
              onSearch={handleSearch}
              loading={loading}
              candidateCount={candidateCount}
            />

            {response && (
              <JDInsightsPanel
                jd={response.jd_decomposition}
                totalEvaluated={response.total_candidates_evaluated}
                processingMs={response.processing_time_ms}
              />
            )}

            {/* How it works */}
            {!response && !loading && (
              <div className="bg-gradient-to-br from-brand-50 to-white border border-brand-100 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-gray-800 mb-3">How it works</h3>
                <ol className="space-y-2">
                  {[
                    ['🧠', 'Deep JD Analysis', 'Extracts skills, seniority, culture signals'],
                    ['🔍', 'Semantic Search', 'Beyond keywords — meaning-aware matching'],
                    ['🧬', 'Career DNA', '6-axis candidate fingerprinting'],
                    ['⚡', 'Ensemble Ranking', 'Adaptive multi-signal scoring'],
                    ['💬', 'AI Explanations', 'Per-candidate reasoning'],
                  ].map(([icon, title, desc]) => (
                    <li key={title as string} className="flex items-start gap-2">
                      <span className="text-base leading-none mt-0.5">{icon}</span>
                      <div>
                        <p className="text-xs font-medium text-gray-800">{title as string}</p>
                        <p className="text-xs text-gray-500">{desc as string}</p>
                      </div>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-2">
            {/* Error */}
            {error && (
              <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-4">
                <AlertCircle size={16} className="text-red-500 shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-800">Search failed</p>
                  <p className="text-xs text-red-600 mt-0.5">{error}</p>
                </div>
              </div>
            )}

            {/* Loading skeleton */}
            {loading && (
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-4 bg-white rounded-xl border border-gray-200 mb-2">
                  <div className="w-5 h-5 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
                  <p className="text-sm text-gray-600">Analyzing JD and ranking candidates...</p>
                </div>
                {[1, 2, 3].map(i => (
                  <div key={i} className="bg-white rounded-xl border border-gray-200 p-4">
                    <div className="flex gap-3">
                      <div className="skeleton w-14 h-14 rounded-full" />
                      <div className="flex-1 space-y-2">
                        <div className="skeleton h-4 w-48 rounded" />
                        <div className="skeleton h-3 w-36 rounded" />
                        <div className="skeleton h-3 w-64 rounded" />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Results */}
            {response && !loading && (
              <>
                <div className="flex items-center justify-between mb-3">
                  <h2 className="font-semibold text-gray-900">
                    {response.results.length} Ranked Candidates
                  </h2>
                  <p className="text-xs text-gray-500">
                    from {response.total_candidates_evaluated} evaluated in {(response.processing_time_ms / 1000).toFixed(2)}s
                  </p>
                </div>

                <div className="space-y-3">
                  {response.results.map((result) => (
                    <CandidateCard
                      key={result.candidate.id}
                      result={result}
                      highlight={result.rank <= 3}
                    />
                  ))}
                </div>
              </>
            )}

            {/* Empty state */}
            {!response && !loading && !error && (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <div className="w-16 h-16 bg-brand-50 rounded-full flex items-center justify-center mb-4">
                  <Zap size={28} className="text-brand-500" />
                </div>
                <h3 className="font-medium text-gray-700 mb-1">Ready to discover talent</h3>
                <p className="text-sm text-gray-500 max-w-xs">
                  Paste a job description on the left and click <strong>Discover Candidates</strong> to see AI-ranked results.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}