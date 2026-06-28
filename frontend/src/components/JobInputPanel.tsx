import { useState } from 'react';
import { Search, Sparkles, ChevronDown } from 'lucide-react';

interface Props {
  onSearch: (jdText: string, title: string, topK: number) => void;
  loading: boolean;
  candidateCount: number;
}

const EXAMPLE_JDS = [
  {
    title: 'Senior ML Engineer',
    text: `We're looking for a Senior ML Engineer to join our AI team. 
    
You will design and deploy large-scale machine learning models in production, working on recommendation systems, NLP, and personalization. You'll collaborate with data scientists and product managers to ship ML features that impact millions of users.

Requirements:
- 5+ years of ML/AI engineering experience
- Deep expertise in Python, PyTorch or TensorFlow
- Experience with NLP, transformers, and LLMs
- Production ML system design (feature stores, model serving, monitoring)
- Proficiency with cloud platforms (AWS/GCP) and containerization (Docker, Kubernetes)

Preferred:
- Publications at top ML venues (NeurIPS, ICML, ICLR)
- Experience with LLM fine-tuning and RAG architectures
- MLOps tooling (MLflow, Kubeflow)
- Leadership or mentoring experience`
  },
  {
    title: 'Full Stack Engineer',
    text: `We are hiring a Full Stack Engineer for our fast-growing fintech startup.

You'll build and own end-to-end features from database to UI for our payments platform serving 2M+ merchants. We move fast, ship often, and care deeply about code quality.

Requirements:
- 3+ years of full-stack development experience
- Proficiency in React, TypeScript, and Node.js
- Strong backend skills with PostgreSQL and Redis
- Experience with REST APIs and GraphQL
- Comfort with Docker and cloud deployments (AWS)

Nice to have:
- Experience in fintech or high-transaction systems
- Knowledge of message queues (Kafka, SQS)
- Previous startup experience`
  },
  {
    title: 'Data Science Manager',
    text: `We are seeking an experienced Data Science Manager to lead our analytics team.

You will manage a team of 8-12 data scientists, set technical direction, and partner with product and engineering to drive data-informed decisions. You'll own the roadmap for our ML initiatives in pricing, demand forecasting, and personalization.

Requirements:
- 7+ years in data science / ML, with 2+ years in management
- Strong Python and SQL skills
- Track record of shipping ML models to production
- Experience with stakeholder management and cross-functional collaboration
- MBA or advanced degree in quantitative field preferred`
  }
];

export function JobInputPanel({ onSearch, loading, candidateCount }: Props) {
  const [jdText, setJdText] = useState('');
  const [title, setTitle] = useState('');
  const [topK, setTopK] = useState(10);
  const [showExamples, setShowExamples] = useState(false);

  const handleSubmit = () => {
    if (!jdText.trim()) return;
    onSearch(jdText.trim(), title.trim() || 'Role', topK);
  };

  const loadExample = (ex: typeof EXAMPLE_JDS[0]) => {
    setJdText(ex.text.trim());
    setTitle(ex.title);
    setShowExamples(false);
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-semibold text-gray-900 flex items-center gap-2">
            <Sparkles size={16} className="text-brand-500" />
            Job Description
          </h2>
          <p className="text-xs text-gray-500 mt-0.5">
            {candidateCount} candidates indexed
          </p>
        </div>
        <button
          onClick={() => setShowExamples(s => !s)}
          className="flex items-center gap-1 text-xs text-brand-600 hover:text-brand-700 font-medium"
        >
          Examples <ChevronDown size={13} className={`transition-transform ${showExamples ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Example JDs */}
      {showExamples && (
        <div className="mb-3 grid grid-cols-1 gap-1.5">
          {EXAMPLE_JDS.map((ex) => (
            <button
              key={ex.title}
              onClick={() => loadExample(ex)}
              className="text-left px-3 py-2 rounded-lg bg-gray-50 hover:bg-brand-50 text-sm text-gray-700 hover:text-brand-700 transition-colors border border-gray-100 hover:border-brand-200"
            >
              📋 {ex.title}
            </button>
          ))}
        </div>
      )}

      {/* Job Title */}
      <input
        type="text"
        placeholder="Job title (e.g., Senior ML Engineer)"
        value={title}
        onChange={e => setTitle(e.target.value)}
        className="w-full mb-3 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500"
      />

      {/* JD Text */}
      <textarea
        className="w-full h-52 px-3 py-2 border border-gray-200 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500 font-mono text-gray-700 leading-relaxed"
        placeholder="Paste the full job description here...

Include: required skills, experience level, responsibilities, and any culture signals. The more detail, the better the ranking."
        value={jdText}
        onChange={e => setJdText(e.target.value)}
      />

      {/* Controls */}
      <div className="flex items-center justify-between mt-3 gap-3">
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500">Top</label>
          <select
            value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-brand-500"
          >
            {[5, 10, 15, 20, 30].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
          <label className="text-xs text-gray-500">results</label>
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading || !jdText.trim()}
          className="flex items-center gap-2 px-4 py-2 bg-brand-500 hover:bg-brand-600 disabled:bg-gray-300 text-white text-sm font-medium rounded-lg transition-colors"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Ranking...
            </>
          ) : (
            <>
              <Search size={15} />
              Discover Candidates
            </>
          )}
        </button>
      </div>
    </div>
  );
}