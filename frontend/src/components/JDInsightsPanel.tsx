import type { JDDecomposition } from '../services/api';
import { Brain, Users, Layers, Clock } from 'lucide-react';

interface Props {
  jd: JDDecomposition;
  totalEvaluated: number;
  processingMs: number;
}

const SENIORITY_COLORS: Record<string, string> = {
  junior: 'bg-green-100 text-green-700',
  mid: 'bg-blue-100 text-blue-700',
  senior: 'bg-purple-100 text-purple-700',
  lead: 'bg-orange-100 text-orange-700',
  executive: 'bg-red-100 text-red-700',
};

export function JDInsightsPanel({ jd, totalEvaluated, processingMs }: Props) {
  const weightData = [
    { label: 'Skills', value: jd.skill_importance_weight, color: '#4f6ef7' },
    { label: 'Experience', value: jd.experience_importance_weight, color: '#10b981' },
    { label: 'Culture Fit', value: jd.culture_fit_weight, color: '#f59e0b' },
  ];

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 space-y-4">
      <h3 className="font-semibold text-gray-900 flex items-center gap-2 text-sm">
        <Brain size={15} className="text-brand-500" />
        JD Analysis
      </h3>

      {/* Meta stats */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-lg font-bold text-gray-900">{totalEvaluated}</p>
          <p className="text-xs text-gray-500">Evaluated</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-lg font-bold text-gray-900">{(processingMs / 1000).toFixed(1)}s</p>
          <p className="text-xs text-gray-500">Time</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-2">
          <span className={`text-xs font-semibold px-2 py-1 rounded-full ${SENIORITY_COLORS[jd.role_seniority] || 'bg-gray-100 text-gray-600'}`}>
            {jd.role_seniority}
          </span>
          <p className="text-xs text-gray-500 mt-0.5">{jd.role_type}</p>
        </div>
      </div>

      {/* Ranking Weights */}
      <div>
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Adaptive Weights</p>
        {weightData.map(({ label, value, color }) => (
          <div key={label} className="flex items-center gap-2 mb-1.5">
            <span className="text-xs text-gray-500 w-20 shrink-0">{label}</span>
            <div className="flex-1 bg-gray-100 rounded-full h-1.5">
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.round(value * 100)}%`, backgroundColor: color }}
              />
            </div>
            <span className="text-xs font-medium text-gray-700 w-8 text-right">
              {Math.round(value * 100)}%
            </span>
          </div>
        ))}
      </div>

      {/* Required Skills */}
      <div>
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">Required Skills</p>
        <div className="flex flex-wrap gap-1">
          {jd.required_skills.slice(0, 10).map(skill => (
            <span key={skill} className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full border border-blue-100">
              {skill}
            </span>
          ))}
        </div>
      </div>

      {/* Culture Signals */}
      {jd.culture_signals.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5 flex items-center gap-1">
            <Users size={11} /> Culture Signals
          </p>
          <div className="flex flex-wrap gap-1">
            {jd.culture_signals.map(signal => (
              <span key={signal} className="text-xs bg-purple-50 text-purple-700 px-2 py-0.5 rounded-full">
                {signal}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Experience bar */}
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <Clock size={11} />
        <span>Min. {jd.min_years_experience}y experience required</span>
      </div>
    </div>
  );
}