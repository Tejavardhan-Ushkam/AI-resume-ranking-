import type { ScoreBreakdown } from '../services/api';

interface Props {
  breakdown: ScoreBreakdown;
}

const DIMENSIONS = [
  { key: 'semantic_match' as const, label: 'Semantic Fit', color: '#4f6ef7' },
  { key: 'skills_overlap' as const, label: 'Skills Match', color: '#10b981' },
  { key: 'career_dna_fit' as const, label: 'Career DNA', color: '#f59e0b' },
  { key: 'behavioral_score' as const, label: 'Signals', color: '#8b5cf6' },
];

export function ScoreBreakdownBar({ breakdown }: Props) {
  return (
    <div className="space-y-1.5">
      {DIMENSIONS.map(({ key, label, color }) => {
        const pct = Math.round(breakdown[key] * 100);
        return (
          <div key={key} className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-24 shrink-0">{label}</span>
            <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${pct}%`, backgroundColor: color }}
              />
            </div>
            <span className="text-xs font-medium text-gray-700 w-8 text-right">{pct}%</span>
          </div>
        );
      })}
    </div>
  );
}