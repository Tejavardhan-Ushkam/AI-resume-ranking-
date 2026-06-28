import { useState } from 'react';
import {
  ChevronDown, ChevronUp, MapPin, Briefcase, Star, AlertCircle,
  CheckCircle, XCircle, Activity
} from 'lucide-react';
import type { CandidateRankResult } from '../services/api';
import { CareerDNAChart } from './CareerDNAChart';
import { ScoreBreakdownBar } from './ScoreBreakdownBar';

interface Props {
  result: CandidateRankResult;
  highlight?: boolean;
}

function MatchBadge({ pct }: { pct: number }) {
  const color =
    pct >= 80 ? 'bg-emerald-100 text-emerald-700 border-emerald-200' :
    pct >= 60 ? 'bg-blue-100 text-blue-700 border-blue-200' :
    pct >= 40 ? 'bg-amber-100 text-amber-700 border-amber-200' :
                'bg-red-100 text-red-700 border-red-200';

  return (
    <div className={`flex items-center justify-center w-14 h-14 rounded-full border-2 font-bold text-lg ${color}`}>
      {pct}%
    </div>
  );
}

function RankBadge({ rank }: { rank: number }) {
  if (rank === 1) return <span className="text-lg">🥇</span>;
  if (rank === 2) return <span className="text-lg">🥈</span>;
  if (rank === 3) return <span className="text-lg">🥉</span>;
  return (
    <span className="text-xs font-semibold text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
      #{rank}
    </span>
  );
}

export function CandidateCard({ result, highlight }: Props) {
  const [expanded, setExpanded] = useState(result.rank <= 3);
  const { candidate, rank, score_breakdown, match_percentage, matched_skills, missing_skills, strengths, risks, ai_summary } = result;
  const dna = candidate.career_dna;

  return (
    <div className={`bg-white rounded-xl border transition-all duration-200 ${
      highlight ? 'border-brand-500 shadow-md shadow-brand-500/10' : 'border-gray-200 shadow-sm hover:shadow-md'
    }`}>
      {/* Card Header */}
      <div className="p-4">
        <div className="flex items-start gap-3">
          {/* Rank + Match */}
          <div className="flex flex-col items-center gap-1 shrink-0">
            <RankBadge rank={rank} />
            <MatchBadge pct={Math.round(match_percentage)} />
          </div>

          {/* Candidate Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-semibold text-gray-900 text-base">{candidate.name}</h3>
              {candidate.behavioral_signals.open_to_work && (
                <span className="text-xs bg-green-50 text-green-700 px-2 py-0.5 rounded-full border border-green-200">
                  Open to Work
                </span>
              )}
            </div>
            <p className="text-sm text-gray-600">{candidate.current_title}</p>
            {candidate.current_company && (
              <p className="text-xs text-gray-500">{candidate.current_company}</p>
            )}
            <div className="flex items-center gap-3 mt-1 flex-wrap">
              <span className="flex items-center gap-1 text-xs text-gray-500">
                <MapPin size={11} /> {candidate.location}
              </span>
              <span className="flex items-center gap-1 text-xs text-gray-500">
                <Briefcase size={11} /> {candidate.years_of_experience}y exp
              </span>
              {candidate.behavioral_signals.last_active_days_ago > 0 && (
                <span className="flex items-center gap-1 text-xs text-gray-500">
                  <Activity size={11} /> Active {candidate.behavioral_signals.last_active_days_ago}d ago
                </span>
              )}
            </div>

            {/* Matched skills chips */}
            <div className="flex flex-wrap gap-1 mt-2">
              {matched_skills.slice(0, 6).map(skill => (
                <span key={skill} className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">
                  ✓ {skill}
                </span>
              ))}
              {missing_skills.slice(0, 2).map(skill => (
                <span key={skill} className="text-xs bg-red-50 text-red-400 px-2 py-0.5 rounded-full">
                  ✗ {skill}
                </span>
              ))}
            </div>
          </div>

          {/* DNA Preview (always visible) */}
          {dna && (
            <div className="shrink-0 hidden sm:block">
              <CareerDNAChart dna={dna} size={130} />
            </div>
          )}

          {/* Expand toggle */}
          <button
            onClick={() => setExpanded(e => !e)}
            className="shrink-0 text-gray-400 hover:text-gray-600 transition-colors p-1"
          >
            {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
        </div>

        {/* AI Summary — always visible */}
        {ai_summary && (
          <div className="mt-3 pl-0 border-l-2 border-brand-500/30 pl-3">
            <p className="text-xs text-gray-600 italic leading-relaxed">{ai_summary}</p>
          </div>
        )}
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-gray-100 pt-3 space-y-4">
          {/* Score Breakdown */}
          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Score Breakdown</p>
            <ScoreBreakdownBar breakdown={score_breakdown} />
          </div>

          {/* Strengths & Risks */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {strengths.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                  <CheckCircle size={11} /> Strengths
                </p>
                <ul className="space-y-1">
                  {strengths.map((s, i) => (
                    <li key={i} className="text-xs text-gray-700 flex items-start gap-1.5">
                      <Star size={10} className="text-emerald-500 mt-0.5 shrink-0" />
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {risks.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-amber-600 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                  <AlertCircle size={11} /> Watch-outs
                </p>
                <ul className="space-y-1">
                  {risks.map((r, i) => (
                    <li key={i} className="text-xs text-gray-700 flex items-start gap-1.5">
                      <XCircle size={10} className="text-amber-500 mt-0.5 shrink-0" />
                      {r}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Summary */}
          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Candidate Summary</p>
            <p className="text-xs text-gray-600 leading-relaxed">{candidate.summary}</p>
          </div>

          {/* Skills full list */}
          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">All Skills</p>
            <div className="flex flex-wrap gap-1">
              {candidate.skills.map(skill => (
                <span
                  key={skill}
                  className={`text-xs px-2 py-0.5 rounded-full ${
                    matched_skills.includes(skill)
                      ? 'bg-blue-50 text-blue-700'
                      : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {skill}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}