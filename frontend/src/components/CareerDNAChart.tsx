import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';
import type { CareerDNA } from '../services/api';

interface Props {
  dna: CareerDNA;
  size?: number;
}

const DNA_LABELS: Record<keyof CareerDNA, string> = {
  skill_depth_score: 'Skill Depth',
  growth_velocity: 'Growth',
  domain_breadth: 'Breadth',
  tenure_stability: 'Stability',
  recency_score: 'Recency',
  collaboration_signal: 'Leadership',
};

export function CareerDNAChart({ dna, size = 180 }: Props) {
  const data = (Object.keys(DNA_LABELS) as Array<keyof CareerDNA>).map((key) => ({
    axis: DNA_LABELS[key],
    value: Math.round(dna[key] * 100),
    fullMark: 100,
  }));

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Career DNA</p>
      <ResponsiveContainer width={size} height={size}>
        <RadarChart data={data} margin={{ top: 5, right: 15, bottom: 5, left: 15 }}>
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis
            dataKey="axis"
            tick={{ fontSize: 9, fill: '#64748b' }}
          />
          <Radar
            dataKey="value"
            stroke="#4f6ef7"
            fill="#4f6ef7"
            fillOpacity={0.2}
            strokeWidth={1.5}
          />
          <Tooltip
            formatter={(val: number) => [`${val}%`, '']}
            contentStyle={{ fontSize: 11, borderRadius: 6 }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}