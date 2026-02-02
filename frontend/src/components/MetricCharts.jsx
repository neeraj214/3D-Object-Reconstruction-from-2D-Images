import React from 'react'
import { motion } from 'framer-motion'

function Bar({ label, value, max = 1.0, colorClass = 'bg-blue-600', delay = 0 }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div className="mb-4">
      <div className="flex justify-between text-sm font-medium text-gray-300 mb-1">
        <span>{label}</span>
        <span className="font-mono text-white">{value?.toFixed ? value.toFixed(3) : value}</span>
      </div>
      <div className="w-full h-2.5 bg-surface-soft rounded-full overflow-hidden border border-white/5">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, delay, ease: "easeOut" }}
          className={`h-full rounded-full ${colorClass}`}
        />
      </div>
    </div>
  )
}

export default function MetricCharts({ metrics }) {
  // Normalize/extract metrics with defaults
  const chamfer = metrics?.chamfer_distance ?? 0.0
  // Note: chamfer is "lower is better", so we might want to invert it for a "score" or just show it raw. 
  // The original code did 1.0 - chamfer, implying a "quality" score. I will keep that logic but label it clearly.
  const qualityScore = Math.max(0, 1.0 - chamfer)

  const fscore = metrics?.fscore?.p2 ?? 0.0
  const iou = metrics?.iou ?? 0.0

  return (
    <div className="bg-surface-glass backdrop-blur-md rounded-xl border border-white/10 p-5 shadow-lg">
      <h3 className="text-lg font-semibold text-white mb-4">Quality Metrics</h3>
      <Bar label="Shape Consistency (1 - Chamfer)" value={qualityScore} max={1.0} colorClass="bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]" delay={0.1} />
      <Bar label="F-Score (Precision)" value={fscore} max={1.0} colorClass="bg-brand-primary shadow-[0_0_10px_rgba(59,130,246,0.3)]" delay={0.2} />
      <Bar label="IoU (Overlap)" value={iou} max={1.0} colorClass="bg-violet-500 shadow-[0_0_10px_rgba(139,92,246,0.3)]" delay={0.3} />
    </div>
  )
}