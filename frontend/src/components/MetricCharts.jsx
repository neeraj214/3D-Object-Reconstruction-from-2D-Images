import React from 'react'
import { motion } from 'framer-motion'

function Bar({ label, value, max=1.0, colorClass='bg-blue-600', delay=0 }) {
  const pct = Math.min(100, Math.max(0, (value/max)*100))
  
  return (
    <div className="mb-4">
      <div className="flex justify-between text-sm font-medium text-gray-700 mb-1">
        <span>{label}</span>
        <span className="font-mono text-gray-900">{value?.toFixed ? value.toFixed(3) : value}</span>
      </div>
      <div className="w-full h-2.5 bg-gray-100 rounded-full overflow-hidden">
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
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Quality Metrics</h3>
      <Bar label="Shape Consistency (1 - Chamfer)" value={qualityScore} max={1.0} colorClass="bg-emerald-500" delay={0.1} />
      <Bar label="F-Score (Precision)" value={fscore} max={1.0} colorClass="bg-blue-600" delay={0.2} />
      <Bar label="IoU (Overlap)" value={iou} max={1.0} colorClass="bg-violet-600" delay={0.3} />
    </div>
  )
}