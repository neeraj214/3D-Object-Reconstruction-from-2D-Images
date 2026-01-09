import React from 'react'

function Bar({ label, value, max=1.0, color='#2563eb' }) {
  const pct = Math.min(100, Math.max(0, (value/max)*100))
  return (
    <div className="mb-2">
      <div className="flex justify-between text-sm"><span>{label}</span><span>{value?.toFixed ? value.toFixed(3) : value}</span></div>
      <div className="w-full h-2 bg-gray-200 rounded">
        <div style={{ width: pct+'%', background: color }} className="h-2 rounded" />
      </div>
    </div>
  )
}

export default function MetricCharts({ metrics }) {
  const chamfer = metrics?.chamfer_distance || 0.0
  const fscore = metrics?.fscore?.p2 || 0.0
  const iou = metrics?.iou || 0.0
  return (
    <div>
      <Bar label="Chamfer Distance" value={1.0 - chamfer} max={1.0} color="#059669" />
      <Bar label="F-Score p2" value={fscore} max={1.0} color="#2563eb" />
      <Bar label="IoU" value={iou} max={1.0} color="#7c3aed" />
    </div>
  )
}