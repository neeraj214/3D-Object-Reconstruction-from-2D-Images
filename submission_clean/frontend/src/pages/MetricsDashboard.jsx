import React, { useEffect, useState } from 'react'
import { getLatestMetrics } from '../services/reconstructionService'
import MetricCharts from '../components/MetricCharts.jsx'

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null)
  useEffect(()=>{ (async()=>{ try{ const m = await getLatestMetrics(); setMetrics(m) } catch{} })() },[])
  return (
    <div className="card">
      <h3 className="text-lg font-semibold mb-2">Metrics Dashboard</h3>
      {metrics ? (
        <div className="grid md:grid-cols-2 gap-4">
          <MetricCharts metrics={metrics} />
          <pre className="bg-gray-100 p-3 rounded overflow-auto">{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      ) : (
        <div>Loading metrics...</div>
      )}
    </div>
  )
}