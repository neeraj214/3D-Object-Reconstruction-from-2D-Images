import React from 'react'
import { motion } from 'framer-motion'
import PointCloudCanvas from './PointCloudCanvas.jsx'
import MeshViewer from './MeshViewer.jsx'
import MetricCharts from './MetricCharts.jsx'

export default function ResultPanel({ result }) {
  if (!result) {
    return (
      <div className="animate-pulse flex space-x-4 p-4 border border-gray-200 rounded-xl bg-white">
        <div className="flex-1 space-y-4 py-1">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          </div>
        </div>
      </div>
    )
  }

  const stats = result.stats
  const confidence = result.confidence ?? result.confidence_score ?? 0
  const numPoints = stats ? stats.num_points : result.num_points || 0
  const processingTime = result.processing_time || 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Header Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-surface-soft/50 p-4 rounded-xl backdrop-blur-md border border-white/5">
          <div className="text-sm text-gray-400">Points Generated</div>
          <div className="text-2xl font-bold text-white font-display">{numPoints.toLocaleString()}</div>
        </div>
        <div className="bg-surface-soft/50 p-4 rounded-xl backdrop-blur-md border border-white/5">
          <div className="text-sm text-gray-400">Confidence</div>
          <div className="text-2xl font-bold text-brand-accent">{(confidence * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-surface-soft/50 p-4 rounded-xl backdrop-blur-md border border-white/5">
          <div className="text-sm text-gray-400">Processing Time</div>
          <div className="text-2xl font-bold text-white font-display">{processingTime.toFixed(2)}s</div>
        </div>
        <div className="bg-surface-soft/50 p-4 rounded-xl backdrop-blur-md border border-white/5">
          <div className="text-sm text-gray-400">Device</div>
          <div className="text-xl font-bold text-white truncate font-display" title={result.device || 'Unknown'}>{result.device || 'CUDA'}</div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Viewer - Takes 2/3 width on large screens */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-surface-glass rounded-2xl shadow-2xl border border-white/10 overflow-hidden">
            <div className="p-4 border-b border-white/10 flex justify-between items-center bg-brand-darker/50">
              <h3 className="font-semibold text-white">{result.mesh_url ? '3D Mesh' : '3D Point Cloud'}</h3>
              <span className="text-xs px-2 py-1 bg-brand-primary/10 text-brand-primary border border-brand-primary/20 rounded-full font-medium">Interactive</span>
            </div>
            <div className="p-1 bg-black/80 aspect-video relative">
              {/* Viewer Container */}
              {result.mesh_url ? (
                <MeshViewer
                  objUrl={result.mesh_url}
                  mtlUrl={result.mesh_url.replace(/\.obj$/i, '.mtl')}
                  autoRotate={true}
                />
              ) : (
                <PointCloudCanvas
                  plyUrl={null}
                  points={result.point_cloud_coordinates}
                  autoRotate={true}
                />
              )}
            </div>
          </div>
        </div>

        {/* Metrics Sidebar - Takes 1/3 width */}
        <div className="space-y-6">
          {/* <MetricCharts metrics={stats || {}} />  -- Might need update if it has internal styles */}

          <div className="bg-brand-primary/10 rounded-2xl p-6 border border-brand-primary/20">
            <h4 className="font-semibold text-brand-primary mb-3 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
              Analysis Summary
            </h4>
            <p className="text-sm text-blue-100 leading-relaxed">
              The model reconstructed the object with <strong>{(confidence * 100).toFixed(0)}% confidence</strong>.
              {stats && stats.chamfer_distance < 0.05
                ? " Shape consistency is high, indicating accurate geometry recovery."
                : " Some geometric details might be smoothed out."}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}