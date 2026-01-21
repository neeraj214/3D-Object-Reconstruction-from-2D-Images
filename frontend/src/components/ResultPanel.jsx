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
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
          <div className="text-sm text-gray-500">Points Generated</div>
          <div className="text-2xl font-bold text-gray-900">{numPoints.toLocaleString()}</div>
        </div>
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
          <div className="text-sm text-gray-500">Confidence</div>
          <div className="text-2xl font-bold text-blue-600">{(confidence * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
          <div className="text-sm text-gray-500">Processing Time</div>
          <div className="text-2xl font-bold text-gray-900">{processingTime.toFixed(2)}s</div>
        </div>
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
          <div className="text-sm text-gray-500">Device</div>
          <div className="text-xl font-bold text-gray-900 truncate" title={result.device || 'Unknown'}>{result.device || 'CUDA'}</div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Viewer - Takes 2/3 width on large screens */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div className="p-4 border-b border-gray-100 flex justify-between items-center">
              <h3 className="font-semibold text-gray-900">{result.mesh_url ? '3D Mesh' : '3D Point Cloud'}</h3>
              <span className="text-xs px-2 py-1 bg-blue-50 text-blue-600 rounded-full font-medium">Interactive</span>
            </div>
            <div className="p-1 bg-gray-900">
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
          <MetricCharts metrics={stats || {}} />
          
          <div className="bg-blue-50 rounded-xl p-5 border border-blue-100">
            <h4 className="font-semibold text-blue-900 mb-2">Analysis Summary</h4>
            <p className="text-sm text-blue-800 leading-relaxed">
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