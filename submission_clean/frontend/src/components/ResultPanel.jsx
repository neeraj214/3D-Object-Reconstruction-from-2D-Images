import React from 'react'
import PointCloudCanvas from './PointCloudCanvas.jsx'

export default function ResultPanel({ result }) {
  if (!result) {
    return <p>Loading...</p>
  }
  const stats = result.stats
  const confidence = result.confidence ?? result.confidence_score ?? 0
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <div className="card space-y-2">
        <div className="font-medium">Reconstruction Stats</div>
        {stats ? (
          <div className="text-sm">
            <div>Points: {stats.num_points}</div>
            <div>Quality: {Number(result.reconstruction_quality || stats.quality_score || 0).toFixed(3)}</div>
          </div>
        ) : <div>No stats</div>}
        <div className="mt-2">Confidence: {(confidence*100).toFixed(1)}%</div>
      </div>
      <div className="card">
        <PointCloudCanvas plyUrl={null} points={result.point_cloud_coordinates} autoRotate={false} />
      </div>
    </div>
  )
}