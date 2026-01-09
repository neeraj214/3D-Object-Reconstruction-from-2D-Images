import React from 'react'
import Viewer3D from '../components/Viewer3D.jsx'

export default function PointCloudViewer({ plyUrl }) {
  return (
    <div className="card">
      <Viewer3D plyUrl={plyUrl} />
    </div>
  )
}