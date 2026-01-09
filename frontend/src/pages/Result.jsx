import React, { useState } from 'react'
import MeshViewer from '../components/MeshViewer.jsx'

export default function Result({ result }) {
  const [mode, setMode] = useState('mesh')
  if (!result) return <p>Loading...</p>
  const obj = result.mesh_url
  const mtl = result.texture_url ? result.mesh_url.replace('.obj','.mtl') : null
  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button className={`btn ${mode==='mesh'?'btn-primary':''}`} onClick={()=>setMode('mesh')}>Mesh</button>
      </div>
      {mode==='mesh' && <MeshViewer objUrl={obj} mtlUrl={mtl} />}
    </div>
  )
}