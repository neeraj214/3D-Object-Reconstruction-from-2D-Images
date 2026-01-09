import React, { useState } from 'react'
import ImageUpload from './ImageUpload.jsx'
import { predict, uploadImage } from '../api/backend'

export default function UploadBox({ onResult }) {
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [useSegmentation, setUseSegmentation] = useState(true)
  const [fScale, setFScale] = useState(1.1)
  const [mode, setMode] = useState('fast')
  const submit = async () => {
    if (!file) return
    setBusy(true); setErr(null)
    try {
      await uploadImage(file)
      const r = await predict(file, { fScale, useSegmentation, mode })
      onResult(r)
    } catch (e) {
      setErr(e.message || 'Upload failed')
    }
    setBusy(false)
  }
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="font-medium">Upload</div>
      </div>
      <ImageUpload onSubmit={f=>setFile(f)} />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={useSegmentation} onChange={e=>setUseSegmentation(e.target.checked)} /> Foreground segmentation
        </label>
        <div className="flex items-center gap-2 text-sm">
          <span>Focal scale</span>
          <input type="range" min="0.8" max="1.4" step="0.01" value={fScale} onChange={e=>setFScale(parseFloat(e.target.value))} />
          <span>{fScale.toFixed(2)}</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span>Mode</span>
          <select value={mode} onChange={e=>setMode(e.target.value)}>
            <option value="fast">fast</option>
            <option value="balanced">balanced</option>
            <option value="quality">quality</option>
          </select>
        </div>
      </div>
      <button className="btn" onClick={submit} disabled={busy}>Reconstruct</button>
      {busy && <div>Reconstructing 3D model…</div>}
      {err && <div className="text-red-600">{err}</div>}
    </div>
  )
}
