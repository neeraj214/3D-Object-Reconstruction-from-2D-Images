import React, { useState } from 'react'
import { predict, health, autoDetectBase } from '../api/backend'
import { useToast, ToastContainer } from '../components/Toast.jsx'
import UploadBox from '../components/UploadBox.jsx'
import ResultPanel from '../components/ResultPanel.jsx'

export default function Upload() {
  const [image, setImage] = useState(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const { toasts, push } = useToast()
  const run = async () => {
    if (!image) return
    setBusy(true)
    try {
      await autoDetectBase()
      const h = await health();
      if (h.status !== 'ok') throw new Error('Backend offline. Set API Base to http://127.0.0.1:8000')
      const r = await predict(image, { mode: 'quality', nPoints: 40000, useSegmentation: true, fScale: 1.2 });
      setResult(r)
      push('Prediction completed','success')
    } catch (e) {
      push(e.message || 'Network Error','error')
    }
    setBusy(false)
  }
  const safeResult = (result && Array.isArray(result.point_cloud_coordinates) && result.point_cloud_coordinates.length>0) ? result : null
  return (
    <div className="space-y-4">
      <div className="card">
        <h3 className="text-lg font-semibold">Upload Image</h3>
        <UploadBox onResult={r=>setResult(r)} />
      </div>
      {safeResult && (<ResultPanel result={safeResult} />)}
      <ToastContainer toasts={toasts} />
    </div>
  )
}
