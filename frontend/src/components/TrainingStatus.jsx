import React, { useEffect, useState } from 'react'
import { api } from '../services/apiClient'

export default function TrainingStatus() {
  const [status, setStatus] = useState(null)
  useEffect(() => {
    let t = setInterval(async () => {
      try {
        const base = api.defaults.baseURL || 'http://127.0.0.1:8000'
        const r = await fetch(base + '/results/checkpoints_v3/status.json', { cache: 'no-store' })
        if (r.ok) setStatus(await r.json())
      } catch {}
    }, 2000)
    return () => clearInterval(t)
  }, [])
  if (!status) return null
  return (
    <div className="p-2 rounded bg-gray-100 text-sm">
      <div>Training: {status.status}</div>
      {status.epoch && <div>Epoch: {status.epoch}</div>}
      {status.loss!=null && <div>Loss: {status.loss}</div>}
      {status.accuracy!=null && <div>Acc: {status.accuracy}</div>}
    </div>
  )
}
