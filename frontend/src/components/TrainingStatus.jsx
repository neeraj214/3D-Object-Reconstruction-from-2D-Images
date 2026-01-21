import React, { useEffect, useState } from 'react'
import { api } from '../services/apiClient'
import { motion } from 'framer-motion'

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
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="p-4 rounded-xl bg-white border border-gray-200 shadow-sm flex items-center justify-between"
    >
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          <div className="absolute inset-0 bg-green-500 rounded-full opacity-30 animate-ping"></div>
        </div>
        <div>
          <div className="text-sm font-semibold text-gray-900">Training Active</div>
          <div className="text-xs text-gray-500">{status.status}</div>
        </div>
      </div>
      
      <div className="flex gap-4 text-sm">
        {status.epoch && (
          <div className="flex flex-col items-end">
            <span className="text-gray-500 text-xs">Epoch</span>
            <span className="font-mono font-medium">{status.epoch}</span>
          </div>
        )}
        {status.loss != null && (
          <div className="flex flex-col items-end">
            <span className="text-gray-500 text-xs">Loss</span>
            <span className="font-mono font-medium text-red-600">{Number(status.loss).toFixed(4)}</span>
          </div>
        )}
        {status.accuracy != null && (
          <div className="flex flex-col items-end">
            <span className="text-gray-500 text-xs">Accuracy</span>
            <span className="font-mono font-medium text-blue-600">{Number(status.accuracy).toFixed(4)}</span>
          </div>
        )}
      </div>
    </motion.div>
  )
}
