import React, { useState } from 'react'
import ImageUpload from './ImageUpload.jsx'
import { predict, reconstructMesh } from '../api/backend'
import { motion, AnimatePresence } from 'framer-motion'

export default function UploadBox({ onResult }) {
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)

  // Settings
  const [useSegmentation, setUseSegmentation] = useState(true)
  const [fScale, setFScale] = useState(1.1)
  const [mode, setMode] = useState('quality') // Default to quality for better results
  const [nPoints, setNPoints] = useState(20000)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const submit = async () => {
    if (!file) {
      setErr("Please select an image first.")
      return
    }
    setBusy(true);
    setErr(null)
    try {
      let r;
      if (mode === 'mesh') {
        r = await reconstructMesh(file, { nPoints })
      } else {
        r = await predict(file, { fScale, useSegmentation, mode, nPoints })
      }

      if (r.status === 'error') throw new Error(r.message)
      onResult(r)
    } catch (e) {
      console.error(e)
      setErr(e.message || 'Reconstruction failed. Please check backend connection.')
    }
    setBusy(false)
  }

  return (
    <div className="bg-surface-glass backdrop-blur-xl rounded-3xl shadow-2xl border border-white/10 overflow-hidden relative">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-brand-primary via-brand-accent to-brand-primary animate-gradient-x"></div>

      <div className="p-8 border-b border-white/10 flex justify-between items-start">
        <div>
          <h3 className="text-2xl font-display font-bold text-white">Input Source</h3>
          <p className="text-sm text-gray-400 mt-1">Upload an image to begin reconstruction.</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10">
          <div className={`w-2 h-2 rounded-full ${busy ? 'bg-amber-500 animate-pulse' : 'bg-green-500'}`}></div>
          <span className="text-[10px] uppercase font-bold tracking-wider text-gray-400">{busy ? 'PROCESSING' : 'READY'}</span>
        </div>
      </div>

      <div className="p-8 space-y-8">
        {/* Image Upload Area */}
        <div className="relative group rounded-2xl ring-1 ring-white/10 hover:ring-brand-primary/50 transition-all shadow-lg hover:shadow-brand/20 p-6 min-h-[240px] bg-surface-soft/20 flex flex-col items-center justify-center">
          <ImageUpload onSubmit={f => { setFile(f); setErr(null); }} />
          {busy && (
            <div className="absolute inset-0 z-10 rounded-2xl bg-surface-dark/60 backdrop-blur-sm flex items-center justify-center">
              <div className="bg-black/50 p-6 rounded-2xl border border-white/10 backdrop-blur-md flex flex-col items-center gap-4">
                <svg className="animate-spin h-10 w-10 text-brand-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span className="text-brand-accent font-mono text-sm animate-pulse">Computing Voxel Space...</span>
              </div>
            </div>
          )}
        </div>

        {/* Configuration Panel */}
        <div className="bg-surface-soft/30 rounded-2xl p-6 space-y-8 border border-white/5">
          <div className="flex items-center justify-between border-b border-white/5 pb-4">
            <span className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <svg className="w-4 h-4 text-brand-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" /></svg>
              Parameters
            </span>
          </div>

          <div className="space-y-6">
            {/* Mode Selection Grid */}
            <div className="space-y-3">
              <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider block">Reconstruction Mode</label>
              <div className="grid grid-cols-2 gap-3">
                {['fast', 'balanced', 'quality', 'mesh'].map((m) => (
                  <motion.button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`text-sm py-3 px-4 rounded-xl transition-all border text-left relative overflow-hidden group ${mode === m
                      ? 'bg-brand-primary/10 border-brand-primary text-white'
                      : 'bg-surface-soft/50 border-white/5 text-gray-500 hover:border-white/20 hover:text-gray-300'
                      }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="relative z-10 flex flex-col">
                      <span className="font-bold text-sm tracking-wide">{m.charAt(0).toUpperCase() + m.slice(1)}</span>
                      <span className="text-[10px] opacity-60 font-mono mt-0.5">
                        {m === 'fast' && 'Low res, high speed'}
                        {m === 'balanced' && 'Standard balance'}
                        {m === 'quality' && 'High range details'}
                        {m === 'mesh' && 'Polygon mesh gen'}
                      </span>
                    </div>
                    {mode === m && (
                      <div className="absolute inset-0 bg-gradient-to-r from-brand-primary/0 via-brand-primary/10 to-brand-primary/0 animate-shimmer"></div>
                    )}
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Sliders */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Focal Scale</label>
                  <span className="text-xs font-mono text-brand-accent">{fScale.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.8" max="1.4" step="0.01"
                  value={fScale}
                  onChange={e => setFScale(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-surface-dark rounded-lg appearance-none cursor-pointer accent-brand-accent hover:accent-brand-primary transition-all"
                />
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Point Density</label>
                  <span className="text-xs font-mono text-brand-accent">{(nPoints / 1000).toFixed(0)}k</span>
                </div>
                <input
                  type="range"
                  min="5000" max="50000" step="1000"
                  value={nPoints}
                  onChange={e => setNPoints(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-surface-dark rounded-lg appearance-none cursor-pointer accent-brand-accent hover:accent-brand-primary transition-all"
                />
              </div>
            </div>

            <button
              type="button"
              className="text-xs flex items-center gap-2 text-gray-500 hover:text-white transition-colors py-2"
              onClick={() => setShowAdvanced(v => !v)}
            >
              <span className={`transform transition-transform duration-200 ${showAdvanced ? 'rotate-90' : ''}`}>▶</span>
              Advanced Configurations
            </button>

            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="pt-2 border-t border-white/5 overflow-hidden"
                >
                  <div className="p-3 bg-black/20 rounded-xl border border-white/5 flex items-center justify-between">
                    <span className="text-sm text-gray-300">Background Segmentation</span>
                    <motion.input
                      type="checkbox"
                      checked={useSegmentation}
                      onChange={e => setUseSegmentation(e.target.checked)}
                      className="w-5 h-5 text-brand-primary rounded focus:ring-brand-primary bg-surface-dark border-white/10"
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-2">
          <button
            className={`w-full py-4 px-6 rounded-xl font-bold text-lg text-white transition-all transform active:scale-[0.98] relative overflow-hidden group shadow-glow-lg ${busy || !file
              ? 'bg-gray-600/50 cursor-not-allowed grayscale'
              : 'bg-gradient-to-r from-brand-primary via-blue-500 to-brand-accent'
              }`}
            onClick={submit}
            disabled={busy || !file}
          >
            <span className="relative z-10 flex items-center justify-center gap-3">
              {!busy && <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>}
              {busy ? 'Engaging Neural Net...' : 'Initialize Reconstruction'}
            </span>
            {!busy && <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>}
          </button>
        </div>

        {/* Error Message */}
        {err && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-200 text-sm flex items-center gap-3"
          >
            <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            {err}
          </motion.div>
        )}
      </div>
    </div>
  )
}
