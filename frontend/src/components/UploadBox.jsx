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
    <div className="bg-white rounded-xl shadow-soft border border-gray-100 overflow-hidden">
      <div className="p-8 border-b border-gray-100">
        <h3 className="text-2xl font-bold text-gray-900">Start a Reconstruction</h3>
        <p className="text-sm text-gray-600 mt-2">Upload an image and generate a high-quality 3D point cloud.</p>
      </div>
      
      <div className="p-8 space-y-8">
        {/* Image Upload Area */}
        <div className="relative group rounded-2xl ring-1 ring-blue-100 hover:ring-blue-300 transition-all shadow-soft hover:shadow-brand p-4 min-h-48">
          <ImageUpload onSubmit={f => { setFile(f); setErr(null); }} />
          {busy && (
            <div className="absolute inset-0 z-10 rounded-xl bg-gradient-to-r from-gray-100 to-gray-200 opacity-60 animate-pulse pointer-events-none"></div>
          )}
        </div>

        {/* Configuration Panel */}
        <div className="bg-gray-50 rounded-xl p-6 space-y-6">
          <div className="flex items-center justify-between">
             <span className="text-base font-semibold text-gray-800">Reconstruction Controls</span>
             <button 
               type="button" 
               className="text-xs px-2 py-1 rounded-md bg-white border border-gray-200 text-gray-600 hover:bg-gray-100"
               onClick={() => setShowAdvanced(v => !v)}
               aria-expanded={showAdvanced}
               title={showAdvanced ? 'Hide advanced options' : 'Show advanced options'}
             >
               {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
             </button>
          </div>
         
          <div className="space-y-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Basic</div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Mode Selection */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider" title="Choose speed vs quality tradeoff">Quality Mode</label>
                <span className="text-xs text-gray-500">{mode === 'mesh' ? 'Mesh Output' : 'Point Cloud'}</span>
              </div>
              <div className="flex bg-white rounded-md shadow-sm p-1 border border-gray-200">
                {['fast', 'balanced', 'quality', 'mesh'].map((m) => (
                  <motion.button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`flex-1 text-sm py-1.5 rounded transition-all ${
                      mode === m 
                        ? 'bg-blue-600 text-white shadow-sm' 
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    layout
                    transition={{ type: 'spring', stiffness: 350, damping: 25 }}
                  >
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Focal Scale Slider */}
            <div className="space-y-2">
               <div className="flex justify-between">
                  <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider" title="Adjust synthetic camera focal length">Focal Scale</label>
                  <motion.span 
                    key={fScale}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-xs font-mono text-blue-600 bg-blue-50 px-2 py-0.5 rounded"
                  >
                    {fScale.toFixed(2)}
                  </motion.span>
               </div>
               <motion.input 
                 type="range" 
                 min="0.8" 
                 max="1.4" 
                 step="0.01" 
                 value={fScale} 
                 onChange={e => setFScale(parseFloat(e.target.value))}
                 className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                 title="Higher values increase synthetic focal length and perceived depth"
                 whileHover={{ scaleY: 1.05 }}
                 transition={{ type: 'spring', stiffness: 250, damping: 20 }}
               />
            </div>

             {/* Point Count */}
             <div className="space-y-2">
               <div className="flex justify-between">
                  <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider" title="Number of points in output point cloud">Point Count</label>
                  <motion.span
                    key={nPoints}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-xs font-mono text-blue-600 bg-blue-50 px-2 py-0.5 rounded"
                  >
                    {nPoints.toLocaleString()}
                  </motion.span>
               </div>
               <motion.input 
                 type="range" 
                 min="5000" 
                 max="50000" 
                 step="1000" 
                 value={nPoints} 
                 onChange={e => setNPoints(parseInt(e.target.value))}
                 className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                 title="Increase for denser point clouds, at the cost of time"
                 whileHover={{ scaleY: 1.05 }}
                 transition={{ type: 'spring', stiffness: 250, damping: 20 }}
               />
            </div>

            {/* Advanced Options */}
            <AnimatePresence initial={false}>
            {showAdvanced && (
              <motion.div 
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25, ease: 'easeOut' }}
                className="md:col-span-2 space-y-3 pt-4"
              >
                <div className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">Advanced</div>
                <div className="flex items-center space-x-3">
                  <motion.input 
                    id="seg-toggle"
                    type="checkbox" 
                    checked={useSegmentation} 
                    onChange={e => setUseSegmentation(e.target.checked)}
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    whileTap={{ scale: 0.9 }}
                  />
                  <motion.label 
                    htmlFor="seg-toggle" 
                    className="text-sm text-gray-700 cursor-pointer select-none" 
                    title="Removes background to improve reconstruction focus"
                    initial={false}
                    animate={{ color: useSegmentation ? '#1f2937' : '#6b7280' }}
                    transition={{ duration: 0.2 }}
                  >
                    Enable Background Removal (Segmentation)
                  </motion.label>
                  {useSegmentation && (
                    <span className="relative inline-flex">
                      <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                      <span className="absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-40 animate-ping"></span>
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500">Advanced options may increase processing time.</p>
              </motion.div>
            )}
            </AnimatePresence>
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-2">
          <button 
            className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-all transform active:scale-[0.98] focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-200 flex items-center justify-center gap-2 ${
              busy || !file 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-gradient-to-r from-brand-primary to-blue-700 shadow-lg hover:shadow-brand'
            }`}
            onClick={submit}
            disabled={busy || !file}
            aria-busy={busy}
          >
            {busy ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Processing...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                <span>Generate 3D Model</span>
              </>
            )}
          </button>
        </div>

        {/* Error Message */}
        {err && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm flex items-center gap-2 animate-fadeIn">
            <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            {err}
          </div>
        )}
      </div>
    </div>
  )
}
