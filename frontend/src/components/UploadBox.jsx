import React, { useState } from 'react'
import ImageUpload from './ImageUpload.jsx'
import { predict, reconstructMesh } from '../api/backend'

export default function UploadBox({ onResult }) {
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  
  // Settings
  const [useSegmentation, setUseSegmentation] = useState(true)
  const [fScale, setFScale] = useState(1.1)
  const [mode, setMode] = useState('quality') // Default to quality for better results
  const [nPoints, setNPoints] = useState(20000)

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
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <div className="p-6 border-b border-gray-100">
        <h3 className="text-lg font-semibold text-gray-800">New Reconstruction</h3>
        <p className="text-sm text-gray-500 mt-1">Upload a 2D image to generate a 3D point cloud.</p>
      </div>
      
      <div className="p-6 space-y-6">
        {/* Image Upload Area */}
        <ImageUpload onSubmit={f => { setFile(f); setErr(null); }} />

        {/* Configuration Panel */}
        <div className="bg-gray-50 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
             <span className="text-sm font-medium text-gray-700">Settings</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Mode Selection */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Quality Mode</label>
              <div className="flex bg-white rounded-md shadow-sm p-1 border border-gray-200">
                {['fast', 'balanced', 'quality', 'mesh'].map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`flex-1 text-sm py-1.5 rounded transition-all ${
                      mode === m 
                        ? 'bg-blue-600 text-white shadow-sm' 
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Focal Scale Slider */}
            <div className="space-y-2">
               <div className="flex justify-between">
                  <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Focal Scale</label>
                  <span className="text-xs font-mono text-blue-600 bg-blue-50 px-2 py-0.5 rounded">{fScale.toFixed(2)}</span>
               </div>
               <input 
                 type="range" 
                 min="0.8" 
                 max="1.4" 
                 step="0.01" 
                 value={fScale} 
                 onChange={e => setFScale(parseFloat(e.target.value))}
                 className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
               />
            </div>

             {/* Point Count */}
             <div className="space-y-2">
               <div className="flex justify-between">
                  <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Point Count</label>
                  <span className="text-xs font-mono text-blue-600 bg-blue-50 px-2 py-0.5 rounded">{nPoints.toLocaleString()}</span>
               </div>
               <input 
                 type="range" 
                 min="5000" 
                 max="50000" 
                 step="1000" 
                 value={nPoints} 
                 onChange={e => setNPoints(parseInt(e.target.value))}
                 className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
               />
            </div>

            {/* Toggles */}
            <div className="flex items-center space-x-3 pt-6">
               <input 
                 id="seg-toggle"
                 type="checkbox" 
                 checked={useSegmentation} 
                 onChange={e => setUseSegmentation(e.target.checked)}
                 className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
               />
               <label htmlFor="seg-toggle" className="text-sm text-gray-700 cursor-pointer select-none">
                 Enable Background Removal (Segmentation)
               </label>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-2">
          <button 
            className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-all transform active:scale-[0.98] flex items-center justify-center gap-2 ${
              busy || !file 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700 shadow-lg hover:shadow-blue-500/30'
            }`}
            onClick={submit}
            disabled={busy || !file}
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
