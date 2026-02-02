import React, { useState } from 'react'
import { useToast, ToastContainer } from '../components/Toast.jsx'
import UploadBox from '../components/UploadBox.jsx'
import ResultPanel from '../components/ResultPanel.jsx'
import { motion, AnimatePresence } from 'framer-motion'

export default function Upload({ setToast }) {
  const [result, setResult] = useState(null)

  // Wrapper to sync local toast with global toast if provided, or use local
  const { toasts, push } = useToast()

  const handleResult = (r) => {
    setResult(r)
    const msg = `Reconstruction complete! (${r.num_points} points)`
    if (setToast) setToast({ message: msg, type: 'success' })
    else push(msg, 'success')
  }

  return (
    <div className="max-w-[1400px] mx-auto space-y-8">
      {/* Header Section */}
      <div className="text-left space-y-2 border-b border-white/5 pb-6">
        <h1 className="text-3xl font-display font-bold text-white tracking-tight flex items-center gap-3">
          <span className="p-2 rounded-lg bg-brand-primary/10 text-brand-primary">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
          </span>
          Reconstruction Studio
        </h1>
        <p className="text-gray-400 text-sm max-w-2xl pl-12">
          Deconstruct reality. Upload your 2D image to initialize the hybrid neural pipeline.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Upload & Config */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-5 space-y-6"
        >
          <UploadBox onResult={handleResult} />

          {!result && (
            <div className="p-6 rounded-2xl bg-surface-glass border border-white/5 text-center space-y-4">
              <div className="text-sm font-semibold text-gray-500 uppercase tracking-wider">Accepted Formats</div>
              <div className="flex justify-center gap-4 text-gray-400">
                <span className="px-3 py-1 rounded bg-white/5 border border-white/10 text-xs">.JPG</span>
                <span className="px-3 py-1 rounded bg-white/5 border border-white/10 text-xs">.PNG</span>
                <span className="px-3 py-1 rounded bg-white/5 border border-white/10 text-xs">.WEBP</span>
              </div>
            </div>
          )}
        </motion.div>

        {/* Right Column: Visualization Area */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-7"
        >
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="bg-surface-glass backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden shadow-2xl"
              >
                <div className="p-6 border-b border-white/10 flex justify-between items-center bg-brand-darker/30">
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                    Live View
                  </h2>
                  <div className="text-xs font-mono text-gray-400">SESSION_ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</div>
                </div>
                <div className="p-6">
                  <ResultPanel result={result} />
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="placeholder"
                initial={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-[600px] rounded-2xl border-2 border-dashed border-white/5 bg-surface-soft/20 flex flex-col items-center justify-center p-12 text-center"
              >
                <div className="w-24 h-24 rounded-full bg-brand-darker border border-brand-primary/20 flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(59,130,246,0.1)]">
                  <svg className="w-10 h-10 text-brand-primary/50" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="1" d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" /></svg>
                </div>
                <h3 className="text-xl font-bold text-gray-300 mb-2">Ready to Render</h3>
                <p className="text-gray-500 max-w-sm">
                  Configure your parameters on the left and start the reconstruction process to see the 3D output here.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Local Toast Container (fallback) */}
      {!setToast && <ToastContainer toasts={toasts} />}
    </div>
  )
}
