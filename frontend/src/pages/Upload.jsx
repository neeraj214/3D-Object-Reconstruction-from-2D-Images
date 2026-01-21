import React, { useState } from 'react'
import { useToast, ToastContainer } from '../components/Toast.jsx'
import UploadBox from '../components/UploadBox.jsx'
import ResultPanel from '../components/ResultPanel.jsx'
import { motion } from 'framer-motion'

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
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header Section */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900">3D Object Reconstruction</h1>
        <p className="text-gray-500 max-w-2xl mx-auto">
          Upload a single 2D image of an object to generate a high-quality 3D point cloud model using our hybrid deep learning pipeline.
        </p>
      </div>

      {/* Main Upload Area */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <UploadBox onResult={handleResult} />
      </motion.div>

      {/* Results Section */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="border-t border-gray-200 pt-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Reconstruction Results</h2>
            <ResultPanel result={result} />
          </div>
        </motion.div>
      )}

      {/* Local Toast Container (fallback) */}
      {!setToast && <ToastContainer toasts={toasts} />}
    </div>
  )
}
