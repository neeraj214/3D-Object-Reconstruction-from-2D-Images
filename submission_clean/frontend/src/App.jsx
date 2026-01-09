import React, { useState, useEffect } from 'react'
import { getCategories as getDatasetsList } from './api'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Upload from './pages/Upload.jsx'
import DatasetBrowser from './pages/DatasetBrowser.jsx'
import CategoryView from './pages/CategoryView.jsx'

export default function App() {
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState(null)
  const [datasets, setDatasets] = useState(null)
  const location = useLocation()

  useEffect(() => {
    (async () => {
      try { const cats = await getDatasetsList(); setDatasets(cats) } catch {}
    })();
  }, []);

  return (
    <div className="p-4 font-sans">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold">3D Reconstruction</h2>
        <div className="flex gap-2">
          <Link className="btn" to="/upload">Upload</Link>
          <Link className="btn" to="/datasets">Dataset Browser</Link>
        </div>
      </div>
      {toast && <div className={`p-2 rounded ${toast.type==='error' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>{toast.message}</div>}
      <AnimatePresence mode="wait">
        <motion.div key={location.pathname} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
          <Routes location={location}>
            <Route path="/" element={<Upload />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/datasets" element={<DatasetBrowser datasets={datasets} />} />
            <Route path="/datasets/:dataset/:category" element={<CategoryView />} />
          </Routes>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
