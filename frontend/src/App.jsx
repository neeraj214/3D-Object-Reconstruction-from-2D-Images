import React, { useState, useEffect } from 'react'
import { getCategories as getDatasetsList } from './api'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Upload from './pages/Upload.jsx'
import DatasetBrowser from './pages/DatasetBrowser.jsx'
import CategoryView from './pages/CategoryView.jsx'

export default function App() {
  const [toast, setToast] = useState(null)
  const [datasets, setDatasets] = useState(null)
  const location = useLocation()

  useEffect(() => {
    (async () => {
      try { const cats = await getDatasetsList(); setDatasets(cats) } catch {}
    })();
  }, []);

  const navLinks = [
    { name: 'Upload & Reconstruct', path: '/upload' },
    { name: 'Dataset Browser', path: '/datasets' },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50/40 to-white text-gray-900 font-sans flex flex-col">
      {/* Navigation Bar */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 sm:h-20">
            <div className="flex items-center">
              <Link to="/" className="flex-shrink-0 flex flex-col">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-lg">
                    3D
                  </div>
                  <span className="font-bold text-xl tracking-tight text-gray-900">Reconstruct<span className="text-blue-600">AI</span></span>
                </div>
                <span className="text-xs text-brand-muted mt-0.5">Single‑view 3D reconstruction</span>
              </Link>
              <div className="hidden sm:ml-8 sm:flex sm:space-x-8">
                {navLinks.map((link) => (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200 ${
                      location.pathname.startsWith(link.path) || (link.path === '/upload' && location.pathname === '/')
                        ? 'border-blue-500 text-gray-900'
                        : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    }`}
                  >
                    {link.name}
                  </Link>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-3">
               <div className="text-sm text-gray-400">v1.0.0</div>
               <div className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-green-50 text-green-700 border border-green-200">
                 <span className="inline-block w-2 h-2 rounded-full bg-green-500"></span>
                 <span>API</span>
               </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Toast Notification */}
        <AnimatePresence>
          {toast && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`fixed top-20 right-4 z-50 px-4 py-3 rounded-lg shadow-lg border flex items-center gap-3 ${
                toast.type === 'error' 
                  ? 'bg-red-50 border-red-200 text-red-700' 
                  : 'bg-green-50 border-green-200 text-green-700'
              }`}
            >
              <span>{toast.message}</span>
              <button onClick={() => setToast(null)} className="opacity-50 hover:opacity-100">×</button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Page Transition Wrapper */}
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="w-full"
          >
            <Routes location={location}>
              <Route path="/" element={<Upload setToast={setToast} />} />
              <Route path="/upload" element={<Upload setToast={setToast} />} />
              <Route path="/datasets" element={<DatasetBrowser datasets={datasets} />} />
              <Route path="/datasets/:dataset/:category" element={<CategoryView />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-400">
            © 2025 3D Reconstruction Project. Powered by PyTorch & Three.js.
          </p>
        </div>
      </footer>
    </div>
  )
}
