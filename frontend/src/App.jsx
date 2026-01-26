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

  const Home = () => {
    return (
      <div className="w-full">
        <section className="relative overflow-hidden rounded-3xl px-10 py-16 sm:px-14 sm:py-20 shadow-soft border border-indigo-100 bg-[#f8fafc]">
          <div className="absolute inset-0 pointer-events-none">
            <svg className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] opacity-40" viewBox="0 0 900 900" fill="none">
              <defs>
                <radialGradient id="meshAlt" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#67e8f9" stopOpacity="0.2" />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity="0.12" />
                </radialGradient>
              </defs>
              <circle cx="450" cy="450" r="420" stroke="url(#meshAlt)" strokeWidth="1" />
              <circle cx="450" cy="450" r="300" stroke="url(#meshAlt)" strokeWidth="1" />
              <circle cx="450" cy="450" r="180" stroke="url(#meshAlt)" strokeWidth="1" />
            </svg>
          </div>
          <motion.div 
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="relative grid grid-cols-1 md:grid-cols-2 gap-12 items-center"
          >
            <motion.div 
              className="space-y-7 relative z-10"
              initial={{ opacity: 0, x: -24 }} 
              animate={{ opacity: 1, x: 0 }} 
              transition={{ delay: 0.1, duration: 0.6 }}
            >
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/70 backdrop-blur-md border text-xs text-indigo-700 border-indigo-200">
                <span className="w-2 h-2 rounded-full bg-indigo-600"></span>
                <span>AI-powered 3D Reconstruction</span>
              </div>
              <h1 className="text-6xl font-extrabold tracking-tight text-gray-900">
                3D ReconstructAI
              </h1>
              <p className="text-lg text-slate-700 max-w-2xl">
                Transform a single 2D image into a detailed 3D point cloud using AI.
              </p>
              <div className="flex flex-wrap gap-4">
                <motion.a whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} className="inline-flex items-center gap-2 px-0">
                  <Link to="/upload" className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white font-medium shadow-lg transition-shadow shadow-[0_0_20px_rgba(99,102,241,0.4)] hover:shadow-[0_0_30px_rgba(99,102,241,0.6)]">
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M3 10h4l3-3 4 8 3-5h4"/></svg>
                    Start Reconstruction
                  </Link>
                </motion.a>
                <motion.a whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                  <a href="https://github.com/neeraj214" target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white/70 backdrop-blur-md text-gray-900 font-medium border border-gray-200 hover:border-gray-300 shadow-sm transition">
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5C5.73.5.9 5.33.9 11.6c0 4.87 3.16 8.99 7.53 10.45.55.11.75-.24.75-.53v-1.86c-3.07.67-3.72-1.3-3.72-1.3-.5-1.26-1.22-1.6-1.22-1.6-1-.68.08-.67.08-.67 1.12.08 1.71 1.15 1.71 1.15.99 1.7 2.6 1.21 3.23.93.1-.72.38-1.21.69-1.49-2.45-.28-5.02-1.22-5.02-5.45 0-1.2.43-2.18 1.14-2.95-.11-.28-.5-1.43.11-2.98 0 0 .94-.3 3.07 1.13.9-.25 1.86-.37 2.82-.38.96.01 1.92.13 2.82.38 2.13-1.43 3.07-1.13 3.07-1.13.62 1.55.23 2.7.11 2.98.71.77 1.14 1.75 1.14 2.95 0 4.24-2.59 5.17-5.06 5.44.39.34.74 1.01.74 2.05v3.04c0 .29.2.65.76.53 4.36-1.47 7.52-5.58 7.52-10.45C23.1 5.33 18.27.5 12 .5z"/></svg>
                    View on GitHub
                  </a>
                </motion.a>
              </div>
              <div className="mt-6 grid grid-cols-3 gap-4">
                {[
                  { k: 'Quality', v: 'High' },
                  { k: 'Points', v: '120k+' },
                  { k: 'Inference', v: '~2.1s' },
                ].map((m, i) => (
                  <div key={i} className="px-4 py-3 rounded-xl bg-white/70 backdrop-blur-md border border-gray-100 shadow-soft">
                    <div className="text-xs text-gray-500">{m.k}</div>
                    <div className="text-sm font-semibold text-gray-900">{m.v}</div>
                  </div>
                ))}
              </div>
            </motion.div>
            <motion.div 
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.15, duration: 0.6 }}
              className="relative"
            >
              <div className="relative rounded-3xl bg-white/60 backdrop-blur-xl border border-white/20 ring-1 ring-indigo-200 shadow-soft p-6">
                <div className="text-sm font-semibold text-gray-700 mb-4">Image → 3D Wireframe</div>
                <div className="grid grid-cols-2 gap-6 items-center">
                  <div className="relative rounded-2xl h-56 w-full overflow-hidden border border-indigo-100 bg-white/60">
                    <img
                      src="https://images.unsplash.com/photo-1549187774-b4e9b0445b06?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
                      alt="Luxury chair"
                      className="h-full w-full object-cover"
                      onError={(e) => { e.currentTarget.src = 'https://picsum.photos/id/1069/800/600'; }}
                    />
                    <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-35 mix-blend-soft-light" viewBox="0 0 200 200" fill="none">
                      <circle cx="100" cy="100" r="90" stroke="#22d3ee" strokeOpacity="0.35" strokeWidth="0.8" />
                      {[...Array(10)].map((_, i) => (
                        <ellipse key={i} cx="100" cy="100" rx="90" ry="28" transform={`rotate(${i*18} 100 100)`} stroke="#22d3ee" strokeOpacity="0.25" strokeWidth="0.7" />
                      ))}
                    </svg>
                    <div className="absolute inset-0 bg-gradient-to-tr from-cyan-400/10 to-indigo-500/10"></div>
                  </div>
                  <motion.div 
                    initial={{ rotate: -8 }} 
                    animate={{ rotate: 8 }} 
                    transition={{ repeat: Infinity, repeatType: 'reverse', duration: 6, ease: 'easeInOut' }}
                    className="relative h-56 rounded-2xl ring-1 ring-indigo-100 bg-gradient-to-r from-indigo-50 to-violet-50 overflow-hidden flex items-center justify-center"
                  >
                    <svg className="w-[95%] h-[95%]" viewBox="0 0 200 200" fill="none">
                      <circle cx="100" cy="100" r="80" stroke="#22d3ee" strokeOpacity="0.5" strokeWidth="1" />
                      {[...Array(8)].map((_, i) => (
                        <ellipse key={i} cx="100" cy="100" rx="80" ry="30" transform={`rotate(${i*22.5} 100 100)`} stroke="#22d3ee" strokeOpacity="0.35" strokeWidth="0.8" />
                      ))}
                      {[...Array(6)].map((_, i) => (
                        <circle key={i} cx="100" cy={40 + i*20} r="1.2" fill="#22d3ee" opacity="0.8" />
                      ))}
                    </svg>
                    <div className="absolute inset-0 bg-gradient-to-tr from-cyan-400/10 to-indigo-500/10"></div>
                  </motion.div>
                </div>
                <svg className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10" viewBox="0 0 24 24" fill="none">
                  <defs>
                    <linearGradient id="arrowGradAlt" x1="0" y1="0" x2="24" y2="0">
                      <stop offset="0%" stopColor="#22d3ee"/>
                      <stop offset="100%" stopColor="#6366F1"/>
                    </linearGradient>
                  </defs>
                  <path d="M5 12h9M12 7l5 5-5 5" stroke="url(#arrowGradAlt)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
            </motion.div>
          </motion.div>
        </section>

        <section className="mt-14 grid grid-cols-1 sm:grid-cols-3 gap-6 relative">
          <div className="absolute -z-10 left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 w-[700px] h-[700px] rounded-full blur-3xl bg-gradient-to-br from-indigo-300/20 to-cyan-300/20"></div>
          {[
            { title: 'AI-powered 3D Reconstruction', desc: 'Single-view inference with learned geometry priors', icon: (<svg className="w-6 h-6 text-indigo-600" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeWidth="2" d="M3 8l9-5 9 5-9 5-9-5z"/><path strokeWidth="2" d="M21 8v8l-9 5-9-5V8"/></svg>) },
            { title: 'Adjustable Quality & Precision', desc: 'Tune modes and parameters for speed vs detail', icon: (<svg className="w-6 h-6 text-indigo-600" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeWidth="2" d="M12 6v12M6 12h12"/></svg>) },
            { title: 'Background Segmentation Support', desc: 'Focus on the object to improve reconstruction', icon: (<svg className="w-6 h-6 text-indigo-600" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeWidth="2" d="M9 12l2 2 4-4M3 7h18v10H3z"/></svg>) },
          ].map((f, idx) => (
            <motion.div 
              key={idx} 
              initial={{ opacity: 0, y: 12 }} 
              whileInView={{ opacity: 1, y: 0 }} 
              viewport={{ once: true, margin: '-50px' }}
              whileHover={{ y: -8 }} 
              transition={{ duration: 0.25 }} 
              className="p-6 rounded-2xl bg-white border border-indigo-100 shadow-soft"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-indigo-50 flex items-center justify-center">
                  <span className="text-cyan-600">{f.icon}</span>
                </div>
                <div className="font-semibold text-gray-900">{f.title}</div>
              </div>
              <div className="mt-2 text-sm text-gray-600">{f.desc}</div>
            </motion.div>
          ))}
        </section>

        <section className="mt-14 p-8 rounded-2xl bg-white/70 backdrop-blur-md border border-gray-100 shadow-soft">
          <div className="text-sm font-semibold text-gray-700 mb-4">How It Works</div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            {[
              { t: 'Upload Image', i: (<svg className="w-6 h-6 text-indigo-600" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16"/><path strokeWidth="2" d="M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>), d: 'Drop a photo or browse from your device' },
              { t: 'AI Reconstruction', i: (<svg className="w-6 h-6 text-indigo-600" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeWidth="2" d="M12 3l9 5-9 5-9-5 9-5z"/><path strokeWidth="2" d="M12 13l9 5-9 5-9-5 9-5z"/></svg>), d: 'Generate a high-quality 3D point cloud' },
              { t: 'View / Export 3D Model', i: (<svg className="w-6 h-6 text-indigo-600" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeWidth="2" d="M4 4h16v12H4z"/><path strokeWidth="2" d="M8 20h8"/></svg>), d: 'Inspect results and download for use elsewhere' },
            ].map((s, idx) => (
              <motion.div 
                key={idx} 
                initial={{ opacity: 0, y: 12 }} 
                whileInView={{ opacity: 1, y: 0 }} 
                viewport={{ once: true, margin: '-50px' }}
                whileHover={{ scale: 1.02 }} 
                transition={{ type: 'spring', stiffness: 250, damping: 20 }} 
                className="p-4 rounded-lg border border-gray-200 bg-surface-soft"
              >
                <div className="flex items-center gap-3">
                  {s.i}
                  <div className="font-medium text-gray-900">{s.t}</div>
                </div>
                <div className="mt-1 text-sm text-gray-600">{s.d}</div>
              </motion.div>
            ))}
          </div>
        </section>
        
        <section className="mt-14 text-center">
          <motion.div 
            initial={{ opacity: 0, scale: 0.98 }} 
            whileInView={{ opacity: 1, scale: 1 }} 
            viewport={{ once: true, margin: '-50px' }}
            className="inline-flex items-center justify-between gap-4 px-8 py-6 rounded-2xl bg-white/70 backdrop-blur-md border border-gray-100 shadow-soft"
          >
            <div className="text-lg font-semibold text-gray-900">Ready to reconstruct your first 3D model?</div>
            <Link to="/upload" className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white font-medium shadow-lg hover:shadow-brand transition">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M3 10h4l3-3 4 8 3-5h4"/></svg>
              Upload Image
            </Link>
          </motion.div>
        </section>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50/40 to-white text-gray-900 font-sans flex flex-col">
      {/* Navigation Bar */}
      <nav className="bg-white/90 backdrop-blur-md border-b border-gray-200 shadow-sm sticky top-0 z-50">
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
                    className={`relative inline-flex items-center px-1 pt-1 text-sm font-medium transition-colors duration-200 ${
                      location.pathname.startsWith(link.path) || (link.path === '/upload' && location.pathname === '/')
                        ? 'text-gray-900'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    {link.name}
                    {(location.pathname.startsWith(link.path) || (link.path === '/upload' && location.pathname === '/')) && (
                      <motion.span
                        layoutId="active-underline"
                        className="absolute -bottom-1 left-0 h-0.5 w-full bg-indigo-600 rounded-full"
                      />
                    )}
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
              <Route path="/" element={<Home />} />
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
