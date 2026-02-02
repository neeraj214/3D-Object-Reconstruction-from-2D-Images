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
      try { const cats = await getDatasetsList(); setDatasets(cats) } catch { }
    })();
  }, []);

  const navLinks = [
    { name: 'Upload & Reconstruct', path: '/upload' },
    { name: 'Dataset Browser', path: '/datasets' },
  ]

  const Home = () => {
    return (
      <div className="w-full relative z-10">
        <section className="relative overflow-hidden rounded-3xl px-8 py-16 sm:px-14 sm:py-24 border border-white/10 bg-surface-glass shadow-2xl">
          {/* Animated Glow Background */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-3xl">
            <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] bg-brand-primary/20 rounded-full blur-[100px] animate-blob" />
            <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] bg-brand-secondary/20 rounded-full blur-[100px] animate-blob animation-delay-2000" />
            <div className="absolute top-[40%] left-[40%] w-[300px] h-[300px] bg-brand-accent/20 rounded-full blur-[100px] animate-blob animation-delay-4000" />
          </div>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="relative grid grid-cols-1 md:grid-cols-2 gap-16 items-center"
          >
            <motion.div
              className="space-y-8 relative z-10"
              initial={{ opacity: 0, x: -24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1, duration: 0.6 }}
            >
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface-soft/50 backdrop-blur-md border border-white/10 text-xs text-brand-accent font-medium">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-accent opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-accent"></span>
                </span>
                <span>AI V4.0 Engine Ready</span>
              </div>

              <h1 className="text-5xl sm:text-7xl font-display font-extrabold tracking-tight text-white leading-tight">
                3D Reality <br />
                <span className="text-gradient">Reimagined</span>
              </h1>

              <p className="text-lg text-gray-400 max-w-xl leading-relaxed">
                Transform single 2D images into high-fidelity 3D point clouds with our state-of-the-art hybrid neural architecture.
              </p>

              <div className="flex flex-wrap gap-4 pt-2">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Link to="/upload" className="btn btn-primary text-base px-8 py-3.5">
                    <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                    Start Reconstructing
                  </Link>
                </motion.div>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <a href="https://github.com/neeraj214" target="_blank" rel="noreferrer" className="btn btn-secondary text-base px-8 py-3.5">
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" /></svg>
                    GitHub
                  </a>
                </motion.div>
              </div>

              <div className="grid grid-cols-3 gap-4 pt-4 border-t border-white/10">
                {[
                  { k: 'Resolution', v: 'High-Fidelity' },
                  { k: 'Points', v: '120k+' },
                  { k: 'Inference', v: '< 2.1s' },
                ].map((m, i) => (
                  <div key={i} className="flex flex-col">
                    <span className="text-xs text-gray-400 uppercase tracking-wider">{m.k}</span>
                    <span className="text-lg font-bold text-white font-display">{m.v}</span>
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
              <div className="relative rounded-2xl bg-surface-soft/40 backdrop-blur-xl border border-white/10 p-2 shadow-2xl ring-1 ring-white/10">
                <div className="relative aspect-square rounded-xl overflow-hidden bg-brand-darker">
                  <div className="absolute inset-0 bg-gradient-to-tr from-brand-primary/20 via-transparent to-brand-accent/20"></div>

                  {/* Abstract Tech Visualization */}
                  <svg className="absolute inset-0 w-full h-full p-8" viewBox="0 0 400 400" fill="none">
                    <defs>
                      <linearGradient id="grid-grad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.1" />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                      </linearGradient>
                      <filter id="glow-filter">
                        <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                        <feMerge>
                          <feMergeNode in="coloredBlur" />
                          <feMergeNode in="SourceGraphic" />
                        </feMerge>
                      </filter>
                    </defs>

                    {/* Grid Floor */}
                    <path d="M50 300 L350 300 L400 400 L0 400 Z" fill="url(#grid-grad)" />
                    <path d="M50 300 L350 300" stroke="#3b82f6" strokeWidth="1" strokeOpacity="0.3" />
                    {[...Array(9)].map((_, i) => (
                      <path key={i} d={`M${50 + i * 37.5} 300 L${(i - 4) * 20 + 200} 400`} stroke="#3b82f6" strokeWidth="1" strokeOpacity="0.2" />
                    ))}

                    {/* Central Object (Abstract Cube) */}
                    <g transform="translate(200 200)" filter="url(#glow-filter)">
                      <motion.g
                        animate={{ rotateY: 360 }}
                        transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                      >
                        <path d="M-60 -60 L60 -60 L60 60 L-60 60 Z" stroke="#06b6d4" strokeWidth="2" fill="none" opacity="0.8" />
                        <path d="M-40 -40 L40 -40 L40 40 L-40 40 Z" stroke="#6366f1" strokeWidth="2" fill="none" opacity="0.6" />
                        <circle cx="0" cy="0" r="10" fill="#3b82f6" />
                        {[...Array(6)].map((_, i) => (
                          <motion.circle
                            key={i}
                            r="2"
                            fill="#fff"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: [0, 1, 0], cx: [0, Math.cos(i) * 100], cy: [0, Math.sin(i) * 100] }}
                            transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
                          />
                        ))}
                      </motion.g>
                    </g>
                  </svg>

                  <div className="absolute bottom-6 left-6 right-6">
                    <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-brand-accent shadow-[0_0_10px_rgba(6,182,212,0.8)]"
                        animate={{ width: ["0%", "100%"] }}
                        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                      />
                    </div>
                    <div className="flex justify-between mt-2 text-xs text-brand-accent font-mono">
                      <span>SCANNING_GEOMETRY</span>
                      <span>100%</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </section>

        <section className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              title: 'Hybrid Architecture',
              desc: 'CNN + Transformer fusion for robust global and local geometry extraction.',
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              )
            },
            {
              title: 'Precision Controls',
              desc: 'Fine-tune point density, smoothing factors, and post-processing filters.',
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              )
            },
            {
              title: 'Smart Segmentation',
              desc: 'Automatic background removal ensures clean object reconstruction.',
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )
            },
          ].map((f, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-50px' }}
              transition={{ delay: idx * 0.1 }}
              className="p-6 rounded-2xl bg-surface-soft/30 border border-white/5 hover:bg-surface-soft/50 hover:border-brand-primary/30 transition-all duration-300 group"
            >
              <div className="w-12 h-12 rounded-lg bg-brand-primary/10 flex items-center justify-center text-brand-primary group-hover:bg-brand-primary group-hover:text-white transition-colors duration-300 mb-4">
                {f.icon}
              </div>
              <h3 className="text-xl font-bold text-white mb-2">{f.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </section>

        <section className="mt-24 text-center pb-12">
          <div className="relative inline-block">
            <div className="absolute inset-0 bg-brand-secondary blur-2xl opacity-20 rounded-full"></div>
            <h2 className="relative text-3xl font-display font-bold text-white mb-4">Ready for the Next Dimension?</h2>
          </div>

          <p className="text-gray-400 max-w-2xl mx-auto mb-8">
            Join researchers and developers using our tool for AR/VR asset creation and robotic perception.
          </p>

          <Link to="/upload" className="btn btn-primary px-10 py-4 text-lg rounded-full">
            Deploy Reconstruction
          </Link>
        </section>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-brand-darker text-gray-100 font-sans flex flex-col relative overflow-hidden">
      {/* Global ambient glow */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-1/4 w-[60vw] h-[60vh] bg-brand-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-1/4 w-[60vw] h-[60vh] bg-brand-secondary/5 rounded-full blur-[120px]" />
      </div>

      {/* Navigation Bar */}
      <nav className="fixed w-full top-0 z-50 bg-brand-darker/80 backdrop-blur-lg border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-20 items-center">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-primary to-brand-secondary p-[1px] group-hover:shadow-glow transition-all duration-300">
                <div className="w-full h-full rounded-[11px] bg-brand-darker flex items-center justify-center">
                  <span className="font-display font-bold text-lg bg-clip-text text-transparent bg-gradient-to-br from-brand-primary to-brand-accent">3D</span>
                </div>
              </div>
              <div className="flex flex-col">
                <span className="font-display font-bold text-xl tracking-wide text-white">RECONSTRUCT<span className="text-brand-primary">AI</span></span>
                <span className="text-[10px] uppercase tracking-[0.2em] text-gray-500 font-medium group-hover:text-brand-accent transition-colors">Pro Edition</span>
              </div>
            </Link>

            {/* Desktop Nav */}
            <div className="hidden sm:flex items-center gap-8">
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`text-sm font-medium transition-colors duration-200 relative py-2 ${location.pathname.startsWith(link.path) || (link.path === '/upload' && location.pathname === '/')
                      ? 'text-white'
                      : 'text-gray-400 hover:text-white'
                    }`}
                >
                  {link.name}
                  {(location.pathname.startsWith(link.path) || (link.path === '/upload' && location.pathname === '/')) && (
                    <motion.div
                      layoutId="active-nav-glow"
                      className="absolute bottom-0 left-0 right-0 h-[1px] bg-brand-primary shadow-[0_0_10px_rgba(59,130,246,1)]"
                    />
                  )}
                </Link>
              ))}

              <div className="h-6 w-[1px] bg-white/10 mx-2"></div>

              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/5">
                <div className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.5)]"></div>
                <span className="text-xs font-mono text-gray-300">SYSTEM_ONLINE</span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 pt-32 relative z-10 w-full">
        <AnimatePresence>
          {toast && (
            <motion.div
              initial={{ opacity: 0, y: -20, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.9 }}
              className={`fixed top-24 right-6 z-[60] p-4 rounded-xl border backdrop-blur-xl shadow-2xl flex items-center gap-4 ${toast.type === 'error'
                  ? 'bg-red-500/10 border-red-500/20 text-red-200'
                  : 'bg-green-500/10 border-green-500/20 text-green-200'
                }`}
            >
              <div className={`p-2 rounded-lg ${toast.type === 'error' ? 'bg-red-500/20' : 'bg-green-500/20'}`}>
                {toast.type === 'error' ? (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" d="M5 13l4 4L19 7" /></svg>
                )}
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-bold">{toast.type === 'error' ? 'Error' : 'Success'}</span>
                <span className="text-xs opacity-80">{toast.message}</span>
              </div>
              <button onClick={() => setToast(null)} className="ml-2 hover:text-white transition">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, filter: 'blur(10px)' }}
            animate={{ opacity: 1, filter: 'blur(0px)' }}
            exit={{ opacity: 0, filter: 'blur(10px)' }}
            transition={{ duration: 0.3 }}
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
      <footer className="relative z-10 border-t border-white/5 bg-brand-darker/50 backdrop-blur-md mt-auto">
        <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-sm text-gray-500">
            © 2026 ReconstructAI. Research Preview.
          </div>
          <div className="flex gap-6 text-sm text-gray-500">
            <a href="#" className="hover:text-brand-primary transition">Documentation</a>
            <a href="#" className="hover:text-brand-primary transition">API Reference</a>
            <a href="#" className="hover:text-brand-primary transition">Privacy</a>
          </div>
        </div>
      </footer>
    </div>
  )
}
