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
      <div className="w-full relative z-10 space-y-24 pb-20">

        {/* HERO SECTION */}
        <section className="relative grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="space-y-8 z-10"
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-surface-soft/60 backdrop-blur-md border border-white/10 text-sm text-brand-accent font-medium shadow-[0_0_15px_rgba(6,182,212,0.3)]">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-accent opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-brand-accent"></span>
              </span>
              <span>Next-Gen 3D Reconstruction Engine</span>
            </div>

            <h1 className="text-6xl sm:text-7xl font-display font-extrabold tracking-tight text-white leading-[1.1]">
              Reality <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-primary via-brand-secondary to-brand-accent">Transformed</span>
            </h1>

            <p className="text-xl text-gray-300 max-w-lg leading-relaxed border-l-4 border-brand-primary/50 pl-6">
              Turn any 2D image into a high-fidelity 3D model instantly. Powered by advanced hybrid neural networks for chaotic environments.
            </p>

            <div className="flex flex-wrap gap-5 pt-4">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Link to="/upload" className="btn btn-primary text-lg px-10 py-4 rounded-2xl shadow-glow relative overflow-hidden group">
                  <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
                  <span className="relative flex items-center gap-2">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" /></svg>
                    Launch Studio
                  </span>
                </Link>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <a href="https://github.com/neeraj214" target="_blank" rel="noreferrer" className="btn btn-secondary text-lg px-8 py-4 rounded-2xl">
                  GitHub
                </a>
              </motion.div>
            </div>

            <div className="flex items-center gap-8 pt-8 text-sm font-mono text-gray-500">
              <div className="flex -space-x-3">
                {[1, 2, 3, 4].map(i => (
                  <div key={i} className="w-10 h-10 rounded-full border-2 border-brand-darker bg-surface-soft flex items-center justify-center text-xs text-white">U{i}</div>
                ))}
              </div>
              <p>Trusted by <span className="text-white font-bold">500+</span> Researchers</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="relative"
          >
            <div className="absolute -inset-1 bg-gradient-to-r from-brand-primary to-brand-accent rounded-3xl blur-2xl opacity-40 animate-pulse"></div>
            <div className="relative rounded-3xl overflow-hidden border border-white/10 shadow-2xl bg-black/50 backdrop-blur-sm">
              <img src="/src/assets/hero_image.png" alt="2D to 3D Visualization" className="w-full h-auto object-cover transform hover:scale-105 transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-t from-brand-darker via-transparent to-transparent opacity-60"></div>

              {/* Floating UI Elements on Image */}
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1, duration: 0.5 }}
                className="absolute bottom-6 left-6 right-6 bg-surface-glass backdrop-blur-xl p-4 rounded-xl border border-white/20 flex gap-4 items-center"
              >
                <div className="w-12 h-12 bg-brand-primary/20 rounded-lg flex items-center justify-center border border-brand-primary/50 text-brand-primary">
                  <svg className="w-6 h-6 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="2" d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" /></svg>
                </div>
                <div>
                  <div className="text-xs text-brand-accent uppercase tracking-wider font-bold">Status</div>
                  <div className="text-white font-mono text-sm">Rendering Point Cloud...</div>
                </div>
                <div className="ml-auto text-2xl font-bold text-white">98%</div>
              </motion.div>
            </div>
          </motion.div>
        </section>

        {/* HOW IT WORKS SECTION */}
        <section className="relative py-12">
          <div className="text-center mb-16">
            <span className="text-brand-accent font-mono text-sm tracking-widest uppercase">Workflow</span>
            <h2 className="text-4xl font-display font-bold text-white mt-2">From Pixel to Voxel</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Connecting Line */}
            <div className="hidden md:block absolute top-12 left-[16%] right-[16%] h-0.5 bg-gradient-to-r from-brand-primary/0 via-brand-primary/50 to-brand-primary/0 border-t border-dashed border-white/20 z-0"></div>

            {[
              { step: '01', title: 'Upload', desc: 'Drag & drop any 2D image. Standard formats (JPG, PNG) supported.', icon: 'M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12' },
              { step: '02', title: 'Process', desc: 'Our AI analyzes geometry, depth, and texture in seconds.', icon: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z' },
              { step: '03', title: 'Interact', desc: 'Rotate, zoom, and export your 3D model (PLY/OBJ).', icon: 'M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z' }
            ].map((s, i) => (
              <div key={i} className="relative z-10 flex flex-col items-center text-center group">
                <div className="w-24 h-24 rounded-2xl bg-surface-glass backdrop-blur-xl border border-white/10 flex items-center justify-center mb-6 shadow-xl group-hover:-translate-y-2 transition-transform duration-300 relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-brand-primary/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  <span className="absolute top-2 right-3 text-xs font-bold text-white/20 font-display">{s.step}</span>
                  <svg className="w-10 h-10 text-white group-hover:text-brand-accent transition-colors duration-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d={s.icon} /></svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{s.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed max-w-xs">{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* FEATURES GRID */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { title: 'Neural Rendering', desc: 'Advanced NeRF-inspired algorithms for photorealistic texture recovery.', color: 'from-blue-500/20 to-cyan-500/20' },
            { title: 'Geometric Consistency', desc: 'Enforces shape priors to prevent artifacts in occluded regions.', color: 'from-purple-500/20 to-pink-500/20' },
            { title: 'Real-time Preview', desc: 'WebGL-powered viewer with PBR lighting and post-processing.', color: 'from-amber-500/20 to-orange-500/20' },
            { title: 'Cloud Export', desc: 'Download standard formats compatible with Blender, Unity, and Unreal.', color: 'from-emerald-500/20 to-teal-500/20' },
          ].map((f, i) => (
            <motion.div
              key={i}
              whileHover={{ scale: 1.02 }}
              className={`p-8 rounded-3xl bg-gradient-to-br ${f.color} border border-white/5 backdrop-blur-sm relative overflow-hidden group`}
            >
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <svg className="w-24 h-24 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" /></svg>
              </div>
              <h3 className="text-2xl font-display font-bold text-white mb-3">{f.title}</h3>
              <p className="text-gray-300 leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </section>

        {/* CALL TO ACTION */}
        <section className="relative rounded-[2.5rem] overflow-hidden bg-brand-primary/10 border border-brand-primary/20 p-12 sm:p-20 text-center">
          <div className="absolute inset-0 bg-[url('/src/assets/hero_image.png')] bg-cover bg-center opacity-10 mix-blend-overlay"></div>
          <div className="relative z-10">
            <h2 className="text-4xl sm:text-5xl font-display font-bold text-white mb-6">Ready to Create?</h2>
            <p className="text-xl text-gray-300 mb-10 max-w-2xl mx-auto">Join the revolution in 3D asset generation today.</p>
            <Link to="/upload" className="inline-flex items-center btn btn-primary px-12 py-5 text-xl rounded-full shadow-[0_0_30px_rgba(59,130,246,0.4)] hover:shadow-[0_0_50px_rgba(59,130,246,0.6)] transition-all">
              Try It Now - Free
            </Link>
          </div>
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
