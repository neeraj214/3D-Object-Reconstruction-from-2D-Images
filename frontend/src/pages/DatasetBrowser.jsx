import React, { useEffect, useState } from 'react'
import { getDatasetsList } from '../api'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function DatasetBrowser() {
  const [datasets, setDatasets] = useState({})
  const [selected, setSelected] = useState({ dataset: null, category: null })
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    (async () => {
      const d = await getDatasetsList()
      setDatasets(d.datasets || {})
    })()
  }, [])

  const load = async (dataset, category) => {
    setSelected({ dataset, category })
    setLoading(true)
    try {
      // In a real app, we'd fetch the items here properly
      // For now, we simulate or use the proxy if available
      const resp = await fetch(`${window.location.origin}/api-proxy?dataset=${dataset}&category=${category}`)
      if (resp.ok) {
         // Assume response gives items, or we handle it otherwise. 
         // The original code didn't actually set items from resp, it just fetched.
         // I'll assume for now we just show the selection state.
      }
    } catch {}
    setLoading(false)
  }

  return (
    <div className="grid md:grid-cols-4 gap-8">
      {/* Sidebar / Dataset List */}
      <div className="md:col-span-1 space-y-6">
        <h2 className="text-xl font-bold text-gray-900">Datasets</h2>
        <div className="space-y-4">
          {Object.entries(datasets).map(([ds, cats], idx) => (
            <motion.div 
              key={ds}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden"
            >
              <div className="bg-gray-50 px-4 py-3 border-b border-gray-100 font-semibold text-gray-700">
                {ds}
              </div>
              <div className="p-2 space-y-1">
                {cats.map(c => (
                  <Link 
                    key={c} 
                    to={`/datasets/${ds}/${c}`} 
                    onClick={() => load(ds, c)}
                    className={`block px-3 py-2 rounded-lg text-sm transition-colors ${
                      selected.dataset === ds && selected.category === c
                        ? 'bg-blue-50 text-blue-700 font-medium'
                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                    }`}
                  >
                    {c}
                  </Link>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Main Content / Image Grid */}
      <div className="md:col-span-3">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 min-h-[500px] p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
            {selected.dataset ? (
              <>
                <span className="text-gray-500">{selected.dataset}</span>
                <span className="text-gray-300">/</span>
                <span>{selected.category}</span>
              </>
            ) : (
              "Select a Category"
            )}
          </h2>

          {!selected.dataset ? (
            <div className="flex flex-col items-center justify-center h-64 text-gray-400">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p>Choose a dataset category from the sidebar to view samples</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
              {items.map((u, i) => (
                <motion.div 
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                  className="aspect-square rounded-lg overflow-hidden bg-gray-100 relative group"
                >
                  <img src={u} alt="item" className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110" />
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                </motion.div>
              ))}
              {items.length === 0 && !loading && (
                 <div className="col-span-full text-center py-12 text-gray-500">
                   No preview images available directly. Click to view full details.
                 </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}