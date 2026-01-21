import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getDatasetCategory } from '../services/reconstructionService'
import { motion } from 'framer-motion'

export default function CategoryView() {
  const { dataset, category } = useParams()
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    (async () => {
      setLoading(true)
      try { 
        const r = await getDatasetCategory(category, dataset, 20)
        setItems(r.items || []) 
      } catch {}
      setLoading(false)
    })()
  }, [dataset, category])

  return (
    <div className="space-y-6">
      {/* Header & Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
        <Link to="/datasets" className="hover:text-blue-600 transition-colors">Datasets</Link>
        <span>/</span>
        <span className="font-medium text-gray-900">{dataset}</span>
        <span>/</span>
        <span className="font-medium text-gray-900">{category}</span>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 min-h-[500px]">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">{category} Samples</h2>
        
        {loading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 animate-pulse">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="aspect-square bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
            {items.length > 0 ? items.map((u, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.05 }}
                className="aspect-square rounded-lg overflow-hidden bg-gray-100 relative group cursor-pointer"
              >
                <img src={u} alt="item" className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110" />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                {/* Overlay actions could go here */}
              </motion.div>
            )) : (
              <div className="col-span-full flex flex-col items-center justify-center py-12 text-gray-500">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-3 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p>No images found for this category.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}