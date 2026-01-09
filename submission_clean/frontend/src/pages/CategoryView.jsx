import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { getDatasetCategory } from '../services/reconstructionService'

export default function CategoryView() {
  const { dataset, category } = useParams()
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    (async () => {
      setLoading(true)
      try { const r = await getDatasetCategory(category, dataset, 20); setItems(r.items || []) } catch {}
      setLoading(false)
    })()
  }, [dataset, category])
  return (
    <div>
      <h3 className="text-lg font-semibold mb-2">{dataset} / {category}</h3>
      {loading ? <div>Loading...</div> : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {items.map((u,i)=> (<img key={i} src={u} alt="item" className="rounded" />))}
        </div>
      )}
    </div>
  )
}