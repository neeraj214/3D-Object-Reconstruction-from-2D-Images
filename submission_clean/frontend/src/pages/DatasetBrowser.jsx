import React, { useEffect, useState } from 'react'
import { getDatasetsList } from '../api'
import { Link } from 'react-router-dom'

export default function DatasetBrowser() {
  const [datasets, setDatasets] = useState({})
  const [selected, setSelected] = useState({ dataset: null, category: null })
  const [items, setItems] = useState([])
  useEffect(()=>{ (async()=>{ const d = await getDatasetsList(); setDatasets(d.datasets || {}) })() },[])
  const load = async (dataset, category) => {
    setSelected({ dataset, category })
    // keep previous behavior to show inline grid as well
    try {
      const resp = await fetch(`${window.location.origin}/api-proxy?dataset=${dataset}&category=${category}`)
      if (!resp.ok) throw new Error('')
    } catch {}
  }
  return (
    <div className="grid md:grid-cols-3 gap-4">
      <div className="card">
        <h3 className="font-semibold mb-2">Datasets</h3>
        {Object.entries(datasets).map(([ds, cats])=> (
          <div key={ds} className="mb-3">
            <div className="font-medium">{ds}</div>
            <div className="flex flex-wrap gap-2 mt-1">
              {cats.map(c=> (
                <Link key={c} className="btn" to={`/datasets/${ds}/${c}`} onClick={()=>load(ds, c)}>{c}</Link>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="card md:col-span-2">
        <h3 className="font-semibold mb-2">Images {selected.dataset && `(${selected.dataset} / ${selected.category})`}</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {items.map((u,i)=> (
            <img key={i} src={u} alt="item" className="rounded" />
          ))}
        </div>
      </div>
    </div>
  )
}