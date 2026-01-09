import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

export default function Viewer3D({ plyUrl }) {
  const ref = useRef(null)
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!plyUrl || !el) return
    const W = 600, H = 400
    {
      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setSize(W, H)
      el.innerHTML = ''
      el.appendChild(renderer.domElement)
      const scene = new THREE.Scene()
      const camera = new THREE.PerspectiveCamera(60, W/H, 0.01, 1000)
      camera.position.set(0, 0, 2)
      const light = new THREE.DirectionalLight(0xffffff, 1)
      light.position.set(1,1,1)
      scene.add(light)
      setLoading(true)
      fetch(plyUrl).then(r=>r.text()).then(txt=>{
        const lines = txt.split('\n').slice(0)
        let start = lines.findIndex(l=>l.trim()==='end_header')
        const pts = []
        for (let i=start+1;i<lines.length;i++){ const parts = lines[i].trim().split(/\s+/); if(parts.length>=3){ const x=parseFloat(parts[0]), y=parseFloat(parts[1]), z=parseFloat(parts[2]); if(!Number.isNaN(x)) pts.push(new THREE.Vector3(x,y,z)) }}
        const geom = new THREE.BufferGeometry().setFromPoints(pts)
        const mat = new THREE.PointsMaterial({ size: 0.01, color: 0x00ffcc })
        const cloud = new THREE.Points(geom, mat)
        scene.add(cloud)
        renderer.setAnimationLoop(()=>{ renderer.render(scene, camera) })
        setLoading(false)
      })
    }
  }, [plyUrl])
  return (
    <div>
      {loading && <div style={{marginBottom:8}}>Loading point cloud...</div>}
      <div ref={ref} />
    </div>
  )
}