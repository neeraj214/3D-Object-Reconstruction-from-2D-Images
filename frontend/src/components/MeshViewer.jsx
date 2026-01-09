import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'

export default function MeshViewer({ objUrl, mtlUrl, autoRotate=true }) {
  const ref = useRef(null)
  const [wire, setWire] = useState(false)
  const [showTex, setShowTex] = useState(true)
  useEffect(() => {
    const el = ref.current
    if (!el || !objUrl) return
    const W = el.clientWidth || 640, H = 420
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(W, H)
    el.innerHTML=''
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x101014)
    const camera = new THREE.PerspectiveCamera(60, W/H, 0.01, 1000)
    camera.position.set(0.2, 0.2, 1.2)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.autoRotate = !!autoRotate
    const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1)
    scene.add(light)
    const grid = new THREE.GridHelper(2, 20)
    scene.add(grid)
    const axes = new THREE.AxesHelper(0.5)
    scene.add(axes)
    const loader = new OBJLoader()
    if (mtlUrl) {
      const mtl = new MTLLoader()
      mtl.load(mtlUrl, materials => {
        materials.preload()
        loader.setMaterials(materials)
        loader.load(objUrl, obj => {
          obj.traverse(c => { if (c.isMesh) { c.material.wireframe = wire; c.material.needsUpdate = true; if (!showTex) { c.material.map = null } } })
          scene.add(obj)
        })
      })
    } else {
      loader.load(objUrl, obj => {
        obj.traverse(c => { if (c.isMesh) { c.material.wireframe = wire; c.material.needsUpdate = true; } })
        scene.add(obj)
      })
    }
    const animate = () => { controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate) }
    animate()
    return () => { renderer.dispose() }
  }, [objUrl, mtlUrl, wire, showTex, autoRotate])
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <label className="flex items-center gap-1"><input type="checkbox" checked={wire} onChange={e=>setWire(e.target.checked)} /> Wireframe</label>
        <label className="flex items-center gap-1"><input type="checkbox" checked={showTex} onChange={e=>setShowTex(e.target.checked)} /> Texture</label>
      </div>
      <div ref={ref} style={{ width: '100%', height: 420 }} />
    </div>
  )
}