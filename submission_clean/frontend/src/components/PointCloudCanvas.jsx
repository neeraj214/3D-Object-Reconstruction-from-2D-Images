import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function PointCloudCanvas({ plyUrl, points, pointSize=0.01, autoRotate=false }) {
  const ref = useRef(null)
  const [size, setSize] = useState(pointSize)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const W = el.clientWidth || 600, H = 400
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(W, H)
    el.innerHTML = ''
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0e0e12)
    const camera = new THREE.PerspectiveCamera(60, W/H, 0.01, 1000)
    camera.position.set(0.2, 0.2, 0.8)
    const grid = new THREE.GridHelper(2, 20)
    scene.add(grid)
    const axes = new THREE.AxesHelper(0.5)
    scene.add(axes)
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.position.set(1,1,1)
    scene.add(light)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.autoRotate = !!autoRotate
    controls.autoRotateSpeed = 0.5
    controls.target.set(0,0,0)

    function buildPoints(arr) {
      const geom = new THREE.BufferGeometry()
      const pos = new Float32Array(arr.length * 3)
      for (let i=0;i<arr.length;i++) {
        const x = arr[i][0], y = arr[i][1], z = arr[i][2]
        pos[3*i+0] = x
        pos[3*i+1] = y
        pos[3*i+2] = z
      }
      geom.setAttribute('position', new THREE.BufferAttribute(pos, 3))
      const mat = new THREE.PointsMaterial({ size: size, color: 0x00ffcc, sizeAttenuation: true })
      const cloud = new THREE.Points(geom, mat)
      return cloud
    }

    function addFromPLY(text) {
      const lines = text.split('\n')
      let start = lines.findIndex(l=>l.trim()==='end_header')
      const pts = []
      for (let i=start+1;i<lines.length;i++) {
        const parts = lines[i].trim().split(/\s+/)
        if (parts.length>=3) {
          const x=parseFloat(parts[0]), y=parseFloat(parts[1]), z=parseFloat(parts[2])
          if (!Number.isNaN(x)) pts.push([x,y,z])
        }
      }
      const cloud = buildPoints(pts)
      scene.add(cloud)
    }

    if (points && points.length) {
      scene.add(buildPoints(points))
    } else if (plyUrl) {
      fetch(plyUrl).then(r=>r.text()).then(addFromPLY)
    }

    const animate = () => { controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate) }
    animate()
    return () => { renderer.dispose() }
  }, [plyUrl, points, size, autoRotate])

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <label>Point size</label>
        <input type="range" min="0.002" max="0.05" step="0.001" value={size} onChange={e=>setSize(parseFloat(e.target.value))} />
      </div>
      <div ref={ref} style={{ width: '100%', height: 420 }} />
    </div>
  )
}