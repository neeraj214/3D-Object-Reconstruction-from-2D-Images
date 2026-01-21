import React, { useEffect, useRef, useState, useMemo } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function PointCloudCanvas({ plyUrl, points, pointSize=0.01, autoRotate: initialAutoRotate=false }) {
  const containerRef = useRef(null)
  const rendererRef = useRef(null)
  const sceneRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const pointsRef = useRef(null)
  const animationRef = useRef(null)

  const [size, setSize] = useState(pointSize)
  const [isAutoRotate, setIsAutoRotate] = useState(initialAutoRotate)
  
  // Initialize Three.js scene
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Setup Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    rendererRef.current = renderer
    container.appendChild(renderer.domElement)

    // Setup Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x111827) // Tailwind gray-900
    sceneRef.current = scene

    // Setup Camera
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 1000)
    camera.position.set(0.5, 0.5, 1.0)
    cameraRef.current = camera

    // Setup Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
    dirLight.position.set(1, 2, 3)
    scene.add(dirLight)

    // Helpers
    const grid = new THREE.GridHelper(2, 20, 0x374151, 0x1f2937)
    scene.add(grid)
    
    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.autoRotateSpeed = 2.0
    controlsRef.current = controls

    // Resize Observer
    const resizeObserver = new ResizeObserver(() => {
      if (!container || !camera || !renderer) return
      const width = container.clientWidth
      const height = container.clientHeight
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    })
    resizeObserver.observe(container)

    // Animation Loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate)
      if (controlsRef.current) controlsRef.current.update()
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current)
      }
    }
    animate()

    return () => {
      cancelAnimationFrame(animationRef.current)
      resizeObserver.disconnect()
      if (container && renderer.domElement) container.removeChild(renderer.domElement)
      renderer.dispose()
    }
  }, [])

  // Update Points
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    // Remove old points
    if (pointsRef.current) {
      scene.remove(pointsRef.current)
      pointsRef.current.geometry.dispose()
      pointsRef.current.material.dispose()
      pointsRef.current = null
    }

    let positions = null

    if (points && points.length > 0) {
      // Points passed as array
      const flatPoints = points.flat() // If points is [[x,y,z],...] -> [x,y,z,...]
      // Check if it's already flat or nested
      if (points[0] && Array.isArray(points[0])) {
         positions = new Float32Array(points.length * 3)
         for(let i=0; i<points.length; i++) {
            positions[i*3] = points[i][0]
            positions[i*3+1] = points[i][1]
            positions[i*3+2] = points[i][2]
         }
      } else {
         positions = new Float32Array(points)
      }
    } else if (plyUrl) {
      // TODO: Handle PLY fetching if needed, but for now we focus on points array
    }

    if (positions) {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      
      // Compute bounding box to center the object
      geometry.computeBoundingBox()
      const center = geometry.boundingBox.getCenter(new THREE.Vector3())
      geometry.translate(-center.x, -center.y, -center.z) // Center at origin

      const material = new THREE.PointsMaterial({ 
        size: size, 
        color: 0x38bdf8, // Tailwind sky-400
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9
      })
      
      const pointCloud = new THREE.Points(geometry, material)
      scene.add(pointCloud)
      pointsRef.current = pointCloud
    }

  }, [points, plyUrl]) // Re-run if data changes

  // Update Size
  useEffect(() => {
    if (pointsRef.current) {
      pointsRef.current.material.size = size
    }
  }, [size])

  // Update AutoRotate
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isAutoRotate
    }
  }, [isAutoRotate])

  // Download Handler
  const handleDownload = () => {
    if (!points || points.length === 0) return
    
    let content = "ply\nformat ascii 1.0\nelement vertex " + points.length + "\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    points.forEach(p => {
      content += `${p[0]} ${p[1]} ${p[2]}\n`
    })
    
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'reconstruction.ply'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleResetView = () => {
    if (controlsRef.current && cameraRef.current) {
      controlsRef.current.reset()
      cameraRef.current.position.set(0.5, 0.5, 1.0)
    }
  }

  return (
    <div className="relative w-full h-[500px] bg-gray-900 rounded-xl overflow-hidden shadow-lg border border-gray-800 group">
      {/* 3D Canvas */}
      <div ref={containerRef} className="w-full h-full" />

      {/* Controls Overlay */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <button 
          onClick={handleResetView}
          className="bg-gray-800/80 text-white p-2 rounded-lg hover:bg-gray-700 backdrop-blur-sm transition-colors"
          title="Reset View"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 12"/></svg>
        </button>
        <button 
          onClick={() => setIsAutoRotate(!isAutoRotate)}
          className={`p-2 rounded-lg backdrop-blur-sm transition-colors ${isAutoRotate ? 'bg-blue-600 text-white' : 'bg-gray-800/80 text-white hover:bg-gray-700'}`}
          title="Auto Rotate"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 12"/><path d="M21 3v9h-9"/></svg>
        </button>
        <button 
          onClick={handleDownload}
          className="bg-gray-800/80 text-white p-2 rounded-lg hover:bg-gray-700 backdrop-blur-sm transition-colors"
          title="Download PLY"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
        </button>
      </div>

      {/* Point Size Control */}
      <div className="absolute bottom-4 left-4 right-4 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <div className="bg-gray-800/80 backdrop-blur-sm px-4 py-2 rounded-full flex items-center gap-3">
          <span className="text-xs text-gray-300 font-medium">Point Size</span>
          <input 
            type="range" 
            min="0.002" 
            max="0.05" 
            step="0.001" 
            value={size} 
            onChange={e => setSize(parseFloat(e.target.value))}
            className="w-32 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>
    </div>
  )
}