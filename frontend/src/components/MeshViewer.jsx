import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'

export default function MeshViewer({ objUrl, mtlUrl, autoRotate: initialAutoRotate=true }) {
  const containerRef = useRef(null)
  const rendererRef = useRef(null)
  const sceneRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const meshRef = useRef(null)
  const animationRef = useRef(null)

  const [wire, setWire] = useState(false)
  const [showTex, setShowTex] = useState(true)
  const [isAutoRotate, setIsAutoRotate] = useState(initialAutoRotate)
  const [loading, setLoading] = useState(false)

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
    camera.position.set(0.2, 0.2, 1.2)
    cameraRef.current = camera

    // Setup Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
    dirLight.position.set(1, 2, 3)
    scene.add(dirLight)
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.5)
    scene.add(hemiLight)

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

  // Load Mesh
  useEffect(() => {
    if (!objUrl || !sceneRef.current) return

    const loadMesh = () => {
      setLoading(true)
      
      // Clean up previous mesh
      if (meshRef.current) {
        sceneRef.current.remove(meshRef.current)
        meshRef.current = null
      }

      const loader = new OBJLoader()
      
      const onLoaded = (obj) => {
        // Center and scale if needed
        const box = new THREE.Box3().setFromObject(obj)
        const center = box.getCenter(new THREE.Vector3())
        const size = box.getSize(new THREE.Vector3())
        const maxDim = Math.max(size.x, size.y, size.z)
        const scale = 1.0 / maxDim
        
        obj.position.sub(center) // Center at origin
        obj.scale.multiplyScalar(scale) // Scale to fit unit box
        
        meshRef.current = obj
        sceneRef.current.add(obj)
        setLoading(false)
      }

      if (mtlUrl) {
        const mtl = new MTLLoader()
        mtl.load(mtlUrl, (materials) => {
          materials.preload()
          loader.setMaterials(materials)
          loader.load(objUrl, onLoaded)
        })
      } else {
        loader.load(objUrl, onLoaded)
      }
    }

    loadMesh()
  }, [objUrl, mtlUrl])

  // Update Wireframe/Texture
  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.traverse((c) => {
        if (c.isMesh) {
          c.material.wireframe = wire
          if (!showTex) {
             // Save map if not already saved
             if (c.material.map && !c.userData.map) c.userData.map = c.material.map
             c.material.map = null
          } else {
             // Restore map if exists
             if (c.userData.map) c.material.map = c.userData.map
          }
          c.material.needsUpdate = true
        }
      })
    }
  }, [wire, showTex, meshRef.current])

  // Update AutoRotate
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isAutoRotate
    }
  }, [isAutoRotate])

  const handleResetView = () => {
    if (controlsRef.current && cameraRef.current) {
      controlsRef.current.reset()
      cameraRef.current.position.set(0.2, 0.2, 1.2)
    }
  }

  return (
    <div className="relative w-full h-[500px] bg-gray-900 rounded-xl overflow-hidden shadow-lg border border-gray-800 group">
      {/* 3D Canvas */}
      <div ref={containerRef} className="w-full h-full" />
      
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 backdrop-blur-sm">
          <div className="text-white font-medium animate-pulse">Loading Model...</div>
        </div>
      )}

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
      </div>

      {/* Toggles */}
      <div className="absolute bottom-4 left-4 right-4 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <div className="bg-gray-800/80 backdrop-blur-sm px-4 py-2 rounded-full flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-300 hover:text-white transition-colors">
            <input 
              type="checkbox" 
              checked={wire} 
              onChange={e => setWire(e.target.checked)} 
              className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
            />
            Wireframe
          </label>
          <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-300 hover:text-white transition-colors">
            <input 
              type="checkbox" 
              checked={showTex} 
              onChange={e => setShowTex(e.target.checked)} 
              className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
            />
            Texture
          </label>
        </div>
      </div>
    </div>
  )
}