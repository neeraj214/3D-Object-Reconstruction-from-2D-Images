import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function ImageUpload({ onSubmit }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const handleFile = (f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreview(url)
    if (onSubmit) onSubmit(f)
  }

  const onChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const onDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const onDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const clearFile = (e) => {
    e.stopPropagation()
    setFile(null)
    setPreview(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    if (onSubmit) onSubmit(null)
  }

  // Cleanup object URL
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  return (
    <div className="w-full">
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`
          relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 group
          ${isDragging
            ? 'border-brand-primary bg-brand-primary/10 shadow-glow'
            : 'border-white/10 hover:border-brand-primary/50 hover:bg-brand-primary/5 hover:shadow-glow'
          }
          ${preview ? 'border-solid border-white/20 bg-surface-soft/20' : ''}
        `}
      >
        <input
          type="file"
          accept="image/*"
          onChange={onChange}
          ref={fileInputRef}
          className="hidden"
        />

        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="relative inline-block"
            >
              <img
                src={preview}
                alt="Preview"
                className="max-h-72 mx-auto rounded-xl shadow-2xl border border-white/10 object-contain"
              />
              <button
                onClick={clearFile}
                className="absolute -top-3 -right-3 bg-brand-darker text-red-500 rounded-full p-2 shadow-lg border border-white/10 hover:bg-red-500/10 hover:border-red-500/50 transition-colors"
                title="Remove image"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
              <div className="mt-4 text-sm text-gray-300 font-medium truncate max-w-xs mx-auto">
                {file?.name}
              </div>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              <motion.div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center transition-colors ${isDragging ? 'bg-brand-primary/20 text-brand-primary' : 'bg-white/5 text-gray-500 group-hover:text-brand-primary group-hover:bg-brand-primary/20'}`} whileHover={{ scale: 1.05 }}>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-9 w-9" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </motion.div>
              <div className="text-gray-400">
                <span className="font-medium text-brand-primary hover:text-brand-accent transition-colors">Click to upload</span>
                <span className="text-gray-500"> or drag and drop</span>
              </div>
              <p className="text-xs text-gray-600">PNG, JPG, JPEG up to 10MB</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
