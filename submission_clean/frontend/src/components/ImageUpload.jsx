import React, { useState, useRef } from 'react'

export default function ImageUpload({ onSubmit }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const dropRef = useRef(null)
  const onFile = f => { setFile(f); setPreview(URL.createObjectURL(f)) }
  const onChange = e => { if (e.target.files && e.target.files[0]) onFile(e.target.files[0]) }
  const onDrop = e => { e.preventDefault(); if (e.dataTransfer.files && e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]) }
  const onDrag = e => e.preventDefault()
  const submit = () => { if (file) onSubmit(file) }
  return (
    <div>
      <div ref={dropRef} onDrop={onDrop} onDragOver={onDrag} className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition">
        <input type="file" accept="image/*" onChange={onChange} className="mx-auto" />
        {preview && <img src={preview} alt="preview" className="mt-4 max-w-md mx-auto rounded" />}
      </div>
      <button onClick={submit} className="btn mt-3">Upload</button>
    </div>
  )
}