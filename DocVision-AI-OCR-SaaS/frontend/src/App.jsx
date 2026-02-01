import React, { useState } from 'react'
import axios from 'axios'

export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const submit = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await axios.post('http://127.0.0.1:8000/api/v1/ocr', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResult(res.data)
    } catch (err) {
      setError('Failed to process image')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, Arial' }}>
      <h1>DocVision AI</h1>
      <p>Upload an image (PNG/JPEG) to extract text.</p>
      <form onSubmit={submit}>
        <input type="file" accept="image/png,image/jpeg" onChange={(e)=>setFile(e.target.files[0]||null)} />
        <button type="submit" disabled={loading || !file} style={{ marginLeft: 12 }}>Run OCR</button>
      </form>
      {loading && <p>Processing...</p>}
      {error && <p style={{color:'red'}}>{error}</p>}
      {result && (
        <div>
          <h2>Text</h2>
          <pre style={{whiteSpace:'pre-wrap'}}>{result.text}</pre>
          <h3>Blocks: {result.blocks?.length ?? 0}</h3>
        </div>
      )}
    </div>
  )
}
