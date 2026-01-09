import axios from 'axios'

let baseURL = 'http://127.0.0.1:8000'
if (typeof window !== 'undefined') {
  const stored = window.localStorage.getItem('API_BASE_URL')
  if (stored) baseURL = stored
}

export const api = axios.create({ baseURL, timeout: 300000 })

export async function autoDetectBase() {
  if (typeof window === 'undefined') return baseURL
  const candidates = []
  const stored = window.localStorage.getItem('API_BASE_URL')
  if (stored) candidates.push(stored)
  const host = window.location.hostname || '127.0.0.1'
  candidates.push(`http://${host}:8000`)
  candidates.push('http://127.0.0.1:8000')
  candidates.push('http://localhost:8000')
  candidates.push('http://127.0.0.1:5000')
  candidates.push('http://localhost:5000')
  for (const c of candidates) {
    try {
      const resp = await fetch(`${c}/api/status`)
      if (resp.ok) {
        baseURL = c
        api.defaults.baseURL = c
        return c
      }
    } catch {}
  }
  return baseURL
}

export function setApiBase(url) {
  if (typeof window !== 'undefined') window.localStorage.setItem('API_BASE_URL', url)
  baseURL = url
  api.defaults.baseURL = url
}

export async function health() {
  try { const r = await api.get('/api/status'); return r.data } catch { return { status: 'offline' } }
}

export function getApiBase() {
  return api.defaults.baseURL || ''
}

export function toBackendURL(path) {
  if (!path) return path
  const base = getApiBase()
  if (!base) return path
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  return `${base}${path.startsWith('/') ? '' : '/'}${path}`
}
