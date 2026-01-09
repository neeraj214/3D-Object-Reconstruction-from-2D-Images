import { api } from './apiClient'

export async function predict(file, options = {}) {
  const { fScale = 1.1, useSegmentation = true, nPoints = 16000, mode = 'fast' } = options
  const form = new FormData()
  form.append('file', file)
  const params = new URLSearchParams({ f_scale: String(fScale), use_segmentation: useSegmentation ? 'true' : 'false', n_points: String(nPoints), mode })
  const r = await api.post(`/api/reconstruct?${params.toString()}`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
  return r.data
}

export async function reconstruct(file, options = {}) {
  const { fScale = 1.1, useSegmentation = true, nPoints = 16000, mode = 'fast' } = options
  const form = new FormData()
  form.append('file', file)
  const params = new URLSearchParams({ f_scale: String(fScale), use_segmentation: useSegmentation ? 'true' : 'false', n_points: String(nPoints), mode })
  const r = await api.post(`/api/reconstruct?${params.toString()}`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
  return r.data
}

export async function train(config = {}) {
  const r = await api.post('/train', config)
  return r.data
}

export async function getDatasetsList() {
  const r = await api.get('/datasets/list')
  return r.data
}

export async function getDatasetCategory(name, dataset, maxItems = 12) {
  const params = new URLSearchParams({ dataset, max_items: String(maxItems) })
  const r = await api.get(`/datasets/category/${encodeURIComponent(name)}?${params.toString()}`)
  return r.data
}

export async function getDatasetImages(dataset, category, maxItems = 12) {
  const params = new URLSearchParams({ dataset, category, max_items: String(maxItems) })
  const r = await api.get(`/get_dataset_images?${params.toString()}`)
  return r.data
}

export async function getLatestMetrics() {
  const r = await api.get('/metrics/latest')
  return r.data
}

export async function uploadImage(file) {
  const form = new FormData()
  form.append('file', file)
  const r = await api.post('/api/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
  return r.data
}

export async function modelInfo() {
  const r = await api.get('/api/model-info')
  return r.data
}