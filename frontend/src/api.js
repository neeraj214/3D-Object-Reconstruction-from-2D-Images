export { setApiBase, health } from './services/apiClient'
export { predict, reconstruct as generatePointcloud, train } from './services/reconstructionService'
export { getDatasetsList, getDatasetCategory, getLatestMetrics } from './services/reconstructionService'
// Backward-compatible names used in existing code
export { getDatasetsList as getCategories } from './services/reconstructionService'
export { getLatestMetrics as getMetrics } from './services/reconstructionService'
export { getDatasetCategory as getDatasetImages } from './services/reconstructionService'