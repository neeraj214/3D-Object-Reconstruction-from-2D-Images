import { describe, it, expect } from 'vitest'
import { predict, reconstruct, train, getDatasetsList, getDatasetCategory, getLatestMetrics } from '../src/services/reconstructionService'

describe('reconstructionService exports', () => {
  it('functions exist', () => {
    expect(typeof predict).toBe('function')
    expect(typeof reconstruct).toBe('function')
    expect(typeof train).toBe('function')
    expect(typeof getDatasetsList).toBe('function')
    expect(typeof getDatasetCategory).toBe('function')
    expect(typeof getLatestMetrics).toBe('function')
  })
})