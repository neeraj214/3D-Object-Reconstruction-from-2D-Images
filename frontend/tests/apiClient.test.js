import { describe, it, expect } from 'vitest'
import { api, setApiBase, health } from '../src/services/apiClient'

describe('apiClient', () => {
  it('has axios instance', () => {
    expect(api).toBeTruthy()
  })
  it('can set API base', async () => {
    setApiBase('http://127.0.0.1:5000')
    expect(typeof setApiBase).toBe('function')
  })
  it('health returns object', async () => {
    const h = await health()
    expect(typeof h).toBe('object')
  })
})