// Thin client for the FastAPI backend (same origin; Vite proxies /api in dev).
async function req(path, opts) {
  const r = await fetch(path, opts)
  const body = await r.json().catch(() => null)
  if (!r.ok) {
    const d = body?.detail
    const msg = typeof d === 'string' ? d : d?.message || (Array.isArray(d) ? d.map(x => x.msg).join('; ') : null)
    throw new Error(msg || `Request failed (HTTP ${r.status})`)
  }
  return body
}

export const api = {
  health: () => req('/api/health'),
  circuits: () => req('/api/circuits'),
  pool: () => req('/api/driver-pool'),
  constructors: season => req(`/api/constructors?season=${season}`),
  defaultGrid: circuit => req(`/api/default-grid?circuit_id=${encodeURIComponent(circuit)}`),
  models: () => req('/api/algorithms'),
  conditions: () => req('/api/conditions'),
  predict: body => req('/api/race-prediction', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
}
