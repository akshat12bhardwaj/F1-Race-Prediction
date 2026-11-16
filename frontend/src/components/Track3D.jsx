import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { circuitFrame, constrainCircuitCamera } from '../cameraBounds.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { teamColor } from '../teams.js'
import { BRIDGES, bridgeLift, buildTrack, carGap, frameAt, gridSide, LIGHTS_OUT, replayAt } from '../motion.js'
import { carAppearance } from '../carAssets.js'
import { CAR_LENGTH, CAR_WIDTH, createCar, releaseCar, preloadCars } from '../car.js'

const LIME = '#d2f03c'
const HALF = 7                       // F1 tracks are ~12-15 m wide
const MODES = [['overview', 'Overview'], ['follow', 'Follow'], ['onboard', 'Onboard'], ['top', 'Top']]
const CONTOUR_M = 5                  // terrain contour interval (true metres)

// Road ribbon on the 3D centreline (flat cross-section: no reliable banking data), optionally striped like a kerb.
function ribbon(T, h, lift, stripes) {
  const hw = typeof h === 'function' ? h : () => h
  const n = T.n, pos = [], col = [], idx = []
  const a = new THREE.Color(LIME), b = new THREE.Color('#e9e9e6')
  for (let i = 0; i <= n; i++) {
    const k = i % n, p = (k - 1 + n) % n, q = (k + 1) % n
    let tx = T.xs[q] - T.xs[p], tz = T.zs[q] - T.zs[p]
    const l = Math.hypot(tx, tz) || 1; tx /= l; tz /= l
    const y = T.ys[k] + lift
    const hk = hw(k)
    pos.push(T.xs[k] - tz * hk, y, T.zs[k] + tx * hk, T.xs[k] + tz * hk, y, T.zs[k] - tx * hk)
    if (stripes) { const c = Math.floor(T.cum[k] / 12) % 2 ? a : b; col.push(c.r, c.g, c.b, c.r, c.g, c.b) }
    if (i < n) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2)
  }
  // the trace may run clockwise or anticlockwise: flip triangles so their front face points up
  const ax = pos[3] - pos[0], az = pos[5] - pos[2], bx = pos[6] - pos[0], bz = pos[8] - pos[2]
  if (az * bx - ax * bz < 0) for (let t = 0; t < idx.length; t += 3) [idx[t + 1], idx[t + 2]] = [idx[t + 2], idx[t + 1]]
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  if (stripes) g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
  g.setIndex(idx)
  g.computeVertexNormals()
  return g
}

// DEM terrain (heights relative to the circuit's lowest point), upsampled and lowered under the road so the
// surface never pokes through the asphalt. Outside a band around the track it is the DEM itself.
function terrain(T, t, inner, outer, below) {
  const M = 150, sx = ((t.n - 1) * t.dx) / (M - 1), sz = ((t.n - 1) * t.dz) / (M - 1)
  const H = (gx, gz) => {                                    // bilinear on the DEM grid (grid units)
    const i = Math.min(t.n - 2, Math.floor(gz)), j = Math.min(t.n - 2, Math.floor(gx)), fz = gz - i, fx = gx - j
    const h = (r, c) => t.h[r * t.n + c]
    return (h(i, j) * (1 - fx) + h(i, j + 1) * fx) * (1 - fz) + (h(i + 1, j) * (1 - fx) + h(i + 1, j + 1) * fx) * fz
  }
  const pos = new Float32Array(M * M * 3), idx = []
  for (let r = 0; r < M; r++) for (let c = 0; c < M; c++) {
    const x = t.x0 + c * sx, z = t.z0 + r * sz
    let best = Infinity, yr = 0
    for (let k = 0; k < T.n; k++) { const d = (T.xs[k] - x) ** 2 + (T.zs[k] - z) ** 2; if (d < best) { best = d; yr = T.ground[k] } }
    const d = Math.sqrt(best), u = Math.min(1, Math.max(0, (d - inner) / (outer - inner))), s = u * u * (3 - 2 * u)
    const ht = H((c * sx) / t.dx, (r * sz) / t.dz)
    const o = 3 * (r * M + c)
    pos[o] = x; pos[o + 1] = (yr - below) * (1 - s) + ht * s; pos[o + 2] = z
  }
  for (let r = 0; r < M - 1; r++) for (let c = 0; c < M - 1; c++) {
    const a = r * M + c
    idx.push(a, a + M, a + 1, a + 1, a + M, a + M + 1)
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setIndex(idx); g.computeVertexNormals()
  return g
}

// Contour lines every CONTOUR_M true metres, drawn in the shader from world height.
function contourMaterial() {
  const m = new THREE.MeshStandardMaterial({ color: '#1b1d1f', roughness: 1 })
  m.onBeforeCompile = sh => {
    sh.uniforms.uStep = { value: CONTOUR_M }
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying float vH;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvH = position.y;')
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying float vH;\nuniform float uStep;')
      .replace('#include <dithering_fragment>', `#include <dithering_fragment>
        float q = vH / uStep; float l = 1.0 - min(abs(fract(q - 0.5) - 0.5) / max(fwidth(q), 1e-4), 1.0);
        gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.82, 0.94, 0.24), l * 0.22);`)
  }
  return m
}

// Embankment walls under a raised (bridge) section, left open where the lower road passes underneath.
function skirts(T, h, top, open) {
  const pos = [], n = T.n
  const edge = (k, side) => {
    const p = (k - 1 + n) % n, q = (k + 1) % n
    const tx = T.xs[q] - T.xs[p], tz = T.zs[q] - T.zs[p], l = Math.hypot(tx, tz) || 1
    return [T.xs[k] - side * (tz / l) * h, T.zs[k] + side * (tx / l) * h]
  }
  const clear = k => T.at.every(c => Math.hypot(T.xs[k] - c.x, T.zs[k] - c.z) > open)
  for (let k = 0; k < n; k++) {
    const j = (k + 1) % n
    if (T.lift[k] < 0.2 || T.lift[j] < 0.2 || !clear(k) || !clear(j)) continue
    for (const side of [1, -1]) {
      const [ax, az] = edge(k, side), [bx, bz] = edge(j, side)
      const a1 = T.ys[k] + top, b1 = T.ys[j] + top, a0 = T.ground[k] - 1, b0 = T.ground[j] - 1
      pos.push(ax, a1, az, ax, a0, az, bx, b1, bz, bx, b1, bz, ax, a0, az, bx, b0, bz)
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.computeVertexNormals()
  return g
}

// Round marker sprite (white disc, dark rim); tinted per car by vertex colour. ring = selection halo.
function disc(ring) {
  const c = document.createElement('canvas'); c.width = c.height = 64
  const x = c.getContext('2d')
  x.beginPath(); x.arc(32, 32, ring ? 26 : 24, 0, 7)
  if (ring) { x.lineWidth = 7; x.strokeStyle = '#fff'; x.stroke() }
  else { x.fillStyle = '#fff'; x.fill(); x.lineWidth = 7; x.strokeStyle = '#0b0c0d'; x.stroke() }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace
  return t
}

function checker() {
  const c = document.createElement('canvas'); c.width = 64; c.height = 16
  const x = c.getContext('2d')
  for (let i = 0; i < 16; i++) for (let j = 0; j < 4; j++) { x.fillStyle = (i + j) % 2 ? '#111' : '#f4f5f2'; x.fillRect(i * 4, j * 4, 4, 4) }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace
  return t
}
const dispose = g => g.traverse(o => { if (!o.userData.sharedCarGeometry) o.geometry?.dispose(); [o.material].flat().forEach(m => { m?.map?.dispose(); m?.dispose() }) })

export default function Track3D({ geometry, circuitId, conditions, paused, entries, result, names, selected, onSelect, onToggle, focus, progress, isPlaying }) {
  const wrap = useRef(null), tags = useRef({})
  const S = useRef(null)
  const [mode, setMode] = useState('overview')
  const [modeN, setModeN] = useState(0)                     // re-pressing a camera button re-engages it after manual control
  const pickMode = k => { setMode(k); setModeN(n => n + 1) }
  const [renderError, setRenderError] = useState(null)
  const [assets, setAssets] = useState({ loading: 0, failed: [] })
  const [markers, setMarkers] = useState(true)              // screen-size team markers in the wide views
  const [hover, setHover] = useState(null)
  const hasElev = !!geometry.y
  const B = hasElev ? BRIDGES[circuitId] : null
  const T = useMemo(() => {
    const R = Math.max(...geometry.x.map(Math.abs), ...geometry.z.map(Math.abs))
    const w = 1, cs = 1.15       // measured circuit footprint; modest visual car scale only
    const ground = hasElev ? geometry.y.slice() : new Array(geometry.x.length).fill(0)
    // DEM cannot resolve crossover structures; preserve the explicitly labelled bridge clearance.
    const br = B ? bridgeLift(geometry.x, geometry.z, ground, { over: B.over, clearance: B.clearance_m, ramp: B.ramp_m })
      : { lift: ground.map(() => 0), at: [] }
    return Object.assign(buildTrack(geometry.x, geometry.z, ground.map((v, k) => v + br.lift[k])), { R, w, cs, ground, lift: br.lift, at: br.at })
  }, [geometry, hasElev, B])
  const slots = useMemo(() => {
    const fin = result && Object.fromEntries(result.standings.map(r => [r.driver_id, r.position - 1]))
    return Object.fromEntries(entries.map((e, i) => [e.driver_id, [i, fin?.[e.driver_id] ?? i]]))
  }, [entries, result])
  const live = useRef({}); live.current = { slots, selected, hover, mode, markers, progress, isPlaying, onSelect, onToggle, setHover, names, racing: !!result }

  // ---- renderer, camera, loop: one per component lifetime ----
  useEffect(() => {
    let renderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' }) }
    catch { setRenderError('The 3D viewer needs WebGL. Enable hardware acceleration in your browser and reload.'); return }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    const el = wrap.current
    el.prepend(renderer.domElement)
    const scene = new THREE.Scene()
    scene.fog = new THREE.Fog('#121315', 1, 2)
    const camera = new THREE.PerspectiveCamera(40, 1, 1, 40000)
    const controls = new OrbitControls(camera, renderer.domElement)
    Object.assign(controls, { enableDamping: true, dampingFactor: 0.08, maxPolarAngle: 1.45, screenSpacePanning: false })
    const hemi = new THREE.HemisphereLight('#ffffff', '#1a1b1d', 1.5); scene.add(hemi)
    const sun = new THREE.DirectionalLight('#ffffff', 1.6); sun.position.set(0.6, 1, 0.4); scene.add(sun)
    const s = { hemi, sun, renderer, scene, camera, controls, track: new THREE.Group(), cars: new THREE.Group(), dirty: true, visible: true,
      look: new THREE.Vector3(), last: performance.now(), fly: null, rig: null, pos: {} }
    scene.add(s.track, s.cars)
    S.current = s
    const v = new THREE.Vector3(), want = new THREE.Vector3(), aim = new THREE.Vector3()
    let raf
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const now = performance.now(), dt = Math.min(0.1, (now - s.last) / 1000); s.last = now
      if (!s.visible || document.hidden || !s.T) return
      const L = live.current, playing = L.isPlaying()
      const p = L.progress()
      if (playing) {
        s.fps = s.fps ? s.fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1 : 60
        el.dataset.fps = s.fps.toFixed(0)
      }
      if (playing || s.dirty || s.p !== p) s.place?.(p)
      s.p = p
      // start sequence: in the Overview the camera shows the grid during the lights, then the whole circuit
      if (L.racing && L.mode === 'overview' && !s.manual && s.gridShot) {
        const want = p < LIGHTS_OUT ? 'grid' : 'race'
        if (s.shot !== want) { s.shot = want; s.flyTo(want === 'grid' ? s.gridShot : s.home) }
      }
      let moved = false
      if (s.fly) {                                           // smooth transition to overview / top
        const k = Math.min(1, (now - s.fly.t0) / 700), e = k * k * (3 - 2 * k)
        camera.position.lerpVectors(s.fly.p0, s.fly.p1, e); s.look.lerpVectors(s.fly.l0, s.fly.l1, e)
        camera.lookAt(s.look); moved = true
        if (k >= 1) { controls.target.copy(s.fly.l1); controls.enabled = true; s.fly = null }
      } else if ((L.mode === 'follow' || L.mode === 'onboard') && s.pos[L.selected] && s.rig && !s.manual) {
        // rig locked to the car (no lag at any replay speed); only its direction is smoothed, and entry is blended
        const c = s.pos[L.selected], len = CAR_LENGTH * s.cs, r = s.rig, ob = L.mode === 'onboard'
        if (!r.f) r.f = c.fwd.clone(); else r.f.lerp(c.fwd, 1 - Math.exp(-dt * (ob ? 8 : 3.5))).normalize()
        want.copy(c.p).addScaledVector(r.f, (ob ? 0.02 : -2.2) * len); want.y += (ob ? 0.20 : 0.8) * len
        aim.copy(c.p).addScaledVector(r.f, (ob ? 1.6 : 2) * len); if (ob) aim.y += 0.02 * len
        const k = Math.min(1, (now - r.t0) / 800), e = k * k * (3 - 2 * k)
        camera.position.lerpVectors(r.p0, want, e); s.look.lerpVectors(r.l0, aim, e); camera.lookAt(s.look)
        moved = k < 1 || r.f.distanceToSquared(c.fwd) > 1e-6
        el.dataset.follow = (camera.position.distanceTo(c.p) / len).toFixed(2)
      } else { moved = controls.update(); s.look.copy(controls.target) }
      const gridClose = L.racing && L.mode === 'overview' && p < LIGHTS_OUT && !s.manual
      const tracked = (L.mode === 'follow' || L.mode === 'onboard') && !s.manual
      const target = (s.fly || tracked) ? s.look : controls.target
      if (s.bounds) {
        s.limit = constrainCircuitCamera(camera, target, s.bounds, { manual: !!s.manual, close: gridClose || tracked || ((L.mode === 'follow' || L.mode === 'onboard') && s.manual), anchor: s.manualAnchor })
        Object.assign(controls, { minDistance: s.limit.min, maxDistance: s.limit.max, maxPolarAngle: s.limit.maxPolar })
        moved = moved || s.limit.changed
        const cleared = clampGround(tracked ? 0.35 * s.cs : 1.5 * s.cs)
        moved = moved || cleared
        if (!tracked && !s.fly) s.look.copy(controls.target)
      }
      if (!(moved || s.dirty || playing)) return
      s.dirty = false
      renderer.render(scene, camera)
      el.dataset.cam = camera.position.toArray().map(x => Math.round(x)).join(',')
      el.dataset.p = p.toFixed(3)
      s.placeTags?.()
    }
    // keep the free camera above the terrain and road (a ray straight down onto the ground meshes)
    const ray = new THREE.Raycaster(), downV = new THREE.Vector3(0, -1, 0), from = new THREE.Vector3()
    const clampGround = (clearance) => {
      if (!s.ground?.length) return
      ray.set(from.set(camera.position.x, 1e7, camera.position.z), downV)
      const hit = ray.intersectObjects(s.ground, false)[0], min = (hit ? hit.point.y : -Infinity) + clearance
      if (camera.position.y < min) { camera.position.y = min; camera.lookAt(s.look); return true }
    }
    s.flyTo = to => {
      controls.enabled = false
      s.fly = { t0: performance.now(), p0: camera.position.clone(), l0: s.look.clone(), p1: to.p.clone(), l1: to.l.clone() }
    }
    // the user takes the camera: automatic positioning (follow, onboard, grid shot) pauses until a camera button is pressed
    const takeOver = () => {
      if (s.manual) return
      s.manual = true; s.fly = null; controls.enabled = true
      controls.target.copy(s.look)
      const close = live.current.mode === 'follow' || live.current.mode === 'onboard'
      const car = s.pos[live.current.selected]
      s.manualAnchor = close && car ? car.p.clone().addScaledVector(car.fwd, live.current.mode === 'onboard' ? CAR_LENGTH * s.cs : 0) : null
      if (s.manualAnchor) { controls.target.copy(s.manualAnchor); camera.lookAt(controls.target); s.look.copy(controls.target) }
      el.dataset.manual = '1'
    }
    s.takeOver = takeOver
    const L0 = () => live.current
    loop()
    controls.addEventListener('change', () => { s.dirty = true })
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix(); s.reframe?.(); s.dirty = true
    })
    ro.observe(el)
    const io = new IntersectionObserver(([e]) => { s.visible = e.isIntersecting; s.dirty = true })
    io.observe(el)
    // picking in screen space (cars are small in the overview, so pick the nearest within 22 px)
    const pick = ev => {
      const r = el.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top
      let best = null, bd = 22 * 22
      for (const [id, c] of Object.entries(s.pos)) {
        v.copy(c.p).project(camera)
        if (v.z > 1) continue
        const d = ((v.x + 1) / 2 * r.width - mx) ** 2 + ((1 - v.y) / 2 * r.height - my) ** 2
        if (d < bd) { bd = d; best = id }
      }
      return best
    }
    let down = null
    const canvas = renderer.domElement
    const pd = ev => { down = ev.target === canvas ? { x: ev.clientX, y: ev.clientY, b: ev.button, moved: false } : null }
    const pu = ev => {
      if (down && !down.moved && down.b === 0 && ev.target === canvas) {
        const id = pick(ev)
        if (id) live.current.onSelect(id); else live.current.onToggle?.()      // empty track: pause / resume
      }
      down = null
    }
    let hv = null
    const pm = ev => {
      if (ev.buttons) {                                        // a drag (rotate or pan) is never a pause click
        if (down && !down.moved && Math.hypot(ev.clientX - down.x, ev.clientY - down.y) >= 5) { down.moved = true; takeOver() }
        return
      }
      const id = pick(ev); if (id !== hv) { hv = id; live.current.setHover(id); el.style.cursor = id ? 'pointer' : '' }
    }
    const wh = () => takeOver()
    el.addEventListener('pointerdown', pd, true); el.addEventListener('pointerup', pu); el.addEventListener('pointermove', pm, true)
    canvas.addEventListener('wheel', wh, { passive: true, capture: true })
    el.__apex = {                                            // read-only hooks for the browser tests
      screen: id => { const c = s.pos[id]; if (!c) return null; v.copy(c.p).project(camera); const r = el.getBoundingClientRect(); return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height] },
      models: () => Object.fromEntries(Object.entries(s.carModels || {}).map(([id, car]) => [id, { ...car.userData.appearance, meshes: car.children.length, bounds: new THREE.Box3().setFromObject(car).getSize(new THREE.Vector3()).toArray() }])),
      info: () => ({ calls: renderer.info.render.calls, triangles: renderer.info.render.triangles }),
      cars: () => Object.fromEntries(Object.entries(s.pos).map(([id, c]) => [id, { p: c.p.toArray(), d: c.d, lat: c.lat, rank: c.rank }])),
      size: () => ({ length: CAR_LENGTH * s.cs, width: CAR_WIDTH * s.cs, lane: s.lane, gap: s.gap, lift: s.lift }),
      road: d => frameAt(s.T, d).slice(0, 3),
      centre: () => s.home.l.toArray(),
      grid: () => s.gridBoxes,
      cameraLimits: () => ({ mode: live.current.mode, selected: live.current.selected, manual: !!s.manual, rig: !!s.rig, occupancy: s.bounds && (() => { camera.updateMatrixWorld(); const ps = s.bounds.points.map(p => p.clone().project(camera)); return { minX: Math.min(...ps.map(p => p.x)), maxX: Math.max(...ps.map(p => p.x)), minY: Math.min(...ps.map(p => p.y)), maxY: Math.max(...ps.map(p => p.y)) } })(), local: s.limit?.local, min: controls.minDistance, max: controls.maxDistance, distance: camera.position.distanceTo(controls.target), pan: s.panBounds && { min: s.panBounds.min.toArray(), max: s.panBounds.max.toArray() } }),
      clearance: () => { ray.set(from.set(camera.position.x, 1e7, camera.position.z), downV); const h = ray.intersectObjects(s.ground, false)[0]; return h ? camera.position.y - h.point.y : null },
      target: () => (s.manual || L0().mode === 'overview' || L0().mode === 'top' ? controls.target : s.look).toArray(),
      look: () => s.look.toArray(),
      tags: () => Object.fromEntries(Object.entries(tags.current).filter(([, t]) => t && t.style.opacity === '1').map(([id, t]) => [id, t.textContent])),
      bridge: () => s.T.at.map(c => ({ over: s.T.ys[c.j], under: s.T.ys[c.i], lift: s.T.lift[c.j] })),
      gpu: () => { const gl = renderer.getContext(), x = gl.getExtension('WEBGL_debug_renderer_info'); return x ? gl.getParameter(x.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER) },
    }
    return () => {
      cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); controls.dispose()
      el.removeEventListener('pointerdown', pd, true); el.removeEventListener('pointerup', pu); el.removeEventListener('pointermove', pm, true); canvas.removeEventListener('wheel', wh, true)
      dispose(scene); renderer.dispose(); renderer.domElement.remove(); S.current = null
    }
  }, [])

  // ---- circuit: road, terrain, framing ----
  useEffect(() => {
    const s = S.current
    if (!s) return
    dispose(s.track); s.track.clear()
    const { R, w, cs } = T, dy = R / 2500
    const maxLift = Math.max(...T.lift)
    // drawn car length is CAR_LENGTH * cs: keep slots at least 1.7 lengths apart and lanes 1.6 car widths apart
    Object.assign(s, { R, w, dy, cs, T, gap: Math.max(carGap(T), 1.7 * CAR_LENGTH * cs), lane: 3.8 * w, gridHalf: 2.6 * w })
    const mat = o => new THREE.MeshStandardMaterial({ roughness: 0.95, ...o })
    const road = [
      new THREE.Mesh(ribbon(T, maxLift ? k => (HALF + 14 - 12.2 * Math.min(1, T.lift[k] / (0.2 * maxLift))) * w : (HALF + 14) * w, 1 * dy), mat({ color: '#232427' })),   // run-off narrows onto the bridge
      new THREE.Mesh(ribbon(T, (HALF + 1.8) * w, 2 * dy, true), mat({ vertexColors: true })),
      new THREE.Mesh(ribbon(T, HALF * w, 3 * dy), s.roadMat = mat({ color: '#45474c', roughness: 0.85 })),
    ]
    const [sx, sy, sz, hd] = frameAt(T, 0)
    const line = new THREE.Mesh(new THREE.PlaneGeometry(HALF * 2 * w, 5 * w), new THREE.MeshBasicMaterial({ map: checker(), side: THREE.DoubleSide }))
    line.rotation.set(-Math.PI / 2, 0, hd + Math.PI / 2); line.position.set(sx, sy + 4 * dy, sz)
    s.track.add(...road, line)
    s.ground = [...road]
    if (maxLift) s.track.add(new THREE.Mesh(skirts(T, (HALF + 1.8) * w, dy, (HALF + 6) * w), mat({ color: '#26282c', side: THREE.DoubleSide })))
    const yMin = Math.min(...T.ground)
    if (hasElev && geometry.terrain) {
      const inner = (HALF + 16) * w
      const land = new THREE.Mesh(terrain(T, geometry.terrain, inner, inner + R * 0.08, 2 * dy), contourMaterial())
      s.track.add(land); s.ground.push(land)
      const base = new THREE.Mesh(new THREE.CircleGeometry(R * 6, 48), mat({ color: '#141517' }))
      base.rotation.x = -Math.PI / 2; base.position.y = Math.min(yMin, ...geometry.terrain.h) - 2
      s.track.add(base); s.ground.push(base)
    } else {
      const ground = new THREE.Mesh(new THREE.CircleGeometry(R * 6, 64), mat({ color: '#151618' }))
      ground.rotation.x = -Math.PI / 2; ground.position.y = -dy
      const grid = new THREE.GridHelper(R * 3.6, 36, '#2a2b2e', '#1f2022')
      s.track.add(ground, grid); s.ground.push(ground)
    }
    s.lift = 3 * dy + 0.04
    s.scene.fog.near = R * 4; s.scene.fog.far = R * 9
    s.camera.near = R / 200; s.camera.far = R * 20; s.camera.updateProjectionMatrix()
    // frame the circuit for this stage: move the camera along its view direction until every track point is inside
    // the picture with a margin (perspective makes the near side larger, so a bounding-box formula crops it)
    const x0 = Math.min(...T.xs), x1 = Math.max(...T.xs), z0 = Math.min(...T.zs), z1 = Math.max(...T.zs)
    const midY = (Math.max(...T.ys) + yMin) / 2, centre = new THREE.Vector3((x0 + x1) / 2, midY, (z0 + z1) / 2)
    const el = wrap.current, cam = s.camera.clone()
    const roadPoints = T.xs.map((x, i) => new THREE.Vector3(x, T.ys[i] + 3 * dy, T.zs[i]))
    const fitPoints = roadPoints.flatMap((p, i) => { const q = roadPoints[(i + 1) % T.n], dx = q.x - p.x, dz = q.z - p.z, len = Math.hypot(dx, dz) || 1; return [-1, 1].map(side => new THREE.Vector3(p.x - side * dz / len * 21 * w, p.y, p.z + side * dx / len * 21 * w)) })
    const land = geometry.terrain
    const terrainBounds = new THREE.Box3(new THREE.Vector3(land?.x0 ?? x0 - 21, yMin, land?.z0 ?? z0 - 21), new THREE.Vector3(land ? land.x0 + (land.n - 1) * land.dx : x1 + 21, Math.max(...T.ys), land ? land.z0 + (land.n - 1) * land.dz : z1 + 21))
    s.bounds = { points: fitPoints, road: roadPoints, centre, radius: Math.hypot(x1 - x0, z1 - z0) / 2, terrain: terrainBounds, carLength: CAR_LENGTH * cs }
    s.panBounds = new THREE.Box3(
      new THREE.Vector3(x0 - 30 * w, yMin, z0 - 30 * w),
      new THREE.Vector3(x1 + 30 * w, Math.max(...T.ys), z1 + 30 * w))
    s.reframe = () => {
      cam.aspect = el.clientWidth / Math.max(1, el.clientHeight) || 1.5; cam.updateProjectionMatrix()
      const frame = dir => { const fit = circuitFrame(fitPoints, cam, dir, centre); return { p: fit.target.clone().addScaledVector(dir, fit.distance), l: fit.target } }
      // Pick an orbit heading that fits each circuit into the actual viewer aspect ratio.
      const candidates = Array.from({ length: 12 }, (_, i) => {
        const az = i * Math.PI / 12
        return frame(new THREE.Vector3(Math.sin(az), 1.05, Math.cos(az)).normalize())
      })
      s.home = candidates.reduce((best, shot) => shot.p.distanceToSquared(centre) < best.p.distanceToSquared(centre) ? shot : best)
      s.top = frame(new THREE.Vector3(0, 1, 0.0005).normalize())
      s.camera.aspect = cam.aspect; s.camera.updateProjectionMatrix()
      if (!s.manual && (live.current.mode === 'overview' || live.current.mode === 'top')) {
        const shot = live.current.mode === 'top' ? s.top : s.home
        s.controls.maxDistance = shot.p.distanceTo(shot.l) // update before OrbitControls can apply the old viewport limit
        s.fly = null; s.controls.enabled = true; s.camera.position.copy(shot.p); s.controls.target.copy(shot.l); s.look.copy(shot.l)
        s.camera.lookAt(shot.l)
      }
    }
    s.manual = false; s.manualAnchor = null; delete el.dataset.manual; s.reframe()
    s.shot = null
    s.dirty = true
    wrap.current.dataset.circuit = circuitId
  }, [T])

  // ---- locally loaded GLB cars: shared geometry, four animated wheel pivots ----
  useEffect(() => {
    const s = S.current
    if (!s) return
    dispose(s.cars); s.cars.clear()
    const n = entries.length, models = {}, roots = entries.map(() => new THREE.Group())
    let cancelled = false
    s.carModels = models
    setAssets({ loading: n, failed: [] })
    preloadCars(entries)
    entries.forEach((entry, i) => {
      roots[i].userData.driver = entry.driver_id
      s.cars.add(roots[i])
      createCar(entry).then(car => {
        if (cancelled) { releaseCar(car); return }
        models[entry.driver_id] = car
        roots[i].add(car)
        setAssets(a => ({ ...a, loading: a.loading - 1 }))
        s.place?.(live.current.progress()); s.dirty = true
      }).catch(error => {
        if (cancelled) return
        console.warn('APEX car asset:', entry.driver_id, error)
        setAssets(a => ({ loading: a.loading - 1, failed: [...a.failed, entry.driver_id] }))
      })
    })
    const ring = new THREE.Mesh(new THREE.RingGeometry(3.6, 3.9, 48), new THREE.MeshBasicMaterial({ color: LIME, side: THREE.DoubleSide, transparent: true, opacity: 0.75 }))
    ring.visible = false
    // screen-size markers so every car and its team colour stay readable from the wide views (visual aid only)
    const dots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 13, sizeAttenuation: false, vertexColors: true, map: disc(), alphaTest: 0.4, transparent: true, depthTest: false }))
    dots.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3))
    dots.geometry.setAttribute('color', new THREE.Float32BufferAttribute(entries.flatMap(e => new THREE.Color(teamColor(e.constructor_id)).toArray()), 3))
    const halo = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 30, sizeAttenuation: false, color: LIME, map: disc(true), alphaTest: 0.3, transparent: true, depthTest: false }))
    halo.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3))
    dots.frustumCulled = halo.frustumCulled = false; dots.renderOrder = 2; halo.renderOrder = 3
    s.cars.add(ring, dots, halo)
    const dp = dots.geometry.attributes.position, hp = halo.geometry.attributes.position
    const e3 = new THREE.Euler(0, 0, 0, 'YXZ'), up = new THREE.Vector3(0, 1, 0)
    s.place = p => {
      const L = live.current, at = replayAt(s.T, L.slots, p, s.gap), lane = s.lane
      entries.forEach((e, i) => {
        // lateral: grid file (pole left, P2 right, ...) blending into the racing line; overtaking line on the other side
        const [d, ln, gs] = at[e.driver_id], lat = (ln - 0.4 * (1 - Math.abs(gs))) * lane + gs * s.gridHalf
        const [x, y, z, hd, pt] = frameAt(s.T, d)
        const c = s.pos[e.driver_id] ||= { p: new THREE.Vector3(), fwd: new THREE.Vector3() }
        c.p.set(x - Math.cos(hd) * lat, y + s.lift, z + Math.sin(hd) * lat)
        c.d = d; c.lat = lat
        dp.setXYZ(i, c.p.x, c.p.y + 1.6 * s.cs, c.p.z)
        c.fwd.set(Math.sin(hd) * Math.cos(pt), Math.sin(pt), Math.cos(hd) * Math.cos(pt))
        e3.set(-pt, hd, 0)
        const root = roots[i]
        root.position.copy(c.p); root.rotation.copy(e3); root.scale.setScalar(s.cs)
        const ahead = frameAt(s.T, d + 4)[3], behind = frameAt(s.T, d - 4)[3]
        const turn = Math.atan2(Math.sin(ahead - behind), Math.cos(ahead - behind))
        models[e.driver_id]?.userData.animate(d / s.cs, Math.max(-0.35, Math.min(0.35, turn * 0.45)))
      })
      dp.needsUpdate = true
      entries.map(e => e.driver_id).sort((a, b) => s.pos[b].d - s.pos[a].d).forEach((id, k) => { s.pos[id].rank = k + 1 })   // running order
      const wide = L.mode === 'overview' || L.mode === 'top'
      const c = s.pos[L.selected]
      dots.visible = wide && L.markers; halo.visible = dots.visible && !!c
      if (c) { hp.setXYZ(0, c.p.x, c.p.y + 1.6 * s.cs, c.p.z); hp.needsUpdate = true }
      ring.visible = !!c && L.mode !== 'onboard'
      if (c) { ring.position.copy(c.p).addScaledVector(up, 0.05 * s.cs); ring.scale.setScalar(s.cs); ring.rotation.set(-Math.PI / 2, 0, 0) }
      s.dirty = true
    }
    // grid boxes: a white bracket ahead of each starting slot, painted on the road; slot 0 (pole) on the left
    const len = CAR_LENGTH * s.cs, bw = CAR_WIDTH * s.cs * 1.45, th = 0.09 * len, pos = []
    const quad = (d, lat, along, across) => {                  // rectangle centred at (d, lat), sizes in metres
      const [x, y, z, hd] = frameAt(s.T, d), fx = Math.sin(hd), fz = Math.cos(hd), rx = -Math.cos(hd), rz = Math.sin(hd)
      const cx = x + rx * lat, cz = z + rz * lat, yy = y + 3.5 * s.dy
      const P = (a, b) => [cx + fx * a + rx * b, yy, cz + fz * a + rz * b]
      const [p1, p2, p3, p4] = [P(-along / 2, -across / 2), P(along / 2, -across / 2), P(along / 2, across / 2), P(-along / 2, across / 2)]
      pos.push(...p1, ...p2, ...p3, ...p1, ...p3, ...p4)
    }
    s.gridBoxes = []
    entries.forEach((_, i) => {
      const d = -(i + 0.5) * s.gap, lat = gridSide(i) * s.gridHalf, front = d + 0.62 * len
      quad(front, lat, th, bw)                                                     // line across, in front of the car
      for (const side of [-1, 1]) quad(front - 0.22 * len, lat + side * bw / 2, 0.44 * len, th)   // short side marks
      s.gridBoxes.push({ slot: i, d, lat, side: gridSide(i), heading: frameAt(s.T, d)[3] })
    })
    const boxGeo = new THREE.BufferGeometry(); boxGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    const boxes = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({ color: '#e9e9e6', side: THREE.DoubleSide }))
    s.cars.add(boxes)
    // grid camera: behind and above the back of the grid, looking up the starting straight
    const tail = frameAt(s.T, -(n + 3) * s.gap), mid = frameAt(s.T, -0.35 * n * s.gap), hgt = Math.max(n * s.gap * 0.32, 12 * len)
    s.gridShot = { p: new THREE.Vector3(tail[0], tail[1] + hgt, tail[2]), l: new THREE.Vector3(mid[0], mid[1], mid[2]) }
    s.shot = null
    for (const k of Object.keys(s.pos)) if (!entries.some(e => e.driver_id === k)) delete s.pos[k]
    s.place(live.current.progress())
    return () => { cancelled = true; Object.values(models).forEach(releaseCar) }
  }, [entries, T])

  useEffect(() => { const s = S.current; if (s) { s.place?.(live.current.progress()); s.dirty = true } }, [slots, selected, markers, mode])
  useEffect(() => { const s = S.current; if (s) s.shot = null }, [result])        // a new race: show the grid again
  useEffect(() => { const s = S.current; if (s?.rig) s.rig = { f: null, t0: performance.now(), p0: s.camera.position.clone(), l0: s.look.clone() } }, [selected])   // glide to the newly selected car

  // ---- conditions: visual only (the prediction uses the model inputs, not these effects) ----
  const wx = conditions?.weather ?? 'dry', rain = conditions?.rain_mm_h ?? 0
  useEffect(() => {
    const s = S.current
    if (!s?.roadMat) return
    const k = wx === 'wet' ? 1 : wx === 'damp' ? 0.45 : 0                // wetness of the asphalt
    s.roadMat.color.set('#45474c').lerp(new THREE.Color('#25272b'), k)
    Object.assign(s.roadMat, { roughness: 0.85 - 0.6 * k, metalness: 0.25 * k })
    s.hemi.intensity = 1.5 - 0.45 * k; s.sun.intensity = 1.6 - 0.9 * k
    s.scene.fog.near = s.R * (4 - 2 * k); s.scene.fog.far = s.R * (9 - 4 * k)
    wrap.current.dataset.weather = wx
    s.dirty = true
  }, [wx, T])

  // ---- camera modes ----
  useEffect(() => {
    const s = S.current
    if (!s?.home) return
    const close = mode === 'follow' || mode === 'onboard'
    s.manual = false; s.manualAnchor = null; delete wrap.current.dataset.manual
    s.camera.near = 0.05; s.camera.updateProjectionMatrix()
    if (close) { s.fly = null; s.controls.enabled = true; s.rig = { f: null, t0: performance.now(), p0: s.camera.position.clone(), l0: s.look.clone() }; if (!selected && entries[0]) onSelect(entries[0].driver_id) }
    else { s.shot = mode === 'overview' && live.current.racing && s.p < LIGHTS_OUT ? null : 'race'; s.flyTo(mode === 'top' ? s.top : s.home) }
    wrap.current.dataset.mode = mode
    if (!close) delete wrap.current.dataset.follow
    s.dirty = true
  }, [mode, modeN, T])
  useEffect(() => { if (focus?.n) { setMode(m => (m === 'onboard' ? m : 'follow')); setModeN(n => n + 1) } }, [focus])

  // ---- HTML tags over the canvas: hovered and selected cars, and the top three of the running order ----
  const tagIds = entries.map(e => e.driver_id)
  useEffect(() => {
    const s = S.current
    if (!s) return
    const v = new THREE.Vector3()
    s.placeTags = () => {
      const w = wrap.current, L = live.current
      const first = tagIds.find(id => s.pos[id]?.rank === 1)
      for (const id of tagIds) {
        const el = tags.current[id], c = s.pos[id]
        if (!el || !c) continue
        // racing: the leader carries a live tag; at the flag the podium is listed above the winner
        const done = s.p >= 1, pod = L.racing && done && c.rank <= 3, lead = L.racing && c.rank === 1
        const show = !(id === L.selected && L.mode === 'onboard') && (id === L.selected || id === L.hover || lead || pod)
        if (!show) { el.style.opacity = 0; continue }
        const own = id === L.selected || id === L.hover || !pod
        v.copy(own ? c.p : s.pos[first].p).project(s.camera)
        el.textContent = `${L.racing ? `P${c.rank} ` : ''}${L.names(id, true)}`
        el.style.transform = `translate(${((v.x + 1) / 2) * w.clientWidth}px, ${((1 - v.y) / 2) * w.clientHeight - 26 - (own ? 0 : (3 - c.rank) * 21)}px) translate(-50%, -50%)`
        el.style.opacity = v.z < 1 ? 1 : 0
      }
    }
    s.dirty = true
  }, [tagIds.join(), selected, hover, result])

  const elev = geometry.elevation
  return (
    <div className="track3d" ref={wrap} data-viewer="3d" data-quality="high" data-elevation-scale="1" data-elev={hasElev ? 'dem' : 'flat'} role="img"
      aria-label={`3D view of ${geometry.name}. Click the track to pause or resume the race, click a car to select it. Drag to rotate, scroll to zoom, right-drag to pan.`}>
      {renderError && <div className="track-loading" role="alert">{renderError}</div>}
      {tagIds.map(id => (
        <span key={id} ref={el => (tags.current[id] = el)} className={`tag3d${id === selected ? ' sel' : ''}`} data-driver={id} />
      ))}
      {rain > 0 && <div className={`rain${paused ? ' paused' : ''}`} aria-hidden="true" style={{ '--rain': Math.min(1, 0.25 + rain / 2.5) }} data-rain={rain} />}
      {conditions && <span className="wind-chip" title="Wind speed before the start (a model input for weather-aware models)" data-wind={conditions.wind_kmh}>
        <svg viewBox="0 0 20 12" width="18" height="11" aria-hidden="true"><path d="M1 3h11a2.5 2.5 0 1 0-2.5-2.5M1 6.5h15a2.5 2.5 0 1 1-2.5 2.5M1 10h7" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>
        {Math.round(conditions.wind_kmh)} km/h · {Math.round(conditions.air_temp_c)} °C</span>}
      <div className="cam-bar">
        <div className="seg sm" role="radiogroup" aria-label="Camera">
          {MODES.map(([k, l]) => <button key={k} role="radio" aria-checked={mode === k} className={mode === k ? 'on' : ''} onClick={() => pickMode(k)}>{l}</button>)}
        </div>
        <button className="btn ghost sm" onClick={() => pickMode('overview')}>Reset view</button>
      </div>
      <div className="asset-note" role="status" data-assets-loading={assets.loading} data-assets-failed={assets.failed.length}>
        {assets.loading > 0 ? `Loading 3D cars… ${entries.length - assets.loading}/${entries.length}`
          : assets.failed.length ? `${assets.failed.length} car assets unavailable · driver markers retained` : 'Local GLB cars · original approximate liveries'}
        {selected && (() => { const entry = entries.find(e => e.driver_id === selected); if (!entry) return null; const a = carAppearance(entry); return <span> · {a.modelGap || `${a.era} · ${a.asset?.quality || 'asset unavailable'}`}</span> })()}
        {' · '}<a href="/models/ATTRIBUTION.html" target="_blank" rel="noreferrer">3D credits</a>
      </div>
      <div className="view-opts">
        <button className="btn ghost sm" data-markers aria-pressed={markers} onClick={() => setMarkers(m => !m)}
          title="Team-colour markers above the cars in the Overview and Top views. Visual aid only: cars and track keep their size.">Markers {markers ? 'on' : 'off'}</button>
      </div>
      {elev && <span className="elev-note">{hasElev
        ? `Elevation: Copernicus DEM 30 m · ${elev.range_m} m range · true scale 1:1${B ? ` · crossover bridge approx. +${B.clearance_m} m` : ''}`
        : geometry.source
          ? <>Mapped layout · flat elevation · <a href="https://www.openstreetmap.org/copyright">© OpenStreetMap contributors</a></>
          : 'Flat: no reliable elevation for this circuit'}</span>}
    </div>
  )
}
