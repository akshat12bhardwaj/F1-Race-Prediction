import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { orientCarGeometry } from './carGeometry.js'
import { carAppearance } from './carAssets.js'
export { CAR_LENGTH, CAR_WIDTH } from './carSize.js'

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder)
const cache = new Map()

// Decode each local asset once. Share merged geometry, keep four wheel pivots.
async function template(asset) {
  if (!cache.has(asset.url)) cache.set(asset.url, loader.loadAsync(asset.url).then(gltf => {
    const scene = gltf.scene
    scene.rotation.y = asset.rotation
    scene.updateMatrixWorld(true)
    const bounds = new THREE.Box3().setFromObject(scene), size = bounds.getSize(new THREE.Vector3())
    const scale = Math.min(asset.length / size.z, asset.width / size.x)
    const centre = bounds.getCenter(new THREE.Vector3())
    const norm = new THREE.Matrix4().makeScale(scale, scale, scale)
      .multiply(new THREE.Matrix4().makeTranslation(-centre.x, -bounds.min.y, -centre.z))
    const buckets = new Map()
    scene.traverse(o => {
      if (!o.isMesh) return
      let a = o, wheel = null, assembly = ''
      while (a && a !== scene) {
        if (/^(front|rear)-(left|right)-wheel$/.test(a.name) || /^wheel-(front|rear)-(left|right)$/.test(a.name)) wheel = a.name
        assembly += ` ${a.name}`; a = a.parent
      }
      const key = wheel || 'chassis', g = new THREE.BufferGeometry()
      for (const name of ['position', 'normal']) {
        const source = o.geometry.attributes[name], data = []
        for (let i = 0; i < source.count; i++) data.push(source.getX(i), source.getY(i), source.getZ(i))
        g.setAttribute(name, new THREE.Float32BufferAttribute(data, 3))
      }
      if (o.geometry.index) g.setIndex(o.geometry.index.clone())
      orientCarGeometry(g, o.name, norm.clone().multiply(o.matrixWorld), asset.id === 'formula-f222')
      const raw = g.index ? g.toNonIndexed() : g
      const count = raw.attributes.position.count, colors = [], roles = [], positions = [], normals = [], surfaces = []
      const mat = Array.isArray(o.material) ? o.material[0] : o.material
      const paint = /livery-pearl|^paint$|^white$/.test(mat.name) || (!wheel && /bodywork/.test(assembly))
      const c = mat.color || new THREE.Color('#111215')
      for (let i = 0; i < count; i++) {
        colors.push(c.r, c.g, c.b); roles.push(paint ? 1 : 0); surfaces.push(mat.roughness ?? 0.65, mat.metalness ?? 0.1)
        positions.push(raw.attributes.position.getX(i), raw.attributes.position.getY(i), raw.attributes.position.getZ(i))
        normals.push(raw.attributes.normal.getX(i), raw.attributes.normal.getY(i), raw.attributes.normal.getZ(i))
      }
      // Read accessors: compressed glTF may use normalized integer attributes.
      const clean = new THREE.BufferGeometry()
      clean.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
      clean.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
      clean.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
      clean.setAttribute('paintRole', new THREE.Float32BufferAttribute(roles, 1))
      clean.setAttribute('surface', new THREE.Float32BufferAttribute(surfaces, 2))
      if (!buckets.has(key)) buckets.set(key, [])
      buckets.get(key).push(clean)
      if (raw !== g) raw.dispose(); g.dispose()
    })
    const parts = []
    for (const [name, geometries] of buckets) {
      const geometry = mergeGeometries(geometries)
      geometries.forEach(g => g.dispose())
      geometry.computeBoundingBox()
      const pivot = name === 'chassis' ? new THREE.Vector3() : geometry.boundingBox.getCenter(new THREE.Vector3())
      geometry.translate(-pivot.x, -pivot.y, -pivot.z); geometry.computeBoundingSphere()
      parts.push({ name, geometry, pivot })
    }
    const originalGeometries = new Set(), originalMaterials = new Set()
    scene.traverse(o => { if (o.isMesh) { originalGeometries.add(o.geometry); [o.material].flat().forEach(m => originalMaterials.add(m)) } })
    originalGeometries.forEach(g => g.dispose()); originalMaterials.forEach(m => m.dispose())
    return parts
  }).catch(err => { cache.delete(asset.url); throw err }))
  return cache.get(asset.url)
}

function paintMaterial(palette) {
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.46, metalness: 0.18 })
  m.onBeforeCompile = sh => {
    sh.uniforms.apPrimary = { value: new THREE.Color(palette[0]) }
    sh.uniforms.apAccent = { value: new THREE.Color(palette[1]) }
    sh.uniforms.apTrim = { value: new THREE.Color(palette[2]) }
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute float paintRole; attribute vec2 surface; varying vec2 vSurface; varying float vPaint; varying vec3 vCar;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvPaint = paintRole; vCar = position; vSurface = surface;')
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec2 vSurface; varying float vPaint; varying vec3 vCar; uniform vec3 apPrimary; uniform vec3 apAccent; uniform vec3 apTrim;')
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = vSurface.x;')
      .replace('#include <metalnessmap_fragment>', '#include <metalnessmap_fragment>\nmetalnessFactor = vSurface.y;')
      .replace('#include <color_fragment>', `#include <color_fragment>
        if (vPaint > 0.5) {
          float side = smoothstep(0.36, 0.48, abs(vCar.x));
          float stripe = 1.0 - smoothstep(0.065, 0.10, abs(abs(vCar.x) - 0.19));
          vec3 paint = mix(apPrimary, apAccent, side * 0.92);
          diffuseColor.rgb = mix(paint, apTrim, stripe);
        }`)
  }
  return m
}

export async function createCar(entry) {
  const appearance = carAppearance(entry)
  if (!appearance.asset) throw new Error(`${appearance.era}: licensed model unavailable`)
  const parts = await template(appearance.asset), root = new THREE.Group(), material = paintMaterial(appearance.palette)
  const wheels = []
  for (const part of parts) {
    const mesh = new THREE.Mesh(part.geometry, material)
    mesh.name = part.name; mesh.position.copy(part.pivot); mesh.userData.sharedCarGeometry = true
    if (part.name !== 'chassis') {
      const steer = new THREE.Group(); steer.position.copy(part.pivot); mesh.position.set(0, 0, 0)
      steer.add(mesh); root.add(steer); wheels.push({ mesh, steer, front: part.name.includes('front'), radius: Math.max(0.1, part.pivot.y) })
    } else root.add(mesh)
  }
  root.userData.appearance = appearance
  // Absolute travelled distance keeps wheels correct when scrubbing, pausing or restarting.
  root.userData.animate = (distance, steerAngle) => wheels.forEach(w => { w.mesh.rotation.x = distance / w.radius; w.steer.rotation.y = w.front ? steerAngle : 0 })
  return root
}
export function releaseCar(root) {
  const materials = new Set(); root.traverse(o => { if (o.material) materials.add(o.material) }); materials.forEach(m => m.dispose())
}
export function preloadCars(entries) {
  return Promise.allSettled([...new Map(entries.map(e => { const a = carAppearance(e).asset; return [a?.url, a] })).values()].filter(Boolean).map(template))
}
