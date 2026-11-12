import * as THREE from 'three'

const Y = new THREE.Vector3(0, 1, 0)

// Solve the perspective frustum inequalities, including sample depth. Recentring the
// projected bounds avoids wasting half the viewport on asymmetric/long circuits.
export function circuitFrame(points, camera, back, centre) {
  const right = new THREE.Vector3().crossVectors(Y, back).normalize()
  const up = new THREE.Vector3().crossVectors(back, right).normalize()
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.getEffectiveFOV()) / 2)
  const h = tanV * camera.aspect * 0.9, v = tanV * 0.82
  let xLow = Infinity, xHigh = -Infinity, yLow = Infinity, yHigh = -Infinity, zMax = -Infinity
  const offset = new THREE.Vector3()
  for (const point of points) {
    offset.copy(point).sub(centre)
    const x = offset.dot(right), y = offset.dot(up), z = offset.dot(back)
    xLow = Math.min(xLow, x - h * z); xHigh = Math.max(xHigh, x + h * z)
    yLow = Math.min(yLow, y - v * z); yHigh = Math.max(yHigh, y + v * z)
    zMax = Math.max(zMax, z)
  }
  const distance = Math.max(1, zMax + 1, (xHigh - xLow) / (2 * h), (yHigh - yLow) / (2 * v))
  const target = centre.clone().addScaledVector(right, (xHigh + xLow) / 2).addScaledVector(up, (yHigh + yLow) / 2)
  return { distance, target }
}
export const circuitFit = (...args) => circuitFrame(...args).distance

function nearestRoad(target, road) {
  let best = Infinity, nearest = road[0].clone()
  for (let i = 0; i < road.length; i++) {
    const a = road[i], b = road[(i + 1) % road.length], dx = b.x - a.x, dz = b.z - a.z
    const t = THREE.MathUtils.clamp(((target.x - a.x) * dx + (target.z - a.z) * dz) / (dx * dx + dz * dz || 1), 0, 1)
    const x = a.x + t * dx, z = a.z + t * dz, d = (target.x - x) ** 2 + (target.z - z) ** 2
    if (d < best) { best = d; nearest.set(x, a.y + t * (b.y - a.y), z) }
  }
  return nearest
}

// Called after controls AND automatic camera motion, before every render. OrbitControls'
// maxDistance alone cannot constrain scripted transitions, aspect changes or off-centre panning.
export function constrainCircuitCamera(camera, target, bounds, { manual = false, close = false, anchor = null } = {}) {
  const before = camera.position.clone(), oldTarget = target.clone()
  const back = camera.position.clone().sub(target)
  let distance = Math.max(back.length(), 0.01); back.normalize()
  const local = manual && (close || distance < bounds.radius * 0.22)
  const maxPolar = local || close ? 1.49 : 1.12
  if (manual || !close) {
    const polar = Math.acos(THREE.MathUtils.clamp(back.y, -1, 1))
    if (polar > maxPolar) { back.y = Math.cos(maxPolar); const h = Math.hypot(back.x, back.z) || 1; back.x = back.x / h * Math.sin(maxPolar); back.z = back.z / h * Math.sin(maxPolar) }
  }
  const min = bounds.carLength * 0.7
  const frame = circuitFrame(bounds.points, camera, back, bounds.centre)
  let max = frame.distance
  if (manual && close) max = Math.min(max, bounds.carLength * 12)
  distance = THREE.MathUtils.clamp(distance, close && !manual ? 0.05 : min, max)
  const right = new THREE.Vector3().crossVectors(Y, back).normalize(), up = new THREE.Vector3().crossVectors(back, right).normalize()
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.getEffectiveFOV()) / 2)
  if (manual || !close) {
    const origin = local ? (anchor || nearestRoad(target, bounds.road)) : frame.target
    const delta = target.clone().sub(origin), fraction = local ? 0.22 : 0.07
    const x = THREE.MathUtils.clamp(delta.dot(right), -distance * tanV * camera.aspect * fraction, distance * tanV * camera.aspect * fraction)
    const y = THREE.MathUtils.clamp(delta.dot(up), -distance * tanV * fraction, distance * tanV * fraction)
    const z = local ? THREE.MathUtils.clamp(delta.dot(back), -distance * fraction, distance * fraction) : 0
    target.copy(origin).addScaledVector(right, x).addScaledVector(up, y).addScaledVector(back, z)
    // Targets remain inside the measured terrain footprint even on unusual/asymmetric circuits.
    target.x = THREE.MathUtils.clamp(target.x, bounds.terrain.min.x, bounds.terrain.max.x)
    target.z = THREE.MathUtils.clamp(target.z, bounds.terrain.min.z, bounds.terrain.max.z)
  }
  camera.position.copy(target).addScaledVector(back, distance)
  camera.near = Math.max(0.04, Math.min(0.25, distance / 500))
  camera.updateProjectionMatrix(); camera.lookAt(target); camera.updateMatrixWorld()
  return { min, max, maxPolar, distance, local, changed: before.distanceToSquared(camera.position) > 1e-8 || oldTarget.distanceToSquared(target) > 1e-8 }
}
