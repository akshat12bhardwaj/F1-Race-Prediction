// Track geometry and the illustrative race replay (pure functions; unit-tested in e2e/motion.test.mjs).
//
// The race is NOT a physics or lap-time simulation. It tells the story of the models' answer in about ten seconds:
// every car leaves its starting slot, the field settles through a planned sequence of overtakes, and each car
// arrives in its predicted finishing slot. Only the final order is predicted; the overtakes are illustrative.

export const LAP_SECONDS = 14      // a lap at full straight-line speed (corners make it ~17-20 s at 1x)
export const RACE_SECONDS = 10     // the whole visual race at 1x: one lap, grid to chequered flag
export const MAX_DT = 0.25         // clamp only real stalls (tab switches); slow frame rates still take 10 s
const MIN_CORNER = 0.45            // slowest speed factor in the tightest corner

// Closed centreline -> arc length, corner-speed factor and cumulative pace table. ys (optional) = height per point.
export function buildTrack(xs, zs, ys = null) {
  const n = xs.length
  ys = ys || new Array(n).fill(0)
  const cum = new Float64Array(n + 1)
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    cum[i + 1] = cum[i] + Math.hypot(xs[j] - xs[i], zs[j] - zs[i])
  }
  const L = cum[n]
  const turn = new Float64Array(n), w = 3          // turning angle over a +-3 sample window ~ corner sharpness
  for (let i = 0; i < n; i++) {
    const a = (i - w + n) % n, b = (i + w) % n
    const h1 = Math.atan2(zs[i] - zs[a], xs[i] - xs[a]), h2 = Math.atan2(zs[b] - zs[i], xs[b] - xs[i])
    let d = Math.abs(h2 - h1)
    if (d > Math.PI) d = 2 * Math.PI - d
    turn[i] = d
  }
  const raw = Array.from(turn, t => Math.max(MIN_CORNER, 1 - t / 1.6))
  const speed = raw.map((_, i) => {                // smoothed so braking and acceleration are gradual
    let s = 0
    for (let k = -6; k <= 6; k++) s += raw[(i + k + n) % n]
    return s / 13
  })
  const time = new Float64Array(n + 1)             // seconds to reach each point at 1x (one lap)
  const v0 = L / LAP_SECONDS
  for (let i = 0; i < n; i++) time[i + 1] = time[i] + (cum[i + 1] - cum[i]) / (v0 * (speed[i] + speed[(i + 1) % n]) / 2)
  return { xs, zs, ys, cum, L, speed, time, lap: time[n], n }
}

function locate(T, d) {
  const u = ((d % T.L) + T.L) % T.L
  let lo = 0, hi = T.n
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T.cum[m] <= u) lo = m; else hi = m }
  return [lo, (u - T.cum[lo]) / Math.max(1e-9, T.cum[lo + 1] - T.cum[lo])]
}

// Point on the track surface at distance d: [x, y, z, heading, pitch]. Heading/pitch use a short central
// difference (about one car length either side) so orientation turns smoothly through corners and crests.
export function frameAt(T, d) {
  const [i, f] = locate(T, d), j = (i + 1) % T.n
  const lerp = a => a[i] + (a[j] - a[i]) * f
  const h = Math.max(4, T.L / T.n), p0 = locate(T, d - h), p1 = locate(T, d + h)
  const at = (a, [k, g]) => a[k] + (a[(k + 1) % T.n] - a[k]) * g
  const dx = at(T.xs, p1) - at(T.xs, p0), dz = at(T.zs, p1) - at(T.zs, p0), dy = at(T.ys, p1) - at(T.ys, p0)
  return [lerp(T.xs), lerp(T.ys), lerp(T.zs), Math.atan2(dx, dz), Math.atan2(dy, Math.hypot(dx, dz))]
}

// 2D helper kept for the flat fallback view: [x, z, heading].
export function pointAt(T, d) { const [x, , z, hd] = frameAt(T, d); return [x, z, hd] }

// Distance covered after `sec` seconds of racing (corner-aware: slower through corners).
export function distAtTime(T, sec) {
  const laps = Math.floor(sec / T.lap), r = sec - laps * T.lap
  let lo = 0, hi = T.n
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T.time[m] <= r) lo = m; else hi = m }
  const f = (r - T.time[lo]) / Math.max(1e-9, T.time[lo + 1] - T.time[lo])
  return laps * T.L + T.cum[lo] + (T.cum[lo + 1] - T.cum[lo]) * f
}

export const carGap = T => Math.min(T.L * 0.006, 40)
const smooth = t => t * t * (3 - 2 * t)
const clamp01 = t => Math.min(1, Math.max(0, t))

// Race clock: progress after a frame of dt seconds. Time-based, so the race lasts RACE_SECONDS at any frame rate.
export const advance = (p, dt, speed = 1, seconds = RACE_SECONDS) =>
  Math.min(1, p + (Math.min(MAX_DT, Math.max(0, dt)) * speed) / seconds)

// Overtake plan: the grid order is sorted into the finishing order by rounds of adjacent swaps (odd-even
// transposition sort). In one round the swapping pairs are disjoint, so a car only ever goes side by side with
// the single car it is passing, and every other car is at least one grid gap away. The plan depends only on the
// grid and the predicted order (no randomness), so the race is identical at every speed and on every replay.
//  slots: { driver_id: [grid slot, finish slot] }
const plans = new WeakMap()
export function planRace(slots) {
  let P = plans.get(slots)
  if (P) return P
  const ids = Object.keys(slots)
  const cur = [...ids].sort((a, b) => slots[a][0] - slots[b][0])
  const fin = Object.fromEntries([...ids].sort((a, b) => slots[a][1] - slots[b][1]).map((id, i) => [id, i]))
  const snap = () => Object.fromEntries(cur.map((id, i) => [id, i]))
  const rounds = []
  let before = snap()
  for (let r = 0, idle = 0; idle < 2 && r < 4 * ids.length + 4; r++) {
    const pass = new Set()                                 // cars that gain a place in this round
    for (let k = r % 2; k + 1 < cur.length; k += 2) {
      if (fin[cur[k]] > fin[cur[k + 1]]) { pass.add(cur[k + 1]); [cur[k], cur[k + 1]] = [cur[k + 1], cur[k]] }
    }
    if (!pass.size) { idle++; continue }
    idle = 0; rounds.push({ slot: before, pass }); before = snap()
  }
  rounds.push({ slot: before, pass: new Set() })           // settle: back in line, in the predicted order
  P = { ids, rounds }
  plans.set(slots, P)
  return P
}

// Running slot and lane of every car at progress p: { driver_id: [slot (fractional), lane 0..1] }.
// lane 0 = racing line, 1 = overtaking line. Within a round the passer first pulls out (first 20%), then the two
// cars trade places (20-80%), and it tucks back in at the start of the next round unless it passes again.
const MIX0 = 0.1, MIX1 = 0.9                               // overtakes happen at full pace, between launch and braking
export function raceState(slots, p) {
  const { ids, rounds } = planRace(slots), K = rounds.length
  const t = clamp01((p - MIX0) / (MIX1 - MIX0)) * K, r = Math.min(K - 1, Math.floor(t)), u = t - r
  const now = rounds[r], next = rounds[r + 1] || now, prev = rounds[r - 1]
  const trade = smooth(clamp01((u - 0.2) / 0.6)), side = smooth(clamp01(u / 0.2))
  const out = {}
  for (const id of ids) {
    const l0 = prev?.pass.has(id) ? 1 : 0, l1 = now.pass.has(id) ? 1 : 0
    out[id] = [now.slot[id] + (next.slot[id] - now.slot[id]) * trade, l0 + (l1 - l0) * side]
  }
  return out
}

// ---- start sequence (first 2 s of the 10 s): stationary grid, five red lights on one by one, lights out ----
// Real gaps are ~1 s per light plus a random hold; here they are compressed and fixed so every race starts the same.
export const LIGHT_ON = [0.03, 0.06, 0.09, 0.12, 0.15]    // progress at which light k comes on
export const LIGHTS_OUT = 0.2                              // all five go out: the race starts
export const lightsAt = p => (p >= LIGHTS_OUT ? 0 : LIGHT_ON.filter(t => p >= t).length)
export const raceQ = p => clamp01((p - LIGHTS_OUT) / (1 - LIGHTS_OUT))   // race progress after lights out

// Illustrative running order at progress p: the grid until lights out, exactly the predicted order before the flag.
export function orderAt(slots, p) {
  const st = raceState(slots, raceQ(p))
  return Object.keys(st).sort((a, b) => st[a][0] - st[b][0] || slots[a][1] - slots[b][1])
}

// Staggered F1 grid: odd and even slots on opposite sides of the track. Pole is on the LEFT of the driving direction
// (a visual assumption: the data has no pole-side information). gridSide = -1 left, +1 right.
export const gridSide = slot => (slot % 2 ? 1 : -1)
const MERGE = 0.08                                          // race progress over which the two files merge into one

// Race state at progress p in [0, 1] (p = time / RACE_SECONDS). Returns { driver_id: [distance, lane, side] }:
// distance along the lap, overtaking lane 0..1, and the remaining grid-file offset side in [-1, 1] (fades to 0 after
// the start). Cars sit still on their grid slots until lights out, launch, lap once at a corner-aware pace and brake
// to a stop just past the line. gap = distance between consecutive slots (must exceed the drawn car length).
export function replayAt(T, slots, p, gap = carGap(T)) {
  const q = raceQ(p)
  const e = 0.08   // pace ramps 0 -> 1 over the first e of the race and 1 -> 0 over the last e; A = its integral
  const A = q < e ? q * q / (2 * e) : q > 1 - e ? 1 - e - (1 - q) ** 2 / (2 * e) : q - e / 2
  const base = distAtTime(T, T.lap * A / (1 - e))                    // q = 1 -> exactly one lap
  const st = raceState(slots, q), out = {}, file = 1 - smooth(clamp01(q / MERGE))
  for (const id in st) out[id] = [base - (st[id][0] + 0.5) * gap + q * gap, st[id][1], gridSide(slots[id][0]) * file]
  return out
}

// ---- track crossovers (figure-of-eight layouts) ----
// A 30 m terrain model cannot see a bridge, so both branches of a crossover get the same ground height.
// BRIDGES names, per circuit, which pass through the crossing (in lap order) is the upper one. Suzuka: the back
// straight (second pass) crosses over the Degner-to-hairpin section. The clearance is NOT a surveyed figure:
// it is an explicit visual approximation (about one bridge height), applied only on a short ramp either side.
export const BRIDGES = { suzuka: { over: 'second', clearance_m: 6, ramp_m: 220 } }

// Self-crossings of a closed centreline: [{ i, j, x, z }] where segment i crosses segment j (i < j).
export function crossings(xs, zs) {
  const n = xs.length, out = []
  for (let i = 0; i < n; i++) for (let j = i + 2; j < n; j++) {
    if (i === 0 && j === n - 1) continue
    const i2 = (i + 1) % n, j2 = (j + 1) % n
    const rx = xs[i2] - xs[i], rz = zs[i2] - zs[i], sx = xs[j2] - xs[j], sz = zs[j2] - zs[j], den = rx * sz - rz * sx
    if (Math.abs(den) < 1e-9) continue
    const t = ((xs[j] - xs[i]) * sz - (zs[j] - zs[i]) * sx) / den, u = ((xs[j] - xs[i]) * rz - (zs[j] - zs[i]) * rx) / den
    if (t >= 0 && t < 1 && u >= 0 && u < 1) out.push({ i, j, x: xs[i] + rx * t, z: zs[i] + rz * t })
  }
  return out
}

// Height to add per sample so the upper branch clears the lower one by `clearance` at every crossing.
// Returns { lift: number[], at: crossings }. Samples further than `ramp` (along the track) are untouched.
export function bridgeLift(xs, zs, ys, { over = 'second', clearance, ramp = 220 }) {
  const n = xs.length, lift = new Array(n).fill(0), at = crossings(xs, zs)
  const cum = [0]
  for (let i = 0; i < n; i++) cum.push(cum[i] + Math.hypot(xs[(i + 1) % n] - xs[i], zs[(i + 1) % n] - zs[i]))
  for (const c of at) {
    const up = over === 'second' ? c.j : c.i, low = over === 'second' ? c.i : c.j
    const need = clearance - (ys[up] - ys[low])
    if (need <= 0) continue
    for (let k = 0; k < n; k++) {
      let d = Math.abs(cum[k] - cum[up]); d = Math.min(d, cum[n] - d)
      if (d < ramp) lift[k] = Math.max(lift[k], need * 0.5 * (1 + Math.cos(Math.PI * d / ramp)))
    }
  }
  return { lift, at }
}
