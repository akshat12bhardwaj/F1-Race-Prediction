// Unit tests for track geometry, elevation data and the illustrative replay (node --test e2e/motion.test.mjs).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { advance, bridgeLift, BRIDGES, buildTrack, carGap, crossings, distAtTime, frameAt, gridSide, LAP_SECONDS, lightsAt, LIGHTS_OUT, orderAt, planRace, RACE_SECONDS, raceQ, raceState, replayAt } from '../src/motion.js'
import { CAR_LENGTH } from '../src/carSize.js'

const data = JSON.parse(readFileSync(new URL('../public/circuits.json', import.meta.url)))
const C = data.circuits
const track = id => buildTrack(C[id].x, C[id].z, C[id].y || null)
const monaco = track('monaco'), monza = track('monza'), spa = track('spa')

test('every circuit with elevation has a complete, finite, smooth DEM profile and terrain grid', () => {
  const withY = Object.entries(C).filter(([, c]) => c.y)
  assert.ok(withY.length >= 20, `${withY.length} circuits with measured elevation`)
  for (const [id, c] of withY) {
    assert.equal(c.y.length, c.x.length, id)
    assert.ok(c.y.every(Number.isFinite) && Math.min(...c.y) === 0, `${id}: finite, relative to the lowest point`)
    const step = c.traced_m / c.x.length
    for (let i = 0; i < c.y.length; i++) {
      const dy = Math.abs(c.y[(i + 1) % c.y.length] - c.y[i])
      assert.ok(dy / step < 0.25, `${id}: no spike at sample ${i} (${(100 * dy / step).toFixed(0)}% grade)`)
    }
    assert.equal(c.elevation.quality, 'measured', id)
    assert.ok(c.elevation.max_grade <= 0.2, `${id}: steepest 100 m grade ${c.elevation.max_grade}`)
    assert.equal(c.terrain.h.length, c.terrain.n ** 2, id)
    assert.ok(c.terrain.h.every(Number.isFinite), id)
  }
  for (const [id, c] of Object.entries(C).filter(([, c]) => !c.y)) assert.equal(c.elevation.quality, 'approximate', `${id} is flat only when labelled`)
})

test('known profiles: Spa is hilly, Monza is nearly flat', () => {
  assert.ok(C.spa.y && C.spa.elevation.range_m > 70 && C.spa.elevation.range_m < 140, `Spa range ${C.spa.elevation.range_m} m`)
  assert.ok(C.monza.elevation.range_m < 25, `Monza range ${C.monza.elevation.range_m} m`)
})

test('cars sit on the road surface and pitch with the slope', () => {
  for (let i = 0; i < spa.n; i += 7) {
    const [x, y, z] = frameAt(spa, spa.cum[i])
    assert.ok(Math.abs(x - spa.xs[i]) < 1e-6 && Math.abs(z - spa.zs[i]) < 1e-6 && Math.abs(y - spa.ys[i]) < 1e-6, 'exact at samples')
  }
  const i = 40, mid = (spa.cum[i] + spa.cum[i + 1]) / 2, [, y] = frameAt(spa, mid)
  assert.ok(Math.abs(y - (spa.ys[i] + spa.ys[i + 1]) / 2) < 1e-6, 'linear between samples, like the road mesh')
  let up = 0, down = 0
  for (let k = 1; k < spa.n - 1; k++) {
    const [, , , , pitch] = frameAt(spa, spa.cum[k])
    const slope = spa.ys[k + 1] - spa.ys[k - 1]
    if (Math.abs(slope) > 2) { assert.equal(Math.sign(pitch), Math.sign(slope), `sample ${k}`); slope > 0 ? up++ : down++ }
    assert.ok(Math.abs(pitch) < 0.25, 'pitch stays physical (< 14 deg)')
  }
  assert.ok(up > 5 && down > 5, `Spa has climbs (${up}) and descents (${down})`)
  const [, , , , flat] = frameAt(buildTrack(C.spa.x, C.spa.z), 1000)
  assert.equal(flat, 0, 'no elevation -> level cars')
})

test('heading turns smoothly through corners (no flips between samples)', () => {
  let prev = frameAt(monaco, 0)[3]
  for (let d = 2; d < monaco.L; d += 2) {
    const h = frameAt(monaco, d)[3]
    let dh = Math.abs(h - prev); if (dh > Math.PI) dh = 2 * Math.PI - dh
    assert.ok(dh < 0.5, `heading jump ${dh.toFixed(3)} rad at ${d} m`)
    prev = h
  }
})

test('pace is corner-aware: twisty Monaco takes longer per lap than Monza', () => {
  assert.ok(monaco.lap > monza.lap && monza.lap >= LAP_SECONDS)
  let prev = 0
  for (let t = 0; t <= 2 * monza.lap; t += 0.25) { const d = distAtTime(monza, t); assert.ok(d >= prev); prev = d }
  assert.ok(Math.abs(distAtTime(monza, monza.lap) - monza.L) < 1e-6)
})

const grid = ['a', 'b', 'c', 'd', 'e', 'f']
const finish = ['d', 'a', 'f', 'b', 'c', 'e']
const slots = Object.fromEntries(grid.map((id, i) => [id, [i, finish.indexOf(id)]]))
const byDistance = (T, p) => Object.entries(replayAt(T, slots, p)).sort((x, y) => y[1][0] - x[1][0]).map(x => x[0])

test('race starts on the grid and ends in exactly the predicted order', () => {
  for (const T of [monza, spa]) {
    const s = replayAt(T, slots, 0)
    assert.deepEqual(byDistance(T, 0), grid)
    assert.ok(Object.values(s).every(([d, lane]) => d < 0 && lane === 0), 'behind the line, in lane')
    assert.deepEqual(replayAt(T, slots, LIGHTS_OUT * 0.99), s, 'stationary until the lights go out')
    const e = replayAt(T, slots, 1)
    assert.deepEqual(byDistance(T, 1), finish)
    assert.deepEqual(orderAt(slots, 1), finish, 'running order at the flag is the prediction')
    assert.deepEqual(orderAt(slots, 0), grid)
    assert.ok(Object.values(e).every(([, lane, side]) => Math.abs(lane) < 1e-9 && side === 0))
    const lead = e[finish[0]][0]
    assert.ok(lead > T.L && lead < T.L + carGap(T), 'one lap: winner just past the line')
    for (let k = 1; k < finish.length; k++) assert.ok(Math.abs(e[finish[k - 1]][0] - e[finish[k]][0] - carGap(T)) < 1e-6, 'even gaps')
  }
})

test('race is smooth: cars never reverse or jump', () => {
  let prev = replayAt(spa, slots, 0)
  for (let p = 0.001; p <= 1; p += 0.001) {
    const cur = replayAt(spa, slots, p)
    for (const id of grid) {
      const dd = cur[id][0] - prev[id][0]
      assert.ok(dd >= -1e-6 && dd < 0.02 * spa.L, `${id} at p=${p.toFixed(3)}: step ${dd.toFixed(1)} m`)
    }
    prev = cur
  }
})

test('the whole sequence (lights + race) lasts 10 seconds at any frame rate', () => {
  assert.equal(RACE_SECONDS, 10)
  const run = fps => { let p = 0, t = 0; while (p < 1) { p = advance(p, 1 / fps); t += 1 / fps } return t }
  for (const fps of [10, 24, 60, 144]) assert.ok(Math.abs(run(fps) - 10) <= 1 / fps + 1e-6, `${fps} fps -> ${run(fps).toFixed(2)} s`)
  assert.ok(advance(0, 30) <= 0.025 + 1e-9, 'a stalled tab does not teleport the field')
  assert.equal(advance(0.5, 0), 0.5); assert.equal(advance(0.999, 1), 1)
})

// deterministic shuffle so failures reproduce
function shuffled(n, seed) {
  let s = seed
  const a = [...Array(n).keys()]
  for (let i = n - 1; i > 0; i--) { s = (s * 1103515245 + 12345) % 2147483648; const j = s % (i + 1); [a[i], a[j]] = [a[j], a[i]] }
  return a
}

test('overtake plan: any finishing order is reached, deterministically', () => {
  for (let seed = 1; seed <= 40; seed++) {
    const fin = shuffled(22, seed), sl = Object.fromEntries(fin.map((f, g) => [`d${g}`, [g, f]]))
    const want = Object.keys(sl).sort((a, b) => sl[a][1] - sl[b][1])
    assert.deepEqual(orderAt(sl, 1), want, `seed ${seed}`)
    assert.deepEqual(orderAt(sl, 0.93), want, 'order is settled before the cars brake')
    assert.deepEqual(orderAt(sl, LIGHTS_OUT), Object.keys(sl).sort((a, b) => sl[a][0] - sl[b][0]), 'grid order until lights out')
    assert.ok(planRace(sl).rounds.length <= 22)
    assert.deepEqual(raceState(sl, 0.37), raceState(sl, 0.37), 'no randomness: same progress, same picture')
    for (const [, lane] of Object.values(raceState(sl, 1))) assert.equal(lane, 0)
  }
})

test('cars never overlap: side by side only with a full lane between them (every speed follows this path)', () => {
  // the 3D view draws cars CAR_LENGTH * scale long, slots `gap` apart and lanes LANE apart; here in slot / lane units
  const LEN = 1 / 1.7                                      // car length / slot gap (Track3D keeps gap >= 1.7 lengths)
  for (let seed = 1; seed <= 25; seed++) {
    const fin = shuffled(22, seed), sl = Object.fromEntries(fin.map((f, g) => [`d${g}`, [g, f]]))
    for (let p = 0; p <= 1.0001; p += 0.0005) {            // 2000 samples: finer than any frame step at 0.5x
      const st = Object.values(raceState(sl, p)).sort((a, b) => a[0] - b[0])
      for (let k = 1; k < st.length; k++) {
        const along = st[k][0] - st[k - 1][0], across = Math.abs(st[k][1] - st[k - 1][1])
        assert.ok(along >= LEN * 1.15 || across >= 0.9, `seed ${seed} p=${p.toFixed(4)}: ${along.toFixed(2)} slots apart, ${across.toFixed(2)} lanes`)
      }
    }
  }
  assert.ok(carGap(monaco) > 1.7 * CAR_LENGTH && carGap(spa) > 1.7 * CAR_LENGTH)
})

test('Suzuka crossover: the back straight clears the lower road, the rest of the profile is untouched', () => {
  const only = Object.entries(C).filter(([, c]) => crossings(c.x, c.z).length).map(([id]) => id)
  assert.deepEqual(only, ['suzuka'], 'the only figure-of-eight circuit')
  assert.deepEqual(Object.keys(BRIDGES), ['suzuka'])
  const s = C.suzuka, B = BRIDGES.suzuka, [c] = crossings(s.x, s.z)
  assert.ok(c.i / s.x.length < 0.5 && c.j / s.x.length > 0.7, 'first pass: Degner side; second pass: back straight')
  const { lift } = bridgeLift(s.x, s.z, s.y, { over: B.over, clearance: B.clearance_m, ramp: B.ramp_m })
  const y = s.y.map((v, k) => v + lift[k])
  assert.ok(y[c.j] - y[c.i] >= B.clearance_m - 0.3, `upper road ${(y[c.j] - y[c.i]).toFixed(1)} m above the lower one`)
  assert.equal(lift[c.i], 0, 'the lower road keeps its measured height')
  const touched = lift.filter(v => v > 0).length, step = s.traced_m / s.x.length
  assert.ok(touched * step <= 2 * B.ramp_m + step && touched < s.x.length / 8, `${touched} of ${s.x.length} samples adjusted`)
  for (let k = 0; k < y.length; k++) assert.ok(Math.abs(y[(k + 1) % y.length] - y[k]) / step < 0.12, 'ramp is gentle')
  const T = buildTrack(s.x, s.z, y)
  assert.ok(Math.abs(frameAt(T, T.cum[c.j])[1] - y[c.j]) < 1e-6, 'cars ride the raised road')
})



test('start sequence: five lights on in order, then out; grid is staggered, aligned and overlap-free', () => {
  const seen = []
  for (let p = 0; p <= 0.3; p += 0.001) seen.push(lightsAt(p))
  assert.deepEqual(seen.filter((v, k) => k === 0 || v !== seen[k - 1]), [0, 1, 2, 3, 4, 5, 0], 'off, 1..5 in sequence, all out')
  assert.ok(seen.every((v, k) => k === 0 || v >= seen[k - 1] || v === 0))
  assert.equal(lightsAt(LIGHTS_OUT - 1e-6), 5); assert.equal(lightsAt(LIGHTS_OUT), 0)
  assert.ok(LIGHTS_OUT * RACE_SECONDS <= 2.01, 'the start takes at most 2 of the 10 seconds')
  assert.equal(raceQ(0), 0); assert.equal(raceQ(LIGHTS_OUT), 0); assert.equal(raceQ(1), 1)
  // grid: alternate sides, pole on the left, one slot gap apart along the straight, all facing the driving direction
  const g = replayAt(monza, slots, 0), gap = carGap(monza)
  grid.forEach((id, i) => {
    assert.equal(g[id][2], gridSide(i)); assert.equal(g[id][2], i % 2 ? 1 : -1)
    if (i) assert.ok(Math.abs(g[grid[i - 1]][0] - g[id][0] - gap) < 1e-9, 'one slot behind the car ahead')
  })
  assert.ok(gap > 1.7 * CAR_LENGTH, 'slots are longer than a car: no overlap on the grid')
  const h0 = frameAt(monza, g[grid[0]][0])[3], h5 = frameAt(monza, g[grid[5]][0])[3]
  assert.ok(Math.abs(h0 - h5) < 0.05, 'the grid lies on the start straight (all cars share its heading)')
  // after the start the two files merge into one before any overtaking begins
  const late = replayAt(monza, slots, LIGHTS_OUT + 0.09 * (1 - LIGHTS_OUT))
  assert.ok(Object.values(late).every(([, lane, side]) => side === 0 && lane === 0))
})
