// Unit tests for the lineup editor rules (node --test e2e/lineup.test.mjs).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { assignCar, byName, carKey, GRID_SIZE, GRID_TEAMS, gridProblems, history, payloadEntries, placeDriver, removeDriver, sameLineup, searchText } from '../src/lineup.js'

test('driver search matches accented names and either Unicode composition', () => {
  for (const [name, query] of [['Antônio Pizzonia', 'Antonio'], ['Héctor Rebaque', 'Hector'], ['Éric Bernard', 'eric'], ['Luis Pérez-Sala', 'perez-sala'], ['Maurício Gugelmin', 'mauricio']]) {
    assert.ok(searchText(name).includes(searchText(query)))
    assert.equal(searchText(name), searchText(name.normalize('NFD')))
  }
})

const CARS = [['mclaren', 2025], ['ferrari', 2025], ['red_bull', 2025], ['mercedes', 2020], ['mclaren', 1988], ['ferrari', 2004]]
  .map(([constructor_id, season]) => ({ constructor_id, season, constructor_name: constructor_id }))
const drv = (driver_id, constructor_id, season = 2025) => ({ driver_id, constructor_id, season, first_name: 'X', last_name: driver_id })
const D = Object.fromEntries([drv('norris', 'mclaren'), drv('piastri', 'mclaren'), drv('leclerc', 'ferrari'), drv('hamilton', 'ferrari'),
  drv('max_verstappen', 'red_bull'), drv('tsunoda', 'red_bull'), drv('senna', 'mclaren', 1988), drv('michael_schumacher', 'ferrari', 2004),
  drv('nobody', 'minardi', 1999)].map(d => [d.driver_id, d]))
const entry = id => ({ driver_id: id, season: D[id].season, constructor_id: D[id].constructor_id, constructor_season: D[id].season })
const grid = ['norris', 'piastri', 'leclerc', 'hamilton', 'max_verstappen'].map(entry)
const ids = es => es.map(e => e.driver_id)
const cars = es => es.map(e => carKey(e.constructor_id, e.constructor_season))
const opt = { cars: CARS, byId: D, limit: 8 }
const valid = es => {
  assert.equal(new Set(ids(es)).size, es.length, 'no driver twice')
  const n = cars(es).reduce((m, k) => ({ ...m, [k]: (m[k] || 0) + 1 }), {})
  assert.ok(Object.values(n).every(v => v <= 2), `at most two entries per car-season ${JSON.stringify(n)}`)
}

test('grid driver: insert reorders, swap trades places', () => {
  assert.deepEqual(ids(placeDriver(grid, D.hamilton, 0, opt).entries), ['hamilton', 'norris', 'piastri', 'leclerc', 'max_verstappen'])
  assert.deepEqual(ids(placeDriver(grid, D.hamilton, 0, { ...opt, mode: 'swap' }).entries), ['hamilton', 'piastri', 'leclerc', 'norris', 'max_verstappen'])
  assert.deepEqual(ids(placeDriver(grid, D.norris, 9, opt).entries), ['piastri', 'leclerc', 'hamilton', 'max_verstappen', 'norris'])
  assert.equal(placeDriver(grid, D.norris, 0, opt).entries, grid, 'same slot: unchanged')
})

test('pool driver: joins their own team when it has a free seat on the grid, else a grid team with a seat; replaces in the occupant’s car', () => {
  const PEAK = [['mclaren', 1988], ['ferrari', 2002], ['red_bull', 2011], ['mercedes', 2015], ['minardi', 1991]]
    .map(([constructor_id, season]) => ({ constructor_id, season, constructor_name: constructor_id }))
  const g = [{ driver_id: 'norris', season: 2025, constructor_id: 'mclaren', constructor_season: 1988 },
    { driver_id: 'leclerc', season: 2025, constructor_id: 'ferrari', constructor_season: 2002 }]
  const o = { cars: PEAK, byId: D, limit: GRID_SIZE }
  const a = placeDriver(g, D.senna, 2, o).entries          // own team McLaren is on the grid with a free seat
  assert.deepEqual(a[2], { driver_id: 'senna', season: 1988, constructor_id: 'mclaren', constructor_season: 1988 })
  const n = placeDriver(a, D.nobody, 3, o).entries         // own Minardi not on the grid: the grid team with a seat (Ferrari)
  assert.deepEqual(n[3], { driver_id: 'nobody', season: 1999, constructor_id: 'ferrari', constructor_season: 2002 })
  const v = placeDriver(n, D.max_verstappen, 4, o).entries // every grid team full: his own team, in its peak car
  assert.deepEqual(v[4], { driver_id: 'max_verstappen', season: 2025, constructor_id: 'red_bull', constructor_season: 2011 })
  const r = placeDriver(g, D.michael_schumacher, 0, o).entries
  assert.deepEqual(r[0], { driver_id: 'michael_schumacher', season: 2004, constructor_id: 'mclaren', constructor_season: 1988 }, 'keeps the slot’s car')
  const full = Array.from({ length: GRID_SIZE }, (_, i) => ({ driver_id: `d${i}`, season: 2000, constructor_id: `t${i >> 1}`, constructor_season: 2000 }))
  assert.equal(placeDriver(full, D.senna, GRID_SIZE, o).entries, full, 'a full 22-car grid takes no 23rd driver')
})

test('grid rule: exactly 22 drivers, 11 constructors, two each; incomplete grids explain what is missing', () => {
  const ok = Array.from({ length: GRID_SIZE }, (_, i) => ({ driver_id: `d${i}`, season: 2000, constructor_id: `t${i >> 1}`, constructor_season: 2000 }))
  assert.equal(GRID_SIZE, 22); assert.equal(GRID_TEAMS, 11)
  assert.deepEqual(gridProblems(ok), [])
  assert.match(gridProblems(ok.map((e, i) => i === 1 ? { ...e, driver_id: ok[0].driver_id } : e)).join(' '), /only once/)
  const peaks = Array.from({ length: GRID_TEAMS }, (_, i) => ({ constructor_id: `t${i}`, season: 2000 }))
  assert.deepEqual(gridProblems(ok, peaks), [])
  assert.match(gridProblems(ok.map((e, i) => i === 0 ? { ...e, constructor_season: 1999 } : e), peaks).join(' '), /automatic peak/)
  assert.match(gridProblems(ok.slice(0, 21)).join(' '), /Add 1 more driver .*21\/22.*t10 has 1/)
  const three = ok.map((e, i) => (i === 21 ? { ...e, constructor_id: 't0' } : e))           // 22 drivers, t0 has 3, t10 has 1
  assert.match(gridProblems(three).join(' '), /t0 has 3, t10 has 1/)
  const twelve = ok.map((e, i) => (i === 21 ? { ...e, constructor_id: 'extra' } : e))       // 12 constructors
  assert.match(gridProblems(twelve).join(' '), /exactly 11 constructors \(12/)
  assert.ok(gridProblems([]).length >= 2)
})

test('lists sort A–Z by displayed name and search ignores accents and case', () => {
  const names = ['Ayrton Senna', 'Éric Bernard', 'alex Albon', 'Kimi Räikkönen', 'Zsolt Baumgartner', 'Esteban Ocon']
  assert.deepEqual([...names].sort(byName), ['alex Albon', 'Ayrton Senna', 'Éric Bernard', 'Esteban Ocon', 'Kimi Räikkönen', 'Zsolt Baumgartner'])
  assert.ok(searchText('Kimi Räikkönen').includes(searchText('raikkonen')))
  assert.ok(searchText('Éric Bernard').includes(searchText('ERIC')))
})

test('constructor: free car assigns, full car swaps seats; driver, form season and grid order are kept', () => {
  const a = assignCar(grid, 'norris', { constructor_id: 'mercedes', season: 2020 }, { byId: D })
  assert.deepEqual(ids(a.entries), ids(grid)); assert.equal(a.entries[0].constructor_id, 'mercedes'); assert.equal(a.entries[0].constructor_season, 2020)
  assert.equal(a.entries[0].season, 2025, 'driver season unchanged'); valid(a.entries)
  const b = assignCar(grid, 'max_verstappen', { constructor_id: 'mclaren', season: 2025 }, { byId: D })
  assert.deepEqual(ids(b.entries), ids(grid))
  assert.equal(cars(b.entries)[4], 'mclaren:2025'); assert.equal(cars(b.entries)[1], 'red_bull:2025', 'lower-starting McLaren driver takes the freed car')
  assert.match(b.note, /Seat swap/); valid(b.entries)
  const c = assignCar(grid, 'norris', { constructor_id: 'mclaren', season: 1988 }, { byId: D })
  assert.equal(cars(c.entries)[0], 'mclaren:1988', 'same team, other season = another car'); valid(c.entries)
  assert.equal(assignCar(grid, 'norris', { constructor_id: 'mclaren', season: 2025 }).entries, grid, 'same car: unchanged')
  assert.equal(grid[4].constructor_id, 'red_bull', 'input not mutated')
})

test('remove: a grid may be incomplete while editing, but never empty', () => {
  let es = grid
  for (const id of ['norris', 'piastri', 'leclerc', 'hamilton']) es = removeDriver(es, id).entries
  assert.equal(es.length, 1)
  assert.equal(removeDriver(es, 'max_verstappen').entries, es)
})

test('undo / redo / reset walk the edit history', () => {
  let h = history.start(grid)
  const a = placeDriver(grid, D.hamilton, 0, opt).entries, b = assignCar(a, 'hamilton', { constructor_id: 'ferrari', season: 2004 }, { byId: D }).entries
  h = history.push(history.push(h, a), b)
  assert.equal(history.push(h, b.map(e => ({ ...e }))), h, 'an identical lineup is not a new step')
  h = history.undo(h); assert.ok(sameLineup(h.now, a))
  h = history.undo(h); assert.ok(sameLineup(h.now, grid)); assert.equal(history.undo(h), h)
  h = history.redo(history.redo(h)); assert.ok(sameLineup(h.now, b)); assert.equal(history.redo(h), h)
  assert.ok(!sameLineup(a, b), 'a different car-season is a different lineup')
  h = history.push(h, grid); assert.ok(sameLineup(h.now, grid), 'reset is one more undoable step')
  h = history.push(history.undo(history.undo(h)), grid.slice(0, 4)); assert.equal(h.future.length, 0, 'a new edit clears redo')
})

test('the backend payload is exactly the grid shown, in order, with both seasons', () => {
  const es = assignCar(placeDriver(grid, D.senna, 0, opt).entries, 'senna', { constructor_id: 'mercedes', season: 2020 }, { byId: D }).entries
  const p = payloadEntries(es.map(e => ({ ...e, extra: 1 })))
  assert.deepEqual(p[0], { driver_id: 'senna', season: 1988, constructor_id: 'mercedes', constructor_season: 2020 })
  assert.deepEqual(Object.keys(p[1]), ['driver_id', 'season', 'constructor_id', 'constructor_season'])
})
