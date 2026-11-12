// Lineup rules for the editor (pure functions; unit-tested in e2e/lineup.test.mjs).
// An entry is a driver in their form season driving one car: { driver_id, season, constructor_id, constructor_season }.
// The app's cars are one peak car-season per constructor (backend peak_cars), so a car = a constructor here; the user
// never picks a season. No driver twice, at most two per constructor. The array order IS the grid.
// A race needs exactly 22 drivers in 11 constructors of two (D-036); incomplete grids are fine while editing.
export const MAX_PER_CAR = 2
export const MIN_CARS = 1
export const GRID_SIZE = 22
export const GRID_TEAMS = 11
export const searchText = text => String(text).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase()

export const carKey = (id, season) => `${id}:${season}`
const keyOf = e => carKey(e.constructor_id, e.constructor_season)
const seats = (es, key) => es.filter(e => keyOf(e) === key).length
const last = n => n.split(' ').slice(-1)[0]
const nameOf = d => (d ? last(`${d.first_name} ${d.last_name}`) : '')
export const carLabel = c => (c ? c.constructor_name : '')

// Put driver d on grid slot i. Returns { entries, note } (entries unchanged when nothing can be done).
//  on the grid already: 'insert' moves them there (others shift), 'swap' trades places with the occupant
//  from the pool: replaces the occupant and takes over that car; on an empty slot they join their own team if it has a
//  free seat on the grid, else a grid team with a free seat, else their own team, else the first team with a free seat
export function placeDriver(es, d, i, { mode = 'insert', limit = GRID_SIZE, cars = [], byId = {} } = {}) {
  const from = es.findIndex(e => e.driver_id === d.driver_id)
  if (from >= 0) {
    const to = Math.min(i, es.length - 1)
    if (to === from) return { entries: es }
    const out = [...es]
    if (mode === 'swap') { [out[from], out[to]] = [out[to], out[from]]; return { entries: out, note: `${nameOf(d)} and ${nameOf(byId[es[to].driver_id])} swap grid slots.` } }
    out.splice(to, 0, out.splice(from, 1)[0])
    return { entries: out, note: `${nameOf(d)} moves to P${to + 1}.` }
  }
  if (i < es.length) {
    const out = [...es]
    out[i] = { ...es[i], driver_id: d.driver_id, season: d.season }
    return { entries: out, note: `${nameOf(d)} replaces ${nameOf(byId[es[i].driver_id])} at P${i + 1} in the same car.` }
  }
  if (es.length >= limit) return { entries: es, note: `The grid is full (${limit} cars). Drop the driver on a car to replace it.` }
  const free = c => seats(es, carKey(c.constructor_id, c.season)) < MAX_PER_CAR
  const own = cars.find(c => c.constructor_id === d.constructor_id)
  const onGrid = c => es.some(e => e.constructor_id === c.constructor_id)
  const car = (own && free(own) && onGrid(own) ? own : null) ?? cars.find(c => onGrid(c) && free(c)) ?? (own && free(own) ? own : null) ?? cars.find(free)
  if (!car) return { entries: es, note: 'No car with a free seat.' }
  return { entries: [...es, { driver_id: d.driver_id, season: d.season, constructor_id: car.constructor_id, constructor_season: car.season }],
    note: `${nameOf(d)} joins the grid at P${es.length + 1} in the ${carLabel(car)}.` }
}

// Give a grid driver another car-season. The driver (and their form season) stays the same; only the car, and with it the
// car-performance features sent to the models, changes. A full car hands over the seat of its lower-starting driver, who
// takes the car that was given up, so no car-season is entered more than twice. The grid order never changes.
export function assignCar(es, driverId, car, { byId = {} } = {}) {
  const i = es.findIndex(e => e.driver_id === driverId), k = carKey(car.constructor_id, car.season)
  if (i < 0 || keyOf(es[i]) === k) return { entries: es }
  const out = es.map(e => ({ ...e })), old = { constructor_id: es[i].constructor_id, constructor_season: es[i].constructor_season }
  Object.assign(out[i], { constructor_id: car.constructor_id, constructor_season: car.season })
  const who = nameOf(byId[driverId])
  if (seats(es, k) < MAX_PER_CAR) return { entries: out, note: `${who} now drives the ${carLabel(car)}.` }
  let j = -1
  es.forEach((e, n) => { if (keyOf(e) === k) j = n })
  Object.assign(out[j], old)
  return { entries: out, note: `Seat swap: ${who} takes the ${carLabel(car)}; ${nameOf(byId[es[j].driver_id])} moves to ${who}’s old car.` }
}

// What still stops a prediction ([] = ready): 22 drivers, 11 constructors, two drivers each.
export function gridProblems(es, cars = []) {
  const name = id => cars.find(c => c.constructor_id === id)?.constructor_name ?? id
  const n = {}
  for (const e of es) n[e.constructor_id] = (n[e.constructor_id] || 0) + 1
  const teams = Object.keys(n).length, odd = Object.entries(n).filter(([, k]) => k !== MAX_PER_CAR)
  const out = []
  const ids = es.map(e => e.driver_id)
  if (new Set(ids).size !== ids.length) out.push('Each driver can appear only once. Replace duplicate drivers.')
  if (cars.length && es.some(e => !cars.some(c => c.constructor_id === e.constructor_id && c.season === e.constructor_season))) {
    out.push('Assign an eligible constructor’s automatic peak car to every driver.')
  }
  if (es.length !== GRID_SIZE) out.push(es.length < GRID_SIZE ? `Add ${GRID_SIZE - es.length} more driver${GRID_SIZE - es.length > 1 ? 's' : ''} (${es.length}/${GRID_SIZE}).` : `Remove ${es.length - GRID_SIZE} driver(s) (${es.length}/${GRID_SIZE}).`)
  if (teams !== GRID_TEAMS) out.push(`Use exactly ${GRID_TEAMS} constructors (${teams} on the grid).`)
  if (odd.length) out.push(`Each constructor needs ${MAX_PER_CAR} drivers: ${odd.map(([id, k]) => `${name(id)} has ${k}`).join(', ')}.`)
  return out
}

// A–Z by displayed name, accent- and case-insensitive (é sorts with e).
export const byName = (a, b) => a.localeCompare(b, 'en', { sensitivity: 'base' })

export function removeDriver(es, driverId) {
  if (es.length <= MIN_CARS) return { entries: es, note: `A race needs at least ${MIN_CARS} cars.` }
  return { entries: es.filter(e => e.driver_id !== driverId) }
}

export const sameLineup = (a, b) => a.length === b.length && a.every((e, i) => e.driver_id === b[i].driver_id && e.season === b[i].season && keyOf(e) === keyOf(b[i]))

// Undo / redo over lineups: { past, now, future }.
export const history = {
  start: now => ({ past: [], now, future: [] }),
  push: (h, next) => (sameLineup(next, h.now) ? h : { past: [...h.past, h.now].slice(-100), now: next, future: [] }),
  undo: h => (h.past.length ? { past: h.past.slice(0, -1), now: h.past[h.past.length - 1], future: [h.now, ...h.future] } : h),
  redo: h => (h.future.length ? { past: [...h.past, h.now], now: h.future[0], future: h.future.slice(1) } : h),
}

// What the backend receives: exactly the grid shown, in order.
export const payloadEntries = es => es.map(({ driver_id, season, constructor_id, constructor_season }) => ({ driver_id, season, constructor_id, constructor_season }))

// The logo variant of a manifest entry whose seasons cover this car season (the car's, never the driver's); the
// most era-specific variant wins; none -> null (badge).
const span = v => (v.seasons ? v.seasons[1] - v.seasons[0] : 999)
export const pickLogo = (entry, season) =>
  (entry?.variants ?? (entry ? [entry] : [])).filter(v => !v.seasons || (season >= v.seasons[0] && season <= v.seasons[1]))
    .sort((a, b) => span(a) - span(b))[0] ?? null
