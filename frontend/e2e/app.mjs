// Browser test of the APEX F1 race predictor: historical lineup editor, cross-era prediction, F1 start (staggered grid +
// lights), live timing tower, click-to-pause, free camera, header circuit picker.
// Prereq: npm run build; python -m uvicorn src.api:app --port 8011.   Run: node e2e/app.mjs
import { chromium } from 'playwright-core'
import { mkdirSync } from 'node:fs'

const BASE = process.env.E2E_URL || 'http://127.0.0.1:8011'
const SHOTS = new URL('../../outputs/figures/ui/', import.meta.url)
mkdirSync(SHOTS, { recursive: true })
let failures = 0
const check = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`); if (!cond) failures++ }

const browser = await chromium.launch({ channel: process.env.E2E_BROWSER === 'chrome' ? 'chrome' : 'msedge', headless: true,
  args: process.env.E2E_HARDWARE === '1' ? ['--ignore-gpu-blocklist'] : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] })
const page = await browser.newPage({ viewport: { width: 1600, height: 960 } })
const errors = [], posts = []
page.on('console', m => { if (m.type() === 'error' && !m.text().startsWith('Failed to load resource')) errors.push(m.text()) })
page.on('pageerror', e => errors.push(String(e)))
page.on('response', r => { if (r.status() >= 400 && (r.url().includes('/api/') || /\/(drivers|constructors)\//.test(r.url()))) errors.push(`HTTP ${r.status()} ${r.url()}`) })
page.on('requestfinished', async r => {
  if (r.url().endsWith('/api/race-prediction')) posts.push({ sent: r.postDataJSON(), got: await (await r.response()).json() })
})
const shot = (name, full = false) => page.screenshot({ path: new URL(`${name}.png`, SHOTS).pathname.replace(/^\/(\w:)/, '$1'), fullPage: full })
const V = '[data-viewer="3d"]'
const ds = k => page.locator(V).getAttribute(`data-${k}`)
const apex = (fn, arg) => page.locator(V).evaluate((el, [f, a]) => el.__apex[f](a), [fn, arg])
const wait = ms => page.waitForTimeout(ms)
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b)
const P = async () => Number(await ds('p'))
const summary = () => page.locator('.grid-sum li').evaluateAll(els => els.map(e => [e.dataset.driver, e.dataset.team, Number(e.dataset.carSeason)]))
const grid = () => page.locator('.slots .slot').evaluateAll(els => els.filter(e => e.dataset.driver).map(e => [e.dataset.driver, e.dataset.team, Number(e.dataset.season)]))
const ids = async () => (await grid()).map(x => x[0])
const tower = () => page.locator('.tower li').evaluateAll(els => els.map(e => [e.dataset.driver, Number(e.dataset.pos)]).sort((a, b) => a[1] - b[1]).map(x => x[0]))
const ranks = async () => Object.entries(await apex('cars')).sort((a, b) => a[1].rank - b[1].rank).map(x => x[0])
const predictBtn = '.run .btn.lime'
async function drag(from, to) {
  await page.locator(to).scrollIntoViewIfNeeded(); await page.locator(from).scrollIntoViewIfNeeded()
  const a = await page.locator(from).boundingBox(), b = await page.locator(to).boundingBox()
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2); await page.mouse.down()
  await page.mouse.move(a.x + a.width / 2 + 9, a.y + a.height / 2 + 9, { steps: 3 })
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 14 }); await wait(90)
  await page.mouse.up(); await wait(160)
}
const handle = id => `.slot[data-driver="${id}"] .drag`
const slot = i => `.slot[data-slot="${i}"]`
async function predict() {            // click Predict and return when the race has started
  const n = posts.length
  await page.click(predictBtn)
  await page.waitForSelector('.board.racing', { timeout: 60000 })
  for (let k = 0; k < 40 && posts.length === n; k++) await wait(50)
  return posts[posts.length - 1]
}
// an empty spot of the canvas (no car within 60 px) to click on
const emptySpot = () => page.locator(V).evaluate(el => {
  const r = el.getBoundingClientRect(), cars = Object.keys(el.__apex.cars()).map(id => el.__apex.screen(id))
  for (let y = r.top + r.height * 0.75; y > r.top + 130; y -= 23) for (let x = r.left + 40; x < r.right - 40; x += 37)
    if (cars.every(([cx, cy]) => Math.hypot(cx - x, cy - y) > 60)) return [x, y]
  return null
})

// ---------- 1. shell, header circuit picker ----------
await page.goto(BASE)
await page.waitForFunction(() => document.querySelectorAll('.grid-sum li').length === 22, null, { timeout: 30000 })
const body = await page.locator('body').innerText()
check((await page.title()).startsWith('APEX') && (await page.locator('.logo-word').innerText()) === 'APEX', 'APEX logo and wordmark in the top bar')
check(!/2026 grid|Legends|replay a past|model report|upload|dataset/i.test(body), 'F1-only: no Legends mode, historical replay, model report or dataset upload')
await page.waitForSelector(`${V} canvas`, { timeout: 30000 })
const vp = page.viewportSize(), st = await page.locator('.stage').boundingBox()
check(st.width / vp.width > 0.5 && st.height / vp.height > 0.7, `3D stage dominates the screen (${Math.round(100 * st.width / vp.width)}% × ${Math.round(100 * st.height / vp.height)}%)`)
check(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight + 1), 'whole dashboard fits one desktop screen')
const bg = await page.evaluate(() => ({ lime: getComputedStyle(document.documentElement).getPropertyValue('--lime').trim(), btn: getComputedStyle(document.querySelector('.run .btn.lime')).backgroundColor }))
check(bg.lime === '#d2f03c' && bg.btn === 'rgb(210, 240, 60)', 'dark charcoal + neon-lime theme kept')
check(await page.locator('.side-left select[aria-label="Circuit"], .side-left .cpick').count() === 0 && await page.locator('.topbar .cpick-btn').count() === 1,
  'circuit selector moved from the sidebar to the top navigation')
check(await page.locator('.side-left select').count() === 2 && await page.locator('select[aria-label="Classification model"] option').count() === 10 && await page.locator('select[aria-label="Regression model"] option').count() === 6,
  'sidebar: exactly one classification (10) and one regression (6) picker')
check(/Belgian Grand Prix/.test(await page.locator('.cpick-btn').innerText()), 'picker shows the selected race')
await page.click('.cpick-btn')
const pop = await page.locator('.cpick-pop').boundingBox(), btnBox = await page.locator('.cpick-btn').boundingBox()
check(pop.x >= 0 && pop.x + pop.width <= vp.width && pop.y + pop.height <= vp.height && Math.abs(pop.width - btnBox.width) < 2, `picker menu fits the window and matches the button width (${Math.round(pop.width)} px)`)
const allC = await page.locator('.cpick-pop li[role=option]').count()
await page.fill('.cpick-pop input', 'suzuka')
check(allC >= 39 && await page.locator('.cpick-pop li[role=option]').count() === 1, `circuit search narrows ${allC} circuits to 1`)
await page.keyboard.press('Enter')
await page.waitForSelector(`${V}[data-circuit="suzuka"]`, { timeout: 15000 })
check(await page.locator('.cpick-pop').count() === 0 && /Japanese Grand Prix/.test(await page.locator('.cpick-btn').innerText()), 'keyboard: type + Enter picks Suzuka, updates header and 3D circuit')
await page.click('.cpick-btn'); await page.keyboard.press('Escape')
check(await page.locator('.cpick-pop').count() === 0, 'Escape closes the menu')
await page.click('.cpick-btn'); await page.fill('.cpick-pop input', 'belg'); await page.keyboard.press('ArrowDown'); await page.keyboard.press('ArrowUp'); await page.keyboard.press('Enter')
await page.waitForSelector(`${V}[data-circuit="spa"]`, { timeout: 15000 })
const selects = await page.locator('select').evaluateAll(els => els.map(e => { const r = e.getBoundingClientRect(), p = e.parentElement.getBoundingClientRect(); return [Math.round(r.height), r.right <= p.right + 1] }))
check(selects.every(([h, inside]) => h === selects[0][0] && inside), `dropdowns share one height (${selects[0][0]} px) and stay inside their panels`)

// ---------- 2. 3D camera: orbit, zoom, pan, ground clamp ----------
await page.waitForFunction(v => document.querySelector(v)?.dataset.cam, V); await wait(800)
const c0 = await ds('cam'), vb = await page.locator(V).boundingBox(), cx = vb.x + vb.width / 2, cy = vb.y + vb.height / 2
await page.mouse.move(cx, cy); await page.mouse.down()
for (let k = 1; k <= 10; k++) await page.mouse.move(cx + 12 * k, cy + 3 * k)
await page.mouse.up(); await wait(500)
const c1 = await ds('cam')
check(c1 !== c0, 'left-drag orbits the camera')
const centre = await apex('centre')
const dist = c => Math.hypot(...c.split(',').map((v, i) => Number(v) - centre[i]))
await page.mouse.move(cx, cy); await page.mouse.wheel(0, -600); await wait(600)
check(dist(await ds('cam')) < dist(c1), 'scroll zooms in')
const t0 = await apex('target')
await page.mouse.move(cx, cy); await page.mouse.down({ button: 'right' })
for (let k = 1; k <= 8; k++) await page.mouse.move(cx - 15 * k, cy)
await page.mouse.up({ button: 'right' }); await wait(500)
const t1 = await apex('target')
check(Math.hypot(t1[0] - t0[0], t1[2] - t0[2]) > 1, 'right-drag pans the view')
for (let k = 0; k < 6; k++) { await page.mouse.wheel(0, -900); await wait(120) }
await page.mouse.move(cx, cy - 100); await page.mouse.down()
for (let k = 1; k <= 12; k++) await page.mouse.move(cx, cy - 100 + 25 * k)
await page.mouse.up(); await wait(600)
const clr = await apex('clearance')
check(clr === null || clr > 0, `camera never goes below the terrain or road (clearance ${clr?.toFixed(1)} m)`)
await page.click('button:has-text("Reset view")'); await wait(1300)
check(Math.abs(dist(await ds('cam')) - dist(c0)) / dist(c0) < 0.03, 'Reset view returns to the overview framing')
check(await page.locator('[data-height]').count() === 0 && await page.locator('.view-opts button').filter({ hasText: /^(?:2D|3D|HD|Lite|Height)/ }).count() === 0, 'only the definitive HD 3D experience is exposed')
check(await ds('elevation-scale') === '1', 'terrain elevation stays at true scale')
await page.mouse.move(cx, cy); await page.mouse.wheel(0, 10000); await wait(700)
const limits = await apex('cameraLimits')
check(limits.distance <= limits.max * 1.001, 'zoom-out stops at the circuit framing boundary')
await page.click('button:has-text("Reset view")'); await wait(900)

// ---------- 3. starting grid (before any prediction the cars stand on it) ----------
const boxes = await apex('grid'), sz = await apex('size'), cars0 = await apex('cars'), order0 = (await summary()).map(x => x[0])
check(boxes.length === 22 && boxes.every((b, i) => b.side === (i % 2 ? 1 : -1)), 'staggered grid: pole on the left, then alternating sides (22 grid boxes)')
const onBox = order0.every((id, i) => Math.abs(cars0[id].d - boxes[i].d) < 1e-6 && Math.abs(cars0[id].lat - boxes[i].lat) < 1e-6)
check(onBox, 'every car stands in its own grid box, in starting order')
const heads = boxes.map(b => b.heading), hspread = Math.max(...heads) - Math.min(...heads)
check(order0.every((id, i) => i === 0 || cars0[order0[i - 1]].d - cars0[id].d > sz.length * 1.2), `grid slots are a car length apart (gap ${sz.gap.toFixed(1)} m, car ${sz.length.toFixed(1)} m drawn)`)
let gridMin = Infinity
for (const a of order0) for (const b of order0) if (a < b) gridMin = Math.min(gridMin, Math.hypot(cars0[a].p[0] - cars0[b].p[0], cars0[a].p[2] - cars0[b].p[2]))
check(gridMin > sz.length, `no two cars overlap on the grid (closest ${gridMin.toFixed(1)} m apart, car ${sz.length.toFixed(1)} m long)`)
check(order0.every(id => cars0[id].d < 0), `the whole grid stands behind the start/finish line (heading spread ${hspread.toFixed(2)} rad)`)
const onRoad = await page.locator(V).evaluate((el, ids) => { const A = el.__apex, z = A.size(), c = A.cars(); return Math.max(...ids.map(id => Math.abs(c[id].p[1] - z.lift - A.road(c[id].d)[1]))) }, order0)
check(onRoad < 0.01, `grid cars sit on the road surface (max offset ${onRoad.toFixed(4)})`)

// ---------- 4. lineup editor: 22 drivers, 11 constructors x 2, names only (no seasons), A–Z lists, logos ----------
await page.click('.edit-lineup'); await page.waitForSelector('.editor .slot[data-driver]'); await wait(800)
check(await page.locator('.ed-body > .ed-col').count() === 3, 'Edit Lineup: drivers | grid | constructors')
check(await page.locator('.editor select').count() === 0, 'no era, year or car-season dropdowns anywhere in the editor')
check(await page.locator('.slots .tag.swap').count() === 0 && !/hypothetical/i.test(await page.locator('.slots').innerText()), 'starting-grid cards have no hypothetical badges')
check(await page.locator('.pool-card .num25').count() === 0, 'available drivers have no historical racing-number badges')
check(same(await page.locator('.slots .slot-pos').allTextContents(), Array.from({ length: 22 }, (_, i) => `P${i + 1}`)), 'starting-grid positions P1–P22 remain clearly labelled')
const edText = await page.locator('.editor').innerText()
check(await page.locator('.ed-more').count() === 0 && !/photo credit|wikimedia|licen[cs]e|not eligible|imputed|evidence/i.test(edText), 'no photo-credit panel or technical limitation text in the editor')
const nDrivers = Number(await page.locator('.bench .side-h .count').innerText())
const dNames = await page.locator('.pool-card .who2 > b').evaluateAll(els => els.map(e => e.firstChild.textContent.trim()))
check(nDrivers >= 150 && dNames.length === nDrivers, `the whole eligible driver pool is listed and scrollable: ${dNames.length} of ${nDrivers}`)
const az = xs => xs.every((x, i) => i === 0 || xs[i - 1].localeCompare(x, 'en', { sensitivity: 'base' }) <= 0)
check(az(dNames), `drivers A–Z by displayed name (${dNames[0]} … ${dNames.at(-1)})`)
const cNames = await page.locator('.team-card .who2 > b').evaluateAll(els => els.map(e => e.textContent.trim()))
check(cNames.length === 65 && az(cNames) && cNames.every(n => !/\d{4}/.test(n)), `constructors: one per team, A–Z, no years (${cNames.length})`)
check(await page.locator('.team-card .yr').count() === 0, 'constructor cards carry no season')
for (const [q, want] of [['senna', 'senna'], ['raikkonen', 'raikkonen'], ['Räikkönen', 'raikkonen'], ['PEREZ', 'perez'], ['schumacher', 'michael_schumacher']]) {
  await page.fill('input[aria-label="Search drivers"]', q)
  check(await page.locator(`.pool-card[data-driver="${want}"]`).count() === 1, `driver search "${q}" finds ${want} (accents and case ignored)`)
}
await page.fill('input[aria-label="Search drivers"]', 'senna')
check(/1988 · McLaren/.test(await page.locator('.pool-card[data-driver="senna"]').innerText()) && await page.locator('.pool-card[data-driver="senna"] .who2 > b').innerText() === 'Ayrton Senna',
  'Senna card retains his full name and 1988 form season without a racing-number badge')
for (const q of ['fangio', 'doohan']) { await page.fill('input[aria-label="Search drivers"]', q); check(await page.locator('.pool-card').count() === 0, `${q} is not offered (outside the eligibility rules)`) }
await page.fill('input[aria-label="Search drivers"]', '')
await page.fill('input[aria-label="Search constructors"]', 'merc')
check(same(await page.locator('.team-card').evaluateAll(els => els.map(e => e.dataset.team)), ['mercedes']), 'constructor search by team name')
await page.fill('input[aria-label="Search constructors"]', '')
const mercedesLogo = page.locator('.team-card[data-team="mercedes"] .logo-tile img').first()
await mercedesLogo.scrollIntoViewIfNeeded()
const logoOk = await mercedesLogo.evaluate(async i => { await i.decode(); return i.naturalWidth > 0 })
const marks = await page.locator('.team-card').evaluateAll(els => els.map(e => !!e.querySelector('.logo-tile img, .badge')))
check(logoOk && marks.every(Boolean), `constructor cards show a licensed logo (Mercedes) or the consistent badge fallback (${marks.length} cards)`)
const tiles = await page.locator('.team-card .logo-tile, .team-card .badge').evaluateAll(els => [...new Set(els.map(e => `${e.getBoundingClientRect().width}x${e.getBoundingClientRect().height}`))])
const cards = await page.locator('.team-card, .pool-card').evaluateAll(els => [...new Set(els.map(e => e.getBoundingClientRect().height.toFixed(0)))])
check(tiles.length === 1 && cards.length <= 2, `logos and badges share one size (${tiles[0]}); consistent card heights (${cards.join(', ')} px)`)
const g0 = await grid()
const teams0 = g0.reduce((m, [, c]) => ({ ...m, [c]: (m[c] || 0) + 1 }), {})
check(g0.length === 22 && new Set(g0.map(x => x[0])).size === 22 && Object.keys(teams0).length === 11 && Object.values(teams0).every(n => n === 2),
  'default grid: 22 different drivers, 11 constructors, two each')
check(await page.locator('.ed-grid .tag.ok[data-ready="true"]').count() === 1 && await page.locator('.grid-todo').count() === 0, 'a complete grid is marked ready')
const noHScroll = await page.locator('.editor').evaluate(e => [...e.querySelectorAll('.ed-col, .slots, .ed-list')].every(x => x.scrollWidth <= x.clientWidth + 1))
check(noHScroll, 'the 22-slot grid and both lists fit without horizontal scrolling')
await page.locator('.team-card').first().scrollIntoViewIfNeeded()
await shot('51_editor')
// imagery: each portrait belongs to that driver; gaps fall back to the monogram
const portraitOf = async q => {
  await page.fill('input[aria-label="Search drivers"]', q)
  const card = page.locator('.pool-card').first()
  await card.scrollIntoViewIfNeeded()
  const img = card.locator('img.portrait')
  if (await img.count()) await img.evaluate(i => i.complete ? Promise.resolve() : new Promise(resolve => { i.addEventListener('load', resolve, { once: true }); i.addEventListener('error', resolve, { once: true }) }))
  return card.evaluate(c => ({ id: c.dataset.driver, src: c.querySelector('img.portrait')?.getAttribute('src'), ok: c.querySelector('img.portrait')?.complete && c.querySelector('img.portrait')?.naturalWidth === 256, mono: !!c.querySelector('.portrait.mono'), lazy: c.querySelector('img.portrait')?.loading }))
}
for (const q of ['ayrton', 'alain p', 'mika h', 'nico ros', 'max verstappen', 'Antônio Pizzonia', 'Héctor Rebaque']) {
  const r = await portraitOf(q)
  check(r.src === `/drivers/${r.id}.jpg` && r.ok && r.lazy === 'lazy', `${q}: own portrait, 256 px square, lazy-loaded`)
}
const noPhoto = await portraitOf('baumgartner')
check(noPhoto.mono && !noPhoto.src, 'driver without a usable photo shows the monogram (baumgartner)')
await page.fill('input[aria-label="Search drivers"]', '')

// drag a driver onto P3: he takes over that slot's car
const p3 = g0[2]
await page.fill('input[aria-label="Search drivers"]', 'alesi')
await drag('.pool-card[data-driver="alesi"] .drag', slot(2))
let g = await grid()
check(g[2][0] === 'alesi' && g[2][1] === p3[1] && g.length === 22, `drag a driver onto P3: Alesi replaces ${p3[0]} in the ${p3[1]} car`)
// drag a full constructor onto him: predictable seat swap, grid order kept, both teams keep two
const before = await grid()
await page.fill('input[aria-label="Search constructors"]', 'mercedes')
await drag('.team-card[data-team="mercedes"] .drag', '.slot[data-driver="alesi"]')
const after = await grid()
const count = (gg, c) => gg.filter(x => x[1] === c).length
check(same(after.map(x => x[0]), before.map(x => x[0])) && after[2][1] === 'mercedes' && count(after, 'mercedes') === 2 && count(after, p3[1]) === 2 &&
  /Seat swap/.test(await page.locator('.ed-note').innerText()), `full constructor: seat swap, grid order kept (${await page.locator('.ed-note').innerText()})`)
const peakSeason = (await page.locator('.team-card[data-team="mercedes"]').getAttribute('data-season'))
check(after[2][2] === Number(peakSeason), `the car season is the constructor's peak (${peakSeason}), never chosen by the user`)
await page.click('button:has-text("Undo")'); check(same(await grid(), before), 'Undo reverts the seat swap')
await page.click('button:has-text("Redo")'); check(same(await grid(), after), 'Redo re-applies it')
// click alternative: select a driver + Assign a constructor that is not on the grid -> 12 teams: incomplete, explained
const onGrid = new Set(after.map(x => x[1]))
await page.fill('input[aria-label="Search constructors"]', 'minardi')
check(!onGrid.has('minardi'), 'Minardi starts off the grid')
const victim = after[21][0], mate = after.find((x, i) => i !== 21 && x[1] === after[21][1])[0]
await page.click(handle(victim)); await page.click('.team-card[data-team="minardi"] button')
check((await grid()).find(x => x[0] === victim)[1] === 'minardi', `select ${victim} + Assign: Minardi`)
check(await page.locator('.ed-grid .tag.swap[data-ready="false"]').count() === 1 && /12/.test(await page.locator('.grid-todo').innerText()), `12 constructors flagged: ${await page.locator('.grid-todo').innerText()}`)
await page.click(handle(mate)); await page.click('.team-card[data-team="minardi"] button')
const fixed = await grid(), tf = fixed.reduce((m, [, c]) => ({ ...m, [c]: (m[c] || 0) + 1 }), {})
check(Object.keys(tf).length === 11 && Object.values(tf).every(n => n === 2) && await page.locator('.grid-todo').count() === 0, 'second Minardi seat: back to 11 constructors x 2')
// reorder, remove (incomplete grid blocks the prediction), add back
await drag(handle('alesi'), slot(0))
check((await ids())[0] === 'alesi', 'drag Alesi to pole')
const out = (await ids())[5]
await drag(handle(out), '.bench .ed-list')
check(!(await ids()).includes(out) && (await ids()).length === 21, 'drag a grid driver back to the list removes them (21 left)')
check(/Add 1 more driver/.test(await page.locator('.grid-todo').innerText()), 'the editor says what is missing')
await page.click('button:has-text("Done — Back to Race")')
check(await page.locator(predictBtn).isDisabled() && /Add 1 more driver/.test(await page.locator('.run .warn').innerText()), 'Predict is blocked until the grid is complete')
await page.click('.edit-lineup'); await page.waitForSelector('.editor .slot[data-driver]')
await page.fill('input[aria-label="Search drivers"]', 'mansell')
await drag('.pool-card[data-driver="mansell"] .drag', '.slot.empty >> nth=0')
const full = await grid(), tm = full.reduce((m, [, c]) => ({ ...m, [c]: (m[c] || 0) + 1 }), {})
check(full.length === 22 && new Set(full.map(x => x[0])).size === 22 && Object.values(tm).every(n => n === 2) && Object.keys(tm).length === 11,
  `Mansell joins the empty slot in the team with the free seat (${full.find(x => x[0] === 'mansell')?.[1]})`)
await page.fill('input[aria-label="Search drivers"]', 'senna')
check(await page.locator('.pool-card.on[data-driver="senna"] button').count() === 0, 'a driver already on the grid cannot be added twice')
await shot('52_editor_cross_era')
const edited = await grid()
await page.click('button:has-text("Done — Back to Race")')
check(same((await summary()), edited), 'Done — Back to Race: the dashboard summary shows the edited grid')
check(/^22\/22 cars/.test(await page.locator('[data-count]').first().innerText()) && !(await page.locator(predictBtn).isDisabled()), 'dashboard: 22/22 cars and Predict enabled')

// ---------- 5. predict: start lights, tower, race ----------
check(await page.locator('.playback .seg, [aria-label="Race speed"]').count() === 0 && !/0\.5×|2×|4×/.test(await page.locator('.playback').innerText()), 'no playback speed controls')
// Measure active playback in the browser, excluding inference and deliberate automation pauses.
await page.evaluate(() => {
  const clock = window.__apexReplayClock = { active: 0, last: performance.now(), playing: false, started: false }
  const tick = now => {
    if (clock.playing) clock.active += now - clock.last
    clock.last = now
    clock.playing = !!document.querySelector('.board.racing') && !!document.querySelector('.playback button[aria-label="Pause"]')
    clock.started ||= clock.playing
    if (clock.started && document.querySelector('.board.done')) return
    requestAnimationFrame(tick)
  }
  requestAnimationFrame(tick)
})
const tPredict = Date.now()
const first = await predict()
check(same(first.sent.entries.map(e => [e.driver_id, e.constructor_id, e.constructor_season]), edited), 'backend receives exactly the edited drivers, cars and car seasons')
const sen = first.got.standings.find(r => r.driver_id === 'senna')
const senGrid = edited.findIndex(([id]) => id === 'senna')
check(sen.season === 1988 && sen.constructor_id === edited[senGrid][1] && sen.constructor_season === edited[senGrid][2] && sen.grid === senGrid + 1,
  'backend uses Senna’s own form season with exactly his edited constructor, peak car and starting position')
const lits = await page.evaluate(v => new Promise(res => {          // sample the light panel every frame
  const out = [], f = () => { const l = document.querySelector('.lights'); out.push(l ? Number(l.dataset.lit) : null)
    Number(document.querySelector(v).dataset.p) > 0.3 ? res(out) : requestAnimationFrame(f) }
  f()
}), V)
const seq = lits.filter(x => x !== null).filter((v, k, a) => k === 0 || v !== a[k - 1])
check(seq.slice(-6).join() === '1,2,3,4,5,0' || seq.join().endsWith('2,3,4,5,0'), `start lights: on one by one, then out (${seq.join(' → ')})`)
await page.waitForFunction(v => Number(document.querySelector(v).dataset.p) > 0.3, V)
check(await page.locator('.lights').count() === 0, 'lights panel disappears after the start')
const startCars = await apex('cars')
check(Object.values(startCars).some(c => c.d > -1e9), 'cars launch after lights out')
const pl = await page.locator('.playback').innerText()
check(!/\d+(\.\d)?\s*\/\s*10|remaining|\d+\.\d s|Lap \d/i.test(pl), `no countdown, race timer or lap counter (“${pl.replace(/\s+/g, ' ').slice(0, 90)}”)`)
// tower = cars: pause at several moments and compare the timing tower with the 3D running order
let towerOk = 0, towerChanged = false, prevTower = null
for (let k = 0; k < 5; k++) {
  await wait(900)
  await page.click('.playback button[aria-label="Pause"]'); await wait(250)
  const tw = await tower(), rk = await ranks()
  if (same(tw, rk)) towerOk++
  if (prevTower && !same(prevTower, tw)) towerChanged = true
  prevTower = tw
  await page.click('.playback button[aria-label="Resume"]')
}
check(towerOk === 5, `timing tower order equals the cars' running order at every pause (${towerOk}/5)`)
check(towerChanged, 'timing tower updates as the cars change position')
await page.waitForSelector('.board.done', { timeout: 20000 })
const dur = (Date.now() - tPredict) / 1000
const activeDuration = await page.evaluate(() => window.__apexReplayClock.active / 1000)
check(activeDuration > 9 && activeDuration < 12, `race has ten seconds of active playback (${activeDuration.toFixed(2)} s; ${dur.toFixed(1)} s including inference and automation pauses)`)
const fin = await ranks()
check(same(fin, first.got.standings.map(r => r.driver_id)), '3D cars finish in exactly the predicted order')
const stand = await page.locator('.standings li').evaluateAll(els => els.map(e => e.dataset.driver))
check(same(stand, first.got.standings.map(r => r.driver_id)), 'final standings = backend prediction')
const podIds = await page.locator('#results .pod').evaluateAll(els => els.map(e => e.dataset.driver))
check(same([podIds[1], podIds[0], podIds[2]], first.got.podium) && new Set(podIds).size === 3, `podium = backend podium (${first.got.podium.join(', ')})`)
await shot('53_finish')

// ---------- 6. click-to-pause on the viewport ----------
await page.click('.playback button[aria-label="Restart"]'); await wait(2600)
const spot = await emptySpot()
await page.mouse.click(spot[0], spot[1]); await wait(300)
const pz = await P(), towerP = await tower(), carsP = await apex('cars')
await wait(900)
check(pz > 0 && pz < 1 && (await P()) === pz, `clicking the empty track pauses the race (frozen at ${pz})`)
check(same(await tower(), towerP) && same(await apex('cars'), carsP), 'paused: cars and timing tower stay frozen')
check(await page.locator('.paused-chip').count() === 1 && /Paused/.test(await page.locator('.status').innerText()), 'paused state is shown')
// camera still usable while paused, and a drag is not a pause click
const cp0 = await ds('cam')
await page.mouse.move(spot[0], spot[1]); await page.mouse.down()
for (let k = 1; k <= 8; k++) await page.mouse.move(spot[0] + 14 * k, spot[1])
await page.mouse.up(); await wait(400)
check((await ds('cam')) !== cp0 && (await P()) === pz, 'orbiting while paused moves the camera and does not resume the race')
const spot2 = await emptySpot()
await page.mouse.click(spot2[0], spot2[1]); await wait(600)
check((await P()) > pz, 'clicking again resumes from the same point')
// Restore the full-circuit view after the deliberately unrestricted orbit gesture before screen-space picking.
await page.click('.cam-bar button:has-text("Overview")'); await wait(1000)
// clicking a car selects it instead of toggling playback (checked while paused, so the car stands still)
await page.click('.playback button[aria-label="Pause"]'); await wait(300)
const target = fin[4], xy = await apex('screen', target), pBefore = await P()
await page.mouse.click(xy[0], xy[1]); await wait(400)
check(await page.locator(`.tower li.sel[data-driver="${target}"]`).count() === 1 && (await P()) === pBefore && await page.locator('.paused-chip').count() === 1,
  `clicking a car selects ${target} (highlighted in the tower) and does not toggle playback`)
await page.click('.playback button[aria-label="Resume"]')
// buttons over the canvas do not toggle playback
await page.click('.cam-bar button:has-text("Top")'); await wait(300)
check(await page.locator('.paused-chip').count() === 0, 'camera buttons over the viewport do not pause the race')

// ---------- 7. tower click → follow; manual override; onboard ----------
const pick = fin[6]
await page.click('.playback button[aria-label="Restart"]'); await wait(2300)
await page.click(`.tower li[data-driver="${pick}"] button`); await wait(1300)
const fd = Number(await ds('follow'))
check(await ds('mode') === 'follow' && fd > 2 && fd < 3.5 && await page.locator(`.tower li.sel[data-driver="${pick}"]`).count() === 1,
  `tower click follows ${pick} (${fd} car lengths) and highlights them`)
const vb2 = await page.locator(V).boundingBox()
await page.mouse.move(vb2.x + 200, vb2.y + 300); await page.mouse.down()
for (let k = 1; k <= 8; k++) await page.mouse.move(vb2.x + 200 + 15 * k, vb2.y + 300)
await page.mouse.up(); await wait(600)
// tracking is off: the car drives on while the camera only coasts on its orbit damping
const cam = async () => (await ds('cam')).split(',').map(Number)
const k0 = await cam(), car0 = (await apex('cars'))[pick].p; await wait(1200)
const k1 = await cam(), car1 = (await apex('cars'))[pick].p
const carMoved = Math.hypot(car1[0] - car0[0], car1[2] - car0[2]), camMoved = Math.hypot(k1[0] - k0[0], k1[2] - k0[2])
check(await ds('manual') === '1' && carMoved > 50 && camMoved < 0.25 * carMoved,
  `manual rotate during Follow suspends tracking: car drove ${carMoved.toFixed(0)} m, camera ${camMoved.toFixed(0)} m`)
await page.click('.cam-bar button:has-text("Follow")'); await wait(1200)
check(await ds('manual') === null && Number(await ds('follow')) < 3.5, 'pressing Follow again re-engages tracking')
// Camera checks must have their own playback interval; the preceding gestures may consume the ten-second race.
await page.click('.playback button[aria-label="Restart"]')
await page.click('.cam-bar button:has-text("Onboard")'); await wait(1200)
check(await ds('mode') === 'onboard' && Number(await ds('follow')) > 0.15 && Number(await ds('follow')) < 0.4, `onboard view sits above the cockpit (${await ds('follow')} car lengths)`)
await page.click('.playback button[aria-label="Pause"]'); await wait(800)   // direction smoothing settles
const ob0 = await ds('cam'), obp = await P(); await wait(900)
check((await ds('cam')) === ob0 && (await P()) === obp, 'paused: the onboard camera stops advancing with the race')
await page.click('.playback button[aria-label="Resume"]')
await page.click('.cam-bar button:has-text("Overview")'); await page.click('.playback .skip'); await wait(800)

// ---------- 8. re-run with other models on the same scenario ----------
await page.selectOption('select[aria-label="Classification model"]', 'svm')
check(await page.locator('#results.is-stale').count() === 1, 'changing a model marks the result out of date')
const second = await predict()
check(same(second.sent.entries, first.sent.entries) && second.got.models.classification.key === 'svm' && second.got.models.regression.key === 'ridge', 'same scenario re-predicted with SVM + Ridge')
check(!same(second.got.standings.map(r => r.expected_position), first.got.standings.map(r => r.expected_position)), 'a different classifier gives different predicted positions (real inference)')
await page.click('.playback .skip'); await page.waitForSelector('.board.done')
check(same(await page.locator('.standings li').evaluateAll(els => els.map(e => e.dataset.driver)), second.got.standings.map(r => r.driver_id)), 'new standings shown for the new model pair')
// ---------- 8b. race conditions (D-033) ----------
const conds = await page.locator('.conditions input[type=range]').evaluateAll(els => els.map(e => [e.dataset.cond, Number(e.min), Number(e.max)]))
check(same(conds.map(c => c[0]), ['rain_mm_h', 'air_temp_c', 'humidity_pct', 'wind_kmh']) && conds[1][1] === 10 && conds[1][2] === 33,
  `conditions panel: rain, air temperature, humidity, wind with observed ranges (${conds.map(c => `${c[0]} ${c[1]}–${c[2]}`).join(', ')})`)
check(await page.locator('.conditions .seg[aria-label="Track"] button').count() === 3 && await page.locator('.conditions button:has-text("Reset")').isDisabled(), 'Dry / Damp / Wet presets; Reset disabled at default')
await page.selectOption('select[aria-label="Classification model"]', 'logistic_regression')
const note = await page.locator('[data-weather-models]').innerText()
check(/☂ Logistic Regression/.test(note) && /☂ Ridge/.test(note), `panel shows that both selected models read the weather inputs (${note.replace(/\s+/g, ' ')})`)
const dryRun = await predict(); await page.click('.playback .skip')
await page.click('.conditions .seg[aria-label="Track"] button:has-text("Wet")')
check(await page.locator('#results.is-stale').count() === 1, 'changing conditions marks the prediction out of date')
const hmin = Number(await page.locator('input[data-cond="humidity_pct"]').getAttribute('min'))
check(hmin === 45 && Number(await page.locator('input[data-cond="humidity_pct"]').inputValue()) >= 45, 'rain raises the humidity floor to 45% (no rain in drier air in the data)')
await page.locator('input[data-cond="air_temp_c"]').fill('14'); await page.locator('input[data-cond="wind_kmh"]').fill('28')
check(await ds('weather') === 'wet' && await page.locator('.rain').count() === 1, '3D view: wet track look and rain overlay')
const wetRun = await predict()
check(wetRun.sent.weather === 'wet' && wetRun.sent.rain_mm_h >= 0.25 && wetRun.sent.air_temp_c === 14 && wetRun.sent.wind_kmh === 28, `conditions reach the backend (${JSON.stringify(wetRun.got.conditions)})`)
check(wetRun.got.conditions.air_temp_c === 14 && wetRun.got.conditions.weather === 'wet', 'backend models the requested conditions')
check(!same(wetRun.got.standings.map(r => r.classification.expected_position), dryRun.got.standings.map(r => r.classification.expected_position)),
  'weather-aware classifier changes its prediction for the new conditions')
await wait(2600); await page.click('.playback button[aria-label="Pause"]'); await wait(200)
check(await page.locator('.rain.paused').count() === 1 && same(await tower(), await ranks()), 'paused: rain overlay freezes, timing tower still matches the cars')
await page.click('.playback button[aria-label="Resume"]')
await page.waitForSelector('.board.done', { timeout: 20000 })
check(same(await ranks(), wetRun.got.standings.map(r => r.driver_id)), 'wet race: 3D cars still finish in the predicted order')
await shot('56_wet_race')
await page.click('.conditions button:has-text("Reset")')
check(await ds('weather') === 'dry' && await page.locator('.rain').count() === 0 && await page.locator('.conditions button:has-text("Reset")').isDisabled() &&
  Number(await page.locator('input[data-cond="air_temp_c"]').inputValue()) === 21.1, 'Reset Conditions returns to the dry defaults and the dry look')

// ---------- 9. credits page, phone width ----------
const credits = await (await page.request.get(`${BASE}/credits.html`)).text()
check(/Driver photographs \(\d+ images\)/.test(credits) && /Constructor logos \(\d+ images\)/.test(credits) && /CC BY/.test(credits), 'attribution kept on the separate credits page')
await page.setViewportSize({ width: 390, height: 844 }); await wait(500)
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
check(overflow <= 1 && (await page.locator('.stage').boundingBox()).height > 350, `mobile: track first, no horizontal scroll (${overflow}px)`)
await page.click('.cpick-btn')
const mp = await page.locator('.cpick-pop').boundingBox()
check(mp.x >= 0 && mp.x + mp.width <= 391, 'mobile: circuit menu fits the screen')
await page.keyboard.press('Escape')
await shot('55_mobile', true)

check(errors.length === 0, `no browser console/API/asset errors${errors.length ? ': ' + errors.join(' | ') : ''}`)
await browser.close()
console.log(failures ? `\n${failures} check(s) FAILED` : '\nALL APP CHECKS PASSED')
process.exit(failures ? 1 : 0)
