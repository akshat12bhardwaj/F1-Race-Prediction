// Browser check of constructor logos (npm run e2e:logos; server as for e2e/app.mjs): every eligible constructor shows
// its era's verified logo or the badge, logos follow the assigned car, and screenshots for a visual review.
import { chromium } from 'playwright-core'
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { pickLogo } from '../src/lineup.js'
const base = process.env.E2E_URL || 'http://127.0.0.1:8011'
const dir = fileURLToPath(new URL('../../outputs/asset_review/logos-completion/browser/', import.meta.url))
mkdirSync(dir, { recursive: true })
const LOGOS = JSON.parse(readFileSync(new URL('../src/logos.json', import.meta.url), 'utf8'))
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] })
const page = await browser.newPage({ viewport: { width: 1600, height: 960 } })
const errors = []
page.on('pageerror', e => errors.push(String(e)))
page.on('response', r => { if (r.status() >= 400 && r.url().includes('/constructors/')) errors.push(`HTTP ${r.status()} ${r.url()}`) })
let fails = 0
const check = (ok, msg) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`); if (!ok) fails++ }
const pool = await (await page.request.get(base + '/api/driver-pool')).json()
await page.goto(base); await page.waitForSelector('.edit-lineup'); await page.click('.edit-lineup'); await page.waitForSelector('.team-card')
check(await page.locator('.editor select').count() === 0, 'no era or season selectors')

// the mark a car card shows: logo file + its natural size against the rendered box (aspect kept), or the badge
const mark = async (id, season, name) => {
  await page.fill('input[aria-label="Search constructors"]', name)
  const card = page.locator(`.team-card[data-team="${id}"][data-season="${season}"]`)
  await card.waitFor()
  return card.evaluate(async c => {
    const i = c.querySelector('.logo-tile img')
    if (!i) return { badge: !!c.querySelector('.badge') }
    await i.decode()
    const r = i.getBoundingClientRect()
    return { src: i.getAttribute('src'), nw: i.naturalWidth, nh: i.naturalHeight, w: r.width, h: r.height, lazy: i.loading }
  })
}
const byTeam = Object.groupBy(pool.teams, c => c.constructor_id)
const rows = []
for (const [id, cars] of Object.entries(byTeam).sort()) {
  const want = cars.map(c => [c, pickLogo(LOGOS[id], c.season)])
  const [car, logo] = want.find(([, l]) => l) ?? want.at(-1)           // a logo season when the team has one
  const got = await mark(id, car.season, car.constructor_name)
  const ok = logo ? got.src === logo.path && got.nw > 0 && Math.abs(got.w / got.h - got.nw / got.nh) < 0.08 && got.lazy === 'lazy' : got.badge
  check(ok, `${id} ${car.season}: ${logo ? `${logo.file} (${got.nw}x${got.nh} shown ${got.w.toFixed(0)}x${got.h.toFixed(0)})` : 'badge'}`)
  rows.push({ id, seasons: [cars[0].season, cars.at(-1).season], logo_seasons: want.filter(([, l]) => l).map(([c]) => c.season), shown: got.src ?? 'badge' })
}
for (const { constructor_id: id, season, constructor_name: name } of pool.teams.filter(c => ['mercedes', 'honda', 'renault', 'benetton', 'williams', 'sauber'].includes(c.constructor_id))) {
  const l = pickLogo(LOGOS[id], season), got = await mark(id, season, name)
  check(l ? got.src === l.path : got.badge, `era pick ${id} ${season}: ${l?.file ?? 'badge'}`)
}

// Searchable constructor list; seasons remain internal.
await page.fill('input[aria-label="Search constructors"]', '')
await page.locator('.ed-col').last().screenshot({ path: `${dir}constructors-peak.png` })

// reassign a grid driver's car: the slot's mark follows the car season, not the driver's
const slot = page.locator('.slots .slot[data-driver]').first()
const assign = async (id, season, name) => {
  await page.fill('input[aria-label="Search constructors"]', name)
  if (!(await slot.getAttribute('class')).includes('sel')) await slot.locator('.drag').click()
  await page.locator(`.team-card[data-team="${id}"][data-season="${season}"] button:has-text("Assign")`).click()
  await page.waitForTimeout(200)
  return slot.evaluate(s => s.querySelector('.logo-tile')?.dataset.logoFile ?? (s.querySelector('.car-line .dot') ? 'dot' : null))
}
const driver = await slot.getAttribute('data-driver')
for (const { constructor_id: id, season, constructor_name: name } of pool.teams.filter(c => ['benetton', 'mercedes', 'brabham'].includes(c.constructor_id))) {
  const want = pickLogo(LOGOS[id], season)?.path ?? 'dot'
  const got = await assign(id, season, name)
  check(got === want && (await slot.getAttribute('data-driver')) === driver, `${driver} given the ${season} ${name}: slot shows ${got}`)
}
const mercedes = pool.teams.find(c => c.constructor_id === 'mercedes')
await assign('mercedes', mercedes.season, 'Mercedes')
await page.locator('.slots').screenshot({ path: `${dir}grid-slots.png` })

// failed image -> badge
await page.route('**/constructors/brawn-p0.png', r => r.abort())
await page.fill('input[aria-label="Search constructors"]', 'Brawn')
await page.locator('.team-card[data-team="brawn"] .badge').waitFor()
check(true, 'a failed logo image falls back to the badge')

// race: predicted podium + selected-driver card carry the assigned car's mark
// Restore a valid 11-team grid after exercising arbitrary assignments, then swap into a full Mercedes seat.
await page.click('button:has-text("Restore default lineup")')
const raceDriver = await slot.getAttribute('data-driver')
await assign('mercedes', mercedes.season, 'Mercedes')
await page.click('text=Done — Back to Race')
await page.click('.run .btn.lime')
await page.waitForSelector('.playback .skip:not([disabled])', { timeout: 60000 }); await page.click('.playback .skip')
await page.waitForSelector('.board.done', { timeout: 60000 })
const pods = await page.locator('.pod').evaluateAll(ps => ps.map(p => {
  const g = document.querySelector(`.grid-sum li[data-driver="${p.dataset.driver}"]`)          // the car that driver was given
  return [g.dataset.team, Number(g.dataset.carSeason), p.querySelector('.logo-tile')?.dataset.logoFile ?? null]
}))
check(pods.length === 3 && pods.every(([id, s, got]) => got === (pickLogo(LOGOS[id], s)?.path ?? null)), `podium marks follow each assigned car: ${JSON.stringify(pods)}`)
await page.locator('.podium').screenshot({ path: `${dir}podium.png` })
await page.locator(`.standings li[data-driver="${raceDriver}"] button`).click()
await page.locator('.driver-card').waitFor()
const card = await page.locator('.driver-card').evaluate(c => c.querySelector('.logo-tile')?.dataset.logoFile ?? (c.querySelector('.badge') ? 'badge' : null))
check(card === pickLogo(LOGOS.mercedes, mercedes.season).path, `selected-driver card shows the assigned peak Mercedes mark (${card})`)
await page.locator('.driver-card').screenshot({ path: `${dir}driver-card.png` })

writeFileSync(dir + 'results.json', JSON.stringify({ constructors: rows.length, with_logo: rows.filter(r => r.logo_seasons.length).length, rows, errors }, null, 1))
await browser.close()
check(!errors.length, `no browser or logo-request errors ${errors.join(' | ')}`)
console.log(`${rows.filter(r => r.logo_seasons.length).length}/${rows.length} constructors show a verified logo at their selected peak season; others retain badges`)
if (fails) { console.log(`${fails} check(s) FAILED`); process.exit(1) }
