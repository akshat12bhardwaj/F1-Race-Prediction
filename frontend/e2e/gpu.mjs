// Real-GPU performance check: opens a visible browser window (hardware WebGL), races the full 22-car grid in HD
// and reports frames per second. Also takes close-ups of the Suzuka crossover.   Run: node e2e/gpu.mjs
import { chromium } from 'playwright-core'
import { mkdirSync } from 'node:fs'

const BASE = process.env.E2E_URL || 'http://127.0.0.1:8011'
const SHOTS = new URL('../../outputs/figures/ui/', import.meta.url)
mkdirSync(SHOTS, { recursive: true })
let failures = 0
const check = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`); if (!cond) failures++ }
const browser = await chromium.launch({ channel: process.env.E2E_BROWSER === 'chrome' ? 'chrome' : 'msedge', headless: false, args: ['--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1600, height: 960 } })
const shot = name => page.screenshot({ path: new URL(`${name}.png`, SHOTS).pathname.replace(/^\/(\w:)/, '$1') })
const V = '[data-viewer="3d"]'
const ds = k => page.locator(V).getAttribute(`data-${k}`)
const wait = ms => page.waitForTimeout(ms)
const circuit = async id => {                       // header circuit picker: search by id, take the first match
  await page.click('.cpick-btn'); await page.fill('.cpick-pop input', { spa: 'spa', monaco: 'monaco', suzuka: 'suzuka' }[id])
  await page.click(`.cpick-pop li[data-circuit="${id}"]`)
}
// frames per second and the slowest frame over `ms` of playback
const measure = ms => page.evaluate(ms => new Promise(res => {
  let n = 0, worst = 0, last = performance.now()
  const t0 = last, f = now => { n++; worst = Math.max(worst, now - last); last = now; now - t0 < ms ? requestAnimationFrame(f) : res({ fps: n / ((now - t0) / 1000), worst }) }
  requestAnimationFrame(f)
}), ms)

await page.goto(BASE)
await page.waitForFunction(() => document.querySelectorAll('.grid-sum li').length === 22, null, { timeout: 30000 })
await page.waitForSelector('[data-assets-loading="0"][data-assets-failed="0"]')
check(await page.locator(V).evaluate(el => Object.keys(el.__apex.models()).length === 22), 'all 22 detailed GLB cars loaded')
await page.waitForFunction(v => document.querySelector(v)?.dataset.cam, V, { timeout: 30000 })
const gpu = await page.locator(V).evaluate(el => el.__apex.gpu())
const hw = !/swiftshader|llvmpipe|software|basic render/i.test(gpu)
console.log(`renderer: ${gpu}`)
check(hw, 'WebGL runs on real graphics hardware (not a software rasteriser)')

for (const [id, mode] of [['spa', 'Overview'], ['spa', 'Follow'], ['monaco', 'Onboard']]) {
  await circuit(id)
  await page.waitForSelector(`${V}[data-circuit="${id}"]`); await wait(600)
  await page.click('.run .btn.lime'); await page.waitForSelector('.board.racing', { timeout: 60000 })
  await page.click(`.cam-bar button:has-text("${mode}")`); await wait(900)
  const m = await measure(6000), info = await page.locator(V).evaluate(el => el.__apex.info())
  check(!hw || m.fps >= 50, `${id} · ${mode}: ${m.fps.toFixed(0)} fps with 22 cars in permanent HD (slowest frame ${m.worst.toFixed(0)} ms, ${info.calls} draw calls, ${info.triangles} triangles)`)
  await page.waitForSelector('.board.done', { timeout: 20000 })
  await page.click('.cam-bar button:has-text("Overview")'); await wait(900)
}
check(await ds('quality') === 'high', 'HD stays active throughout playback')
// heavy rain: the weather overlay must not slow the race down
await page.click('.conditions .seg[aria-label="Track"] button:has-text("Wet")'); await page.locator('input[data-cond="rain_mm_h"]').fill('4')
await page.click('.run .btn.lime'); await page.waitForSelector('.board.racing', { timeout: 60000 }); await wait(900)
const rm = await measure(5000)
check(!hw || rm.fps >= 50, `heavy rain overlay: ${rm.fps.toFixed(0)} fps (slowest frame ${rm.worst.toFixed(0)} ms)`)
await page.waitForSelector('.board.done', { timeout: 20000 })
await page.click('.conditions button:has-text("Reset")')

// Suzuka crossover close-ups: follow the pole car under the bridge (first pass) and over it (second pass)
await circuit('suzuka')
await page.waitForSelector(`${V}[data-circuit="suzuka"]`); await wait(600)
await page.click('.run .btn.lime'); await page.waitForSelector('.board.racing', { timeout: 60000 })
await page.click('.playback .skip'); await page.waitForSelector('.board.done')
await page.click('.standings li:first-child button'); await wait(1200)
const range = page.locator('.playback input[type=range]')
await range.fill('405'); await wait(1500); await shot('47_suzuka_under')
await range.fill('790'); await wait(1500); await shot('48_suzuka_over')
await page.click('.cam-bar button:has-text("Overview")'); await wait(1000)
await page.mouse.move(800, 480); await page.mouse.wheel(0, -900); await wait(900); await shot('49_suzuka_zoom')
await browser.close()
console.log(failures ? `\n${failures} check(s) FAILED` : '\nGPU CHECKS PASSED')
process.exit(failures ? 1 : 0)
