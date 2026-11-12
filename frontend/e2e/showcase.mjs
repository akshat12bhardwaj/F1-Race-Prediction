// Capture the real application for the README. Run with E2E_URL set to a running APEX instance.
import { chromium } from 'playwright-core'
import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import assert from 'node:assert/strict'

const base = process.env.E2E_URL || 'http://127.0.0.1:8012'
const out = fileURLToPath(new URL('../../docs/assets/screenshots/', import.meta.url))
mkdirSync(out, { recursive: true })
const browser = await chromium.launch({ channel: process.env.E2E_BROWSER || 'msedge', headless: true, args: ['--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1600, height: 960 }, deviceScaleFactor: 1 })
const errors = [], captures = []
page.on('pageerror', e => errors.push(e.message))
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()) })
page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`) })
const shot = async name => {
  await page.screenshot({ path: out + name + '.png' })
  captures.push(name)
  console.log('CAPTURE', name)
}
const viewer = '[data-viewer="3d"]'
try {
  await page.goto(base)
  await page.waitForSelector('[data-assets-loading="0"][data-assets-failed="0"]')
  assert.equal(await page.locator('.grid-sum li').count(), 22)
  await page.waitForTimeout(800)
  await shot('dashboard')
  await page.click('.edit-lineup')
  await page.waitForSelector('.slots .slot[data-driver]')
  assert.equal(await page.locator('.editor select, .pool-card .num25, .slots .tag.swap').count(), 0)
  await page.waitForTimeout(500)
  await shot('lineup-editor')
  await page.fill('input[aria-label="Search drivers"]', 'senna')
  await page.locator('.pool-card[data-driver="senna"] img').evaluate(i => i.decode())
  await shot('driver-selection')
  await page.fill('input[aria-label="Search drivers"]', '')

  // Swap both Toyota seats to the eligible peak Aston Martin: keep 22 drivers, 11 teams and two seats each.
  await page.fill('input[aria-label="Search constructors"]', 'Aston Martin')
  const target = await page.locator('.slot[data-slot="20"]').getAttribute('data-driver')
  for (const slot of [20, 21]) {
    await page.locator(`.slot[data-slot="${slot}"] .drag`).click()
    await page.locator('.team-card[data-team="aston_martin"] button').click()
  }
  assert.equal(await page.locator('.ed-grid [data-ready="true"]').count(), 1)
  await shot('constructor-selection')
  await page.click('button:has-text("Done — Back to Race")')
  await page.waitForSelector('[data-assets-loading="0"][data-assets-failed="0"]')
  await page.selectOption('select[aria-label="Classification model"]', 'logistic_regression')
  await page.selectOption('select[aria-label="Regression model"]', 'ridge')
  await page.click('.conditions .seg[aria-label="Track"] button:has-text("Damp")')
  await shot('environment')
  const response = page.waitForResponse(r => r.url().endsWith('/api/race-prediction') && r.request().method() === 'POST')
  await page.click('.run .btn.lime')
  const prediction = await (await response).json()
  assert.equal(prediction.standings.length, 22)
  await page.waitForFunction(v => Number(document.querySelector(v)?.dataset.p) > 0.45, viewer)
  await page.click('.playback button[aria-label="Pause"]')
  assert.equal(await page.locator('.tower li').count(), 22)
  await shot('race-overview')
  await page.locator(`.tower li[data-driver="${target}"] button`).click()
  await page.waitForFunction(v => document.querySelector(v)?.dataset.mode === 'follow' && Number(document.querySelector(v).dataset.follow) < 2.5, viewer)
  await page.waitForTimeout(800)
  await shot('race-follow')
  await page.click('.cam-bar button:has-text("Onboard")')
  await page.waitForTimeout(1100)
  await shot('race-onboard')
  await page.click('.cam-bar button:has-text("Overview")')
  await page.click('.playback button[aria-label="Resume"]')
  await page.waitForSelector('.board.done', { timeout: 20000 })
  const final = await page.locator('.standings li').evaluateAll(es => es.map(e => e.dataset.driver))
  assert.deepEqual(final, prediction.standings.map(r => r.driver_id))
  assert.equal(await page.locator('.pod').count(), 3)
  await page.waitForTimeout(700)
  await shot('prediction-results')
  assert.deepEqual(errors, [])
  writeFileSync(out + 'capture.json', JSON.stringify({ viewport: '1600x960', captures, errors, source: 'Real local APEX UI; seeded model prediction; no mock responses', standings: final }, null, 2))
  console.log(`PASS ${captures.length} real screenshots; 22-car prediction, three podium places and zero browser errors`)
} finally {
  await browser.close()
}
