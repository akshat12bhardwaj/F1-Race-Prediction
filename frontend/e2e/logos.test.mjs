// Constructor-logo manifest checks (node --test e2e/logos.test.mjs): valid ids, files, era ranges, era picks.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { pickLogo } from '../src/lineup.js'
import { teamColor } from '../src/teams.js'

const LOGOS = JSON.parse(readFileSync(new URL('../src/logos.json', import.meta.url), 'utf8'))
const variants = Object.entries(LOGOS).flatMap(([id, e]) => (e.variants ?? [e]).map(v => ({ id, ...v })))
const png = p => readFileSync(new URL(`../public${p}`, import.meta.url))
const file = (id, season) => pickLogo(LOGOS[id], season)?.file ?? null

test('every logo maps to a known constructor id and a real, credited, bounded PNG', () => {
  for (const v of variants) {
    assert.ok(teamColor(v.id).startsWith('#'), `unknown constructor id ${v.id}`)
    assert.ok(existsSync(new URL(`../public${v.path}`, import.meta.url)), `missing ${v.path}`)
    const b = png(v.path)
    const isPng = b.toString('latin1', 1, 4) === 'PNG'
    assert.ok(isPng || (b[0] === 0xff && b[1] === 0xd8), `${v.path} is neither PNG nor JPEG`)
    if (isPng) { const w = b.readUInt32BE(16), h = b.readUInt32BE(20); assert.ok(w > 0 && h > 0 && w <= 400 && h <= 400, `${v.path} is ${w}x${h}`) }
    assert.ok(v.source?.startsWith('https://commons.wikimedia.org/') && v.licence && v.author, `${v.path} lacks credit`)
    if (v.seasons) assert.ok(v.seasons[0] <= v.seasons[1], `${v.path} seasons ${v.seasons}`)
  }
})

test('no file is mapped twice and no team has two variants for the exact same era', () => {
  const paths = variants.map(v => v.path)
  assert.equal(new Set(paths).size, paths.length, 'a logo file is mapped twice')
  for (const [id, e] of Object.entries(LOGOS)) {
    const eras = (e.variants ?? [e]).map(v => `${v.file}|${v.seasons}`)
    assert.equal(new Set(eras).size, eras.length, `${id}: duplicate variant`)
  }
})

test('era-specific picks follow the car season', () => {
  assert.equal(file('mercedes', 2010), 'Logo Mercedes GP Petronas 2011.svg')
  assert.equal(file('mercedes', 2020), 'Mercedes AMG Petronas F1 Logo.svg')
  assert.equal(file('honda', 2007), 'Logo Honda F1 Racing.svg')            // not the 2024 "new H"
  assert.equal(file('bmw_sauber', 2008), 'BMW Sauber F1 Team logo.png')
  assert.equal(file('renault', 1983), 'Renault Vasarely 1981-1992 Logo.png')
  assert.equal(file('renault', 2005), null)                                 // 2000s Renault: no verified mark -> badge
  assert.equal(file('renault', 2017), 'Renault Sport F1 logo as of 2016.png')
  assert.equal(file('sauber', 2024), 'Logo of Stake F1 Team Kick Sauber.png')
  assert.equal(file('sauber', 1995), null)
  assert.equal(file('williams', 1992), null)
  assert.equal(file('williams', 2021), 'Williams Racing Logo 2020.webp')
  assert.equal(file('williams', 2025), 'Atlassian Williams F1 Team logo.svg')
  assert.equal(file('benetton', 1987), null)
  assert.equal(file('benetton', 1994), 'United Colors of Benetton - logo (Italy, 1989-1996).svg')
  assert.equal(file('alfa', 1983), null)
  assert.equal(file('alfa', 2023), 'Alfa Romeo F1 Team Stake Logo.svg')
  assert.equal(file('ligier', 1990), null)                                  // modern Ligier marks never shown on F1 cars
})

test('a cross-era pairing shows the assigned car\'s logo, not the driver\'s team', () => {
  const senna88InMercedes20 = { driver_id: 'senna', season: 1988, constructor_id: 'mercedes', constructor_season: 2020 }
  const l = pickLogo(LOGOS[senna88InMercedes20.constructor_id], senna88InMercedes20.constructor_season)
  assert.equal(l.file, 'Mercedes AMG Petronas F1 Logo.svg')
})
