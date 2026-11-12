import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { CAR_ASSETS, carAppearance } from '../src/carAssets.js'

const appearance = (constructor_id, constructor_season, season = constructor_season) =>
  carAppearance({ driver_id: 'test-driver', constructor_id, constructor_season, season })

test('driver form season never overrides the selected car season', () => {
  const historic = appearance('mclaren', 1988, 2025)
  const contemporary = appearance('red_bull', 2025, 1988)
  assert.equal(historic.asset.id, CAR_ASSETS.turbo.id)
  assert.equal(contemporary.asset.id, CAR_ASSETS.groundEffect.id)
  assert.deepEqual(historic, appearance('mclaren', 1988, 1988))
  assert.deepEqual(contemporary, appearance('red_bull', 2025, 2025))
})

test('required example pairings select era assets without changing entries', () => {
  const cases = [
    ['senna', 1988, 'mclaren', 1988, CAR_ASSETS.turbo],
    ['michael_schumacher', 2004, 'ferrari', 2004, CAR_ASSETS.v10],
    ['hamilton', 2020, 'mercedes', 2020, CAR_ASSETS.halo],
    ['max_verstappen', 2023, 'red_bull', 2023, CAR_ASSETS.groundEffect],
    ['senna', 1988, 'red_bull', 2025, CAR_ASSETS.groundEffect],
  ]
  for (const [driver_id, season, constructor_id, constructor_season, asset] of cases) {
    const entry = Object.freeze({ driver_id, season, constructor_id, constructor_season })
    const before = { ...entry }
    assert.equal(carAppearance(entry).asset, asset)
    assert.deepEqual(entry, before, `${driver_id}: visual mapping must be read-only`)
  }
})

test('regulation boundaries switch body assets and disclose the pre-halo gap', () => {
  for (const [year, expected] of [
    [1981, CAR_ASSETS.turbo], [1989, CAR_ASSETS.turbo],
    [1990, CAR_ASSETS.nineties], [1999, CAR_ASSETS.nineties],
    [2000, CAR_ASSETS.v10], [2009, CAR_ASSETS.v10],
    [2010, CAR_ASSETS.v10], [2017, CAR_ASSETS.v10],
    [2018, CAR_ASSETS.halo], [2021, CAR_ASSETS.halo],
    [2022, CAR_ASSETS.groundEffect], [2025, CAR_ASSETS.groundEffect],
  ]) assert.equal(appearance('ferrari', year).asset, expected, String(year))
  for (const year of [2010, 2014, 2017]) {
    const value = appearance('mercedes', year)
    assert.match(value.modelGap, /unavailable/i)
    assert.match(value.modelGap, /2000s.*approximation/i)
    assert.match(value.era, /pre-halo/i)
  }
  assert.equal(appearance('mercedes', 2018).modelGap, null)
})

test('distinct downloaded era assets retain honest approximation labels', () => {
  const assets = Object.values(CAR_ASSETS)
  assert.equal(new Set(assets.map(a => a.id)).size, assets.length)
  assert.equal(new Set(assets.map(a => a.url)).size, assets.length)
  for (const asset of [CAR_ASSETS.turbo, CAR_ASSETS.nineties, CAR_ASSETS.v10, CAR_ASSETS.halo]) {
    assert.match(asset.quality, /simplified.*approximation/i)
    assert.match(asset.license, /CC0/)
  }
  for (const year of [1988, 1994, 2004, 2016, 2020, 2025]) {
    const value = appearance('ferrari', year)
    assert.equal(value.approximate, true)
    assert.match(value.description, /not an exact/i)
  }
})

test('constructor liveries use multiple colours and respect period changes', () => {
  const early = appearance('mclaren', 1988).palette
  const silver = appearance('mclaren', 2008).palette
  const papaya = appearance('mclaren', 2025).palette
  assert.notDeepEqual(early, silver)
  assert.notDeepEqual(silver, papaya)
  assert.notDeepEqual(appearance('mercedes', 2019).palette, appearance('mercedes', 2020).palette)
  assert.notDeepEqual(appearance('ferrari', 2025).palette, appearance('red_bull', 2025).palette)
  for (const id of ['ferrari', 'red_bull', 'mercedes', 'mclaren', 'benetton', 'brawn', 'renault', 'minardi']) {
    const value = appearance(id, 2004)
    assert.equal(value.palette.length, 3)
    assert.ok(new Set(value.palette).size >= 2, `${id}: a livery needs contrasting regions`)
  }
})

function glbDocument(url) {
  const bytes = readFileSync(new URL(`../public${url}`, import.meta.url))
  assert.equal(bytes.toString('ascii', 0, 4), 'glTF')
  assert.equal(bytes.readUInt32LE(4), 2)
  assert.equal(bytes.readUInt32LE(8), bytes.length, 'header length matches entire download')
  let document
  let offset = 12
  while (offset < bytes.length) {
    const size = bytes.readUInt32LE(offset)
    const kind = bytes.readUInt32LE(offset + 4)
    assert.equal(size % 4, 0, 'chunks are aligned')
    assert.ok(offset + 8 + size <= bytes.length, 'chunk lies inside file')
    if (kind === 0x4e4f534a) document = JSON.parse(bytes.toString('utf8', offset + 8, offset + 8 + size))
    offset += 8 + size
  }
  assert.equal(offset, bytes.length)
  assert.ok(document)
  return document
}

test('all selected GLBs are valid containers without remote buffer or image dependencies', () => {
  for (const asset of Object.values(CAR_ASSETS)) {
    const document = glbDocument(asset.url)
    assert.equal(document.asset.version, '2.0')
    assert.ok(document.meshes.length > 0)
    assert.ok(document.scenes.length > 0)
    for (const resource of [...document.buffers, ...(document.images ?? [])]) {
      assert.ok(!resource.uri || resource.uri.startsWith('data:'), `${asset.id}: external dependency ${resource.uri}`)
    }
    assert.ok(!document.extensionsRequired?.includes('KHR_draco_mesh_compression'), 'no unconfigured Draco decoder dependency')
  }
})

test('historical source records match downloads and preserve CC0 and AI provenance', () => {
  const root = new URL('../public/models/historical/', import.meta.url)
  const records = JSON.parse(readFileSync(new URL('source-manifest.json', root)))
  records.push(JSON.parse(readFileSync(new URL('18685-source.json', root))))
  for (const record of records) {
    assert.equal(record.license.slug, 'cc0-1.0')
    assert.equal(record.aiGenerated, true)
    assert.ok(record.aiModel)
    assert.match(record.url, /^https:\/\/3dassets\.dev\/assets\//)
    assert.equal(readFileSync(new URL(`${record.id}.glb`, root)).length, record.stats.fileSize)
    const document = glbDocument(`/models/historical/${record.id}.glb`)
    for (const corner of ['front-right', 'front-left', 'rear-right', 'rear-left']) {
      assert.ok(document.nodes.some(n => n.name === `wheel-${corner}`), `${record.id}: separate ${corner} pivot`)
    }
  }
  assert.match(readFileSync(new URL('CC0-1.0.txt', root), 'utf8'), /CC0 1.0 Universal/)
})

test('detailed contemporary model embeds original creator and reuse license', () => {
  const document = glbDocument(CAR_ASSETS.groundEffect.url)
  assert.match(document.asset.extras.author, /CarlosCG31/)
  assert.match(document.asset.extras.license, /CC-BY-4\.0/)
  assert.equal(document.asset.extras.source, CAR_ASSETS.groundEffect.source)
})
