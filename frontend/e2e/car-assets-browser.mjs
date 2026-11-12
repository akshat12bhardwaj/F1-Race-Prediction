import { chromium } from 'playwright-core'
import assert from 'node:assert/strict'
import { writeFileSync } from 'node:fs'
const base = process.env.E2E_URL || 'http://127.0.0.1:8011'
const browser = await chromium.launch({ channel:'msedge',headless:false,args:['--ignore-gpu-blocklist'] })
const page = await browser.newPage({viewport:{width:1600,height:960}})
const errors=[], downloads=[]
page.on('pageerror',e=>{errors.push(e.message);console.log('PAGEERROR',e.message)})
page.on('console',m=>{if(m.type()==='warning'||m.type()==='error')console.log(m.type(),m.text())})
page.on('response',r=>{if(r.url().endsWith('.glb'))downloads.push({url:r.url(),status:r.status()})})
const pool=await (await page.request.get(base+'/api/driver-pool')).json()
// A valid full race covering all five GLB bodies, using only eligible constructor peaks and driver form seasons.
const entries=pool.default_grid.map(e=>({...e}))
for(const [at,id] of [[16,'williams'],[18,'alphatauri'],[20,'aston_martin']]){
 const car=pool.teams.find(c=>c.constructor_id===id);assert(car,id+' eligible peak')
 for(const i of [at,at+1])Object.assign(entries[i],{constructor_id:id,constructor_season:car.season})
}
const examples=[entries[0],entries[2],entries[16],entries[18],entries[20]]
await page.route('**/api/driver-pool',async route=>{const response=await route.fetch();await route.fulfill({response,json:{...pool,default_grid:entries}})})
const started = Date.now()
await page.goto(base)
await page.waitForFunction(()=>document.querySelector('[data-assets-loading="0"][data-assets-failed="0"]')&&Object.keys(document.querySelector('[data-viewer="3d"]').__apex.models()).length===22)
const readyMs = Date.now() - started
const V=page.locator('[data-viewer="3d"]'), models=await V.evaluate(el=>el.__apex.models())
const expected=['15176','15177','15173','18685','formula-f222']
examples.forEach((e,i)=>{const m=models[e.driver_id];assert.equal(m.asset.id,expected[i]);assert.equal(m.season,e.constructor_season);assert.equal(m.meshes,5)})
assert.equal(downloads.length,5);assert(downloads.every(r=>r.status===200))
for(const e of examples){
 await page.locator(`.grid-sum li[data-driver="${e.driver_id}"] button`).click()
 await page.locator('.cam-bar button').filter({hasText:/^Follow$/}).click()
 await page.waitForFunction(()=>{const el=document.querySelector('[data-viewer="3d"]');return el.dataset.mode==='follow'&&Number(el.dataset.follow)<2.5})
 await page.waitForTimeout(1100)
 await page.screenshot({path:`../outputs/figures/ui/overhaul-${e.driver_id}.png`})
}
await page.locator('.cam-bar button').filter({hasText:/^Onboard$/}).click()
await page.waitForFunction(()=>Number(document.querySelector('[data-viewer="3d"]').dataset.follow)<.35)
await page.screenshot({path:'../outputs/figures/ui/overhaul-historical-onboard.png'})
await page.locator('.cam-bar button').filter({hasText:/^Overview$/}).click();await page.waitForTimeout(900)
assert.equal(downloads.length,5,'camera switching does not redownload geometry')
await page.locator('.run .btn.lime').click();await page.waitForSelector('.board.racing',{timeout:60000})
await page.waitForSelector('.board.done',{timeout:20000})
const order=await V.evaluate(el=>Object.entries(el.__apex.cars()).sort((a,b)=>a[1].rank-b[1].rank).map(x=>x[0]))
const standings=await page.locator('.standings li').evaluateAll(els=>els.map(e=>e.dataset.driver))
assert.deepEqual(order,standings)
assert.equal(errors.length,0,errors.join('\n'))
writeFileSync('../outputs/logs/car-assets-browser.json',JSON.stringify({readyMs,resources:await page.evaluate(()=>performance.getEntriesByType('resource').filter(r=>r.name.endsWith('.glb')).map(r=>({url:r.name,durationMs:r.duration,transferBytes:r.transferSize}))),examples,models,downloads,errors,order,standings},null,2))
// Missing assets preserve identities and controls without an alternative rendering mode.
const failurePage = await browser.newPage()
await failurePage.route('**/*.glb', route => route.abort())
await failurePage.goto(base)
await failurePage.waitForSelector('[data-assets-loading="0"][data-assets-failed="22"]')
assert.equal(await failurePage.locator('[data-viewer="3d"]').evaluate(el=>Object.keys(el.__apex.cars()).length),22)
assert.match(await failurePage.locator('.asset-note').textContent(), /unavailable/)
assert.equal(await failurePage.locator('.cam-bar button').count(),5)
await browser.close()
console.log('PASS five cross-era GLBs, constructor palettes, 5 meshes/car, cached loading, Follow/Onboard, replay/standings synchronization, failed-asset handling, no browser errors')
