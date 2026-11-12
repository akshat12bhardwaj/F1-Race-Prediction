import { chromium } from 'playwright-core'
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
const base = process.env.E2E_URL || 'http://127.0.0.1:8011'
const dir = fileURLToPath(new URL('../../outputs/asset_review/portraits-completion/browser/', import.meta.url))
mkdirSync(dir, {recursive:true})
const reviewed=JSON.parse(readFileSync(new URL('../../src/reviewed_portraits.json',import.meta.url),'utf8'))
const manifest=JSON.parse(readFileSync(new URL('../src/portraits.json',import.meta.url),'utf8'))
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--ignore-gpu-blocklist']})
const page=await browser.newPage({viewport:{width:1600,height:960}})
const errors=[];page.on('pageerror',e=>errors.push(String(e)))
const pool=await (await page.request.get(base+'/api/driver-pool')).json()
await page.goto(base);await page.waitForSelector('.edit-lineup');await page.click('.edit-lineup');await page.waitForSelector('.pool-card')
if(await page.locator('.editor select').count())throw Error('Obsolete editor season/era selector')
const rows=[]
for (const [id,item] of Object.entries(reviewed)) {
 await page.fill('input[aria-label="Search drivers"]',item.article)
 const card=page.locator(`.pool-card[data-driver="${id}"]`)
 await card.scrollIntoViewIfNeeded()
 await card.locator('img.portrait').evaluate(i=>i.decode())
 const result=await card.locator('img.portrait').evaluate(i=>({src:i.getAttribute('src'),width:i.naturalWidth,height:i.naturalHeight,lazy:i.loading,draggable:i.draggable}))
 if(result.src!==item.path||result.width!==256||result.height!==256||result.lazy!=='lazy'||result.draggable) throw Error(id+' invalid portrait')
 await card.screenshot({path:dir+id+'.png'})
 rows.push({id,...result})
 console.log('PASS rendered lineup card',id)
}
const missing=pool.drivers.filter(d=>!manifest[d.driver_id])
for(const d of missing){
 await page.fill('input[aria-label="Search drivers"]',`${d.first_name} ${d.last_name}`)
 if(await page.locator(`.pool-card[data-driver="${d.driver_id}"] .portrait.mono`).count()!==1)throw Error(d.driver_id+' missing fallback')
}
await page.route('**/drivers/pizzonia.jpg',route=>route.abort())
await page.fill('input[aria-label="Search drivers"]','Pizzonia')
await page.locator('.pool-card[data-driver="pizzonia"] .portrait.mono').waitFor()
console.log('PASS failed image retains initials fallback')
// Enlarge the actual loaded image assets in a separate QA page after checking production cards.
const review=await browser.newPage({viewport:{width:1420,height:420}})
await review.goto(base+'/credits.html')
await review.setContent(`<body style="background:#151719;color:white;font:16px system-ui;display:flex;gap:16px;padding:10px">${Object.entries(reviewed).map(([id,m])=>`<figure style="margin:0"><img src="${base}${m.path}" width=256 height=256 style="object-fit:cover"><figcaption>${m.article}</figcaption></figure>`).join('')}</body>`)
await review.locator('img').evaluateAll(ims=>Promise.all(ims.map(i=>i.decode())))
await review.screenshot({path:dir+'all-new-portraits.png'})
const report={eligible:pool.drivers.length,coverage:pool.drivers.length-missing.length,rows,missing:missing.map(d=>d.driver_id),errors}
writeFileSync(dir+'results.json',JSON.stringify(report,null,2))
await browser.close()
if(errors.length)throw Error(errors.join('\n'))
console.log(`PASS ${rows.length} new portraits visually reviewable; ${missing.length} initials fallbacks; ${report.coverage}/${report.eligible} coverage; no browser errors`)
