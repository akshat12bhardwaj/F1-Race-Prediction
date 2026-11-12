import {chromium} from 'playwright-core'
import assert from 'node:assert/strict'
import {mkdirSync,writeFileSync} from 'node:fs'
import {fileURLToPath} from 'node:url'
const base=process.env.E2E_URL||'http://127.0.0.1:8011'
const shots=new URL('../../outputs/figures/ui/solid-camera/',import.meta.url);mkdirSync(shots,{recursive:true})
const browser=await chromium.launch({channel:'msedge',headless:false,args:['--ignore-gpu-blocklist']})
const page=await browser.newPage({viewport:{width:1600,height:960},hasTouch:true})
const errors=[],records=[];let checks=0
page.on('pageerror',e=>errors.push(e.message))
const V=page.locator('[data-viewer="3d"]')
const check=(ok,label)=>{assert(ok,label);checks++;console.log('PASS',label)}
const state=()=>V.evaluate(el=>el.__apex.cameraLimits())
const shot=name=>page.screenshot({path:fileURLToPath(new URL(name+'.png',shots))})
const mode=async name=>{await page.locator('.cam-bar button').filter({hasText:new RegExp('^'+name+'$')}).click();await page.waitForTimeout(1000)}
async function assertBounds(label,wide){const s=await state();records.push({label,...s});check(s.distance<=s.max+0.1,label+' hard distance limit');const clearance=await V.evaluate(el=>el.__apex.clearance());check(clearance===null||clearance>.3,label+' terrain clearance');if(wide){const o=s.occupancy;if(!(o.minX>-1.15&&o.maxX<1.15&&o.minY>-1.15&&o.maxY<1.15)){console.log('OUTSIDE',label,JSON.stringify(s));await shot('failure-outside')}check(Math.max(o.maxX-o.minX,o.maxY-o.minY)>1.25,label+' circuit prominence');check(o.minX>-1.15&&o.maxX<1.15&&o.minY>-1.15&&o.maxY<1.15,label+' circuit stays visible')}else{const p=await V.evaluate(el=>el.__apex.screen(el.__apex.cameraLimits().selected));if(!p){console.log('MISSING',label,JSON.stringify(s));await shot('failure-missing-car')}const r=await V.boundingBox();check(p&&p[0]>r.x&&p[0]<r.x+r.width&&p[1]>r.y&&p[1]<r.y+r.height,label+' selected car remains visible')}}
async function pan(dx,dy){const r=await V.boundingBox(),x=r.x+r.width*.5,y=r.y+r.height*.5;await page.mouse.move(x,y);await page.mouse.down({button:'right'});await page.mouse.move(x+dx,y+dy,{steps:8});await page.mouse.up({button:'right'});await page.waitForTimeout(200)}
await page.goto(base);await page.waitForFunction(()=>document.querySelectorAll('.grid-sum li').length===22);await page.waitForSelector('[data-assets-loading="0"][data-assets-failed="0"]')
console.log('GPU',await V.evaluate(el=>el.__apex.gpu()))
for(const circuit of ['spa','suzuka','monaco','monza']){
 if(circuit!=='spa'){await page.click('.cpick-btn');await page.fill('.cpick-pop input',circuit);await page.click(`.cpick-pop li[data-circuit="${circuit}"]`)}
 await page.waitForSelector(`[data-viewer="3d"][data-circuit="${circuit}"]`);await page.waitForSelector('[data-assets-loading="0"]')
 for(const name of ['Overview','Top','Follow','Onboard']){
  await mode(name);if(name==='Follow'||name==='Onboard')await shot(`after-${circuit}-${name.toLowerCase()}`)
  const r=await V.boundingBox();await page.mouse.move(r.x+r.width*.5,r.y+r.height*.5);for(let i=0;i<3;i++)await page.mouse.wheel(0,100000);await page.waitForTimeout(300)
  await assertBounds(`${circuit}/${name}/max zoom`,name==='Overview'||name==='Top')
  for(const [dx,dy]of[[3500,2000],[-7000,-4000]]){await pan(dx,dy);await assertBounds(`${circuit}/${name}/max pan ${dx}`,name==='Overview'||name==='Top')}
  await shot(`after-${circuit}-${name.toLowerCase()}-max-pan`)
 }
 await mode('Overview');await page.setViewportSize({width:1100,height:800});await page.waitForTimeout(800);await assertBounds(`${circuit}/resize`,true);await shot(`after-${circuit}-resize`);await page.setViewportSize({width:1600,height:960});await page.waitForTimeout(600)
 await mode('Follow');await page.click('.run .btn.lime');await page.waitForSelector('.board.racing',{timeout:60000});await page.waitForFunction(()=>Number(document.querySelector('[data-viewer="3d"]').dataset.p)>.3)
 await page.click('.playback button[aria-label="Pause"]');await page.waitForTimeout(100)
 const frozen=await V.getAttribute('data-p');const cars=await V.evaluate(el=>el.__apex.cars());await page.waitForTimeout(250)
 check(await V.getAttribute('data-p')===frozen,`${circuit}/race pauses`)
 const liveOrder=Object.entries(cars).sort((a,b)=>a[1].rank-b[1].rank).map(x=>x[0]);const tower=await page.locator('.tower li').evaluateAll(es=>es.sort((a,b)=>Number(a.dataset.pos)-Number(b.dataset.pos)).map(e=>e.dataset.driver))
 check(JSON.stringify(liveOrder)===JSON.stringify(tower),`${circuit}/paused timing matches cars`);await shot(`after-${circuit}-race-paused-follow`)
 await mode('Onboard');await shot(`after-${circuit}-race-paused-onboard`);await page.click('.playback button[aria-label="Resume"]');await page.waitForSelector('.board.done',{timeout:20000})
 const finished=await V.evaluate(el=>Object.entries(el.__apex.cars()).sort((a,b)=>a[1].rank-b[1].rank).map(x=>x[0]));const predicted=await page.locator('.standings li').evaluateAll(es=>es.map(e=>e.dataset.driver))
 check(finished.length===22&&JSON.stringify(finished)===JSON.stringify(predicted),`${circuit}/22 cars resume and finish in predicted order`)

}
// Two-finger pinch exercises the touch path, including handoff from tracked cameras.
await page.click('.cpick-btn');await page.fill('.cpick-pop input','spa');await page.click('.cpick-pop li[data-circuit="spa"]');await page.waitForSelector('[data-circuit="spa"]')
const cdp=await page.context().newCDPSession(page)
for(const name of ['Overview','Top','Follow','Onboard']){await mode(name);const r=await V.boundingBox(),x=r.x+r.width*.5,y=r.y+r.height*.5;await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:x-120,y,id:1},{x:x+120,y,id:2}]});for(const n of [80,40,10,2])await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:x-n,y,id:1},{x:x+n,y,id:2}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await page.waitForTimeout(400);await assertBounds('touch/'+name,name==='Overview'||name==='Top')}
check(errors.length===0,'no browser exceptions')
// Isolated real-GPU car views make culling/lighting defects visible without terrain or near-plane cuts.
for(const asset of ['turbo','nineties','v10','halo','groundEffect'])for(const view of ['side','front','rear','threequarter']){await page.goto(`http://127.0.0.1:5177/e2e/car-inspection.html?asset=${asset}&view=${view}`);await page.waitForFunction(()=>window.ready);check(await page.evaluate(()=>window.inspectCar.materials().every(m=>m.side===0&&!m.transparent&&m.opacity===1)),`${asset}/${view}/opaque front-sided materials`);await shot(`after-${asset}-${view}`)}
writeFileSync(fileURLToPath(new URL('../../outputs/logs/solid-camera-visual.json',import.meta.url)),JSON.stringify({checks,errors,records},null,2))
await browser.close();console.log(`ALL ${checks} CAMERA/VISUAL CHECKS PASSED; 20 era-angle screenshots captured`)
