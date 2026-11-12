import{test}from'node:test'
import assert from'node:assert/strict'
import{readFileSync}from'node:fs'
import*as THREE from'three'
import{circuitFit,constrainCircuitCamera}from'../src/cameraBounds.js'
const circuits=JSON.parse(readFileSync(new URL('../public/circuits.json',import.meta.url))).circuits
function bounds(id){const g=circuits[id],road=g.x.map((x,i)=>new THREE.Vector3(x,g.y?.[i]||0,g.z[i])),box=new THREE.Box3().setFromPoints(road),centre=box.getCenter(new THREE.Vector3());return{points:road,road,centre,radius:box.getSize(new THREE.Vector3()).length()/2,terrain:box.clone().expandByScalar(30),carLength:6.44}}

test('four circuits stay prominent at maximum dolly/pan for different headings and viewport aspects',()=>{
 for(const id of ['spa','suzuka','monaco','monza'])for(const aspect of [.6,1.2,2])for(const az of [0,.8,1.6,2.4]){
  const b=bounds(id),camera=new THREE.PerspectiveCamera(40,aspect,.1,50000),target=b.centre.clone().add(new THREE.Vector3(1e6,1e5,-1e6)),back=new THREE.Vector3(Math.sin(az),.01,Math.cos(az)).normalize();camera.position.copy(target).addScaledVector(back,1e8)
  const limit=constrainCircuitCamera(camera,target,b,{manual:true})
  assert(limit.distance<=limit.max+1e-6);assert(camera.position.distanceTo(target)<=limit.max+1e-6)
  const projected=b.points.map(p=>p.clone().project(camera)),xs=projected.map(p=>p.x),ys=projected.map(p=>p.y)
  const width=Math.max(...xs)-Math.min(...xs),height=Math.max(...ys)-Math.min(...ys)
  assert(Math.max(width,height)>1.25,`${id}/${aspect}/${az}: prominent circuit`)
  assert(Math.min(...xs)>-1.08&&Math.max(...xs)<1.08&&Math.min(...ys)>-1.08&&Math.max(...ys)<1.08,`${id}/${aspect}/${az}: circuit stays in frame at max pan`)
  assert(camera.position.clone().sub(target).normalize().y>=Math.cos(1.12)-1e-8)
 }
})

test('fit is direction/FOV/aspect dependent rather than a single circuit-radius multiplier',()=>{
 const b=bounds('spa'),camera=new THREE.PerspectiveCamera(40,2,.1,50000),back=new THREE.Vector3(.3,1,.8).normalize()
 const wide=circuitFit(b.points,camera,back,b.centre);camera.aspect=.5;const portrait=circuitFit(b.points,camera,back,b.centre);assert(portrait>wide)
 camera.fov=60;const widerFov=circuitFit(b.points,camera,back,b.centre);assert(widerFov<portrait)
})

test('manual Follow/Onboard limit distance and target drift around a road anchor',()=>{
 const b=bounds('suzuka'),anchor=b.road[0].clone(),target=anchor.clone().addScalar(1e5),camera=new THREE.PerspectiveCamera(40,1.2,.1,50000);camera.position.copy(target).add(new THREE.Vector3(1e6,1e6,1e6))
 const limit=constrainCircuitCamera(camera,target,b,{manual:true,close:true,anchor})
 assert.equal(limit.max,b.carLength*12);assert(camera.position.distanceTo(target)<=limit.max+1e-6);assert(target.distanceTo(anchor)<limit.max*.4)
})

test('automatic cockpit camera preserves its close look target and avoids near-plane body cuts',()=>{
 const b=bounds('monza'),target=b.road[0].clone().add(new THREE.Vector3(0,1.5,38)),camera=new THREE.PerspectiveCamera(40,1,.1,50000);camera.position.copy(b.road[0]).add(new THREE.Vector3(0,1.5,.1));const original=target.clone()
 constrainCircuitCamera(camera,target,b,{close:true});assert(target.equals(original));assert(camera.near<=.1)
})
