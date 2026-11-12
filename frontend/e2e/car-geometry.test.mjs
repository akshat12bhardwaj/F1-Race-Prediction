import {test} from 'node:test'
import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {readFileSync} from 'node:fs'
import * as THREE from 'three'
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js'
import {MeshoptDecoder} from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import {orientCarGeometry,F222_REPAIRS} from '../src/carGeometry.js'

const bytes=readFileSync(new URL('../public/models/formula-2022.glb',import.meta.url))
const {scene}=await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'')
scene.rotation.y=Math.PI/2;scene.updateMatrixWorld(true)
function converted(mesh,repair=true){
 const geometry=new THREE.BufferGeometry()
 for(const key of ['position','normal']){const a=mesh.geometry.attributes[key],values=[];for(let i=0;i<a.count;i++)values.push(a.getX(i),a.getY(i),a.getZ(i));geometry.setAttribute(key,new THREE.Float32BufferAttribute(values,3))}
 if(mesh.geometry.index)geometry.setIndex(mesh.geometry.index.clone())
 return orientCarGeometry(geometry,mesh.name,mesh.matrixWorld,repair)
}
function triangles(g,fn){const p=g.attributes.position,n=g.attributes.normal,ix=g.index;for(let i=0;i<(ix?.count??p.count);i+=3){const ids=[0,1,2].map(j=>ix?ix.getX(i+j):i+j);fn(...ids.map(k=>new THREE.Vector3().fromBufferAttribute(p,k)),new THREE.Vector3().fromBufferAttribute(n,ids[0]))}}
function volume(g){let v=0;triangles(g,(a,b,c)=>{v+=a.dot(b.cross(c))/6});return v}

test('actual F222 source reproduces missing exterior sidepod surfaces; repaired shell faces outward',()=>{
 const body=scene.getObjectByName('Object_60')
 for(const repair of [false,true]){
  const mesh=new THREE.Mesh(converted(body,repair),new THREE.MeshBasicMaterial({side:THREE.FrontSide}));mesh.updateMatrixWorld(true)
  for(const side of [-1,1]){
   const hits=new THREE.Raycaster(new THREE.Vector3(side*3,.52,-.5),new THREE.Vector3(-side,0,0)).intersectObject(mesh)
   assert(hits.length>0)
   assert.equal(hits[0].point.x*side>.2,repair,'outside ray must hit the near exterior, not the far interior')
  }
  mesh.geometry.dispose();mesh.material.dispose()
 }
})

test('closed reversed F222 wing and wheel shells have positive signed volume after repair',()=>{
 for(const name of ['Object_112','Object_29','Object_42','Object_44','Object_46','Object_50','Object_52','Object_86','Object_88','Object_90','Object_94','Object_96','Object_104','Object_110']){
  const g=converted(scene.getObjectByName(name));assert(volume(g)>0,`${name}: outward closed shell`);g.dispose()
 }
})

test('all decoded/baked F222 triangles have finite positions, unit normals and matching winding',()=>{
 let total=0,bad=0
 scene.traverse(mesh=>{if(!mesh.isMesh)return;const g=converted(mesh)
  triangles(g,(a,b,c,n)=>{assert([...a,...b,...c,...n].every(Number.isFinite));assert(Math.abs(n.length()-1)<.002);const cross=b.sub(a).cross(c.sub(a));if(cross.dot(n)<-1e-9)bad++;total++});g.dispose()
 })
 assert.equal(total,59884,'repair must not remove original geometry')
 assert.equal(bad,0,'no inward lighting normals relative to triangle winding')
})

test('mirrored transforms preserve front faces for both indexed and non-indexed geometry',()=>{
 for(const indexed of [true,false]){
  let g=new THREE.BoxGeometry(2,1,3);if(!indexed)g=g.toNonIndexed()
  orientCarGeometry(g,'unrelated',new THREE.Matrix4().makeScale(-1,1,1))
  assert(volume(g)>0)
  let wrong=0;triangles(g,(a,b,c,n)=>{if(b.sub(a).cross(c.sub(a)).dot(n)<0)wrong++});assert.equal(wrong,0);g.dispose()
 }
})

test('historical assets remain unchanged by modern source-specific repairs',async()=>{
 for(const file of ['15176','15173','15177','18685']){
  const b=readFileSync(new URL(`../public/models/historical/${file}.glb`,import.meta.url));const {scene}=await new GLTFLoader().parseAsync(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),'');scene.updateMatrixWorld(true)
  scene.traverse(o=>{if(!o.isMesh)return;const g=converted(o,false);assert(g.attributes.position.count>0);assert.equal(o.material.transparent,false);assert.equal(o.material.opacity,1);g.dispose()})
 }
})

test('source-specific repairs are pinned to the audited binary and existing mesh names',()=>{
 const record=JSON.parse(readFileSync(new URL('../public/models/formula-2022-source.json',import.meta.url),'utf8'))
 assert.equal(createHash('sha256').update(bytes).digest('hex'),record.sha256)
 for(const names of Object.values(F222_REPAIRS))for(const name of names)assert(scene.getObjectByName(name),name)
})
