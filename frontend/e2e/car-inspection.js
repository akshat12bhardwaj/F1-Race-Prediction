import * as THREE from 'three'
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js'
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js'
import {MeshoptDecoder} from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import {createCar} from '../src/car.js'
import {CAR_ASSETS} from '../src/carAssets.js'
const params=new URLSearchParams(location.search), asset=CAR_ASSETS[params.get('asset')||'groundEffect']
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(1);document.body.append(renderer.domElement)
const scene=new THREE.Scene();scene.background=new THREE.Color('#41474e');scene.add(new THREE.HemisphereLight('#ffffff','#647383',2));const sun=new THREE.DirectionalLight('#fff8f0',3);sun.position.set(3,6,4);scene.add(sun)
const camera=new THREE.PerspectiveCamera(35,innerWidth/innerHeight,.02,100);const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,.55,0)
let car
if(params.has('raw')){const g=await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(asset.url);car=g.scene;car.rotation.y=asset.rotation;car.updateMatrixWorld(true);const b=new THREE.Box3().setFromObject(car),sz=b.getSize(new THREE.Vector3()),k=Math.min(asset.length/sz.z,asset.width/sz.x),c=b.getCenter(new THREE.Vector3());const root=new THREE.Group();root.add(car);root.scale.setScalar(k);root.position.set(-c.x*k,-b.min.y*k,-c.z*k);car=root}
else{const years={turbo:1988,nineties:1994,v10:2004,halo:2020,groundEffect:2025};car=await createCar({constructor_id:'ferrari',constructor_season:years[params.get('asset')||'groundEffect']})}
scene.add(car);const floor=new THREE.Mesh(new THREE.PlaneGeometry(30,30),new THREE.MeshStandardMaterial({color:'#6c7278',roughness:1}));floor.rotation.x=-Math.PI/2;floor.position.y=-.015;scene.add(floor)
window.inspectCar={materials:()=>{const all=[];car.traverse(o=>{if(o.isMesh)all.push({side:o.material.side,transparent:o.material.transparent,opacity:o.material.opacity})});return all},view:(name)=>{const views={side:[8,2,0],front:[0,1.8,9],rear:[0,1.8,-9],threequarter:[6,4,7],top:[0,10,.001]};camera.position.fromArray(views[name]);controls.update();renderer.render(scene,camera)},stats:()=>renderer.info.render}
window.inspectCar.view(params.get('view')||'threequarter');window.ready=true;function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera)}loop()
