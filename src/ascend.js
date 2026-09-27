import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { GammaCorrectionShader } from 'three/examples/jsm/shaders/GammaCorrectionShader.js';
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import Lenis from 'lenis';

// Year update
const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = new Date().getFullYear();

// CONFIG verbatim
const CONFIG = {
  rimColor: '#c1faff', rimPower: 2.4, nightLights: 10, terrainDepth: 0.33, terrainShade: 1.3,
  oceanGlint: 0.45, oceanDeep: 0.12, oceanFlow: 3, oceanFlowSpeed: 0.8, oceanFlowScale: 2.1,
  glowColor: '#3a6cff', glowIntensity: 3.35, planetRadius: 1.95, spin: 0.03, initRotation: 2.07,
  tilt: 0.37, autoRotate: 0,
  cloud1Height: 1.005, cloud1Opacity: 0.6, cloud1Spin: 0.06,
  cloud2Height: 1.03, cloud2Opacity: 0.5, cloud2Spin: 0.14,
  cloud3Height: 1.075, cloud3Opacity: 0.5, cloud3Spin: 0.1,
  bgColor: '#040a1e', flameColor: '#3a6cff', flameColor2: '#c1faff', flameAmt: 0.15,
  atmoColor: '#9fc4ff', atmoCount: 320, atmoSize: 22, atmoSpeed: 0.8,
  starColor: '#cfe0ff', starCount: 1400, starSize: 1.6, starFlicker: 1,
  markerColor: '#ffd27a', markerCount: 60, markerSize: 16, markerSpeed: 0.5,
};

const LAYERS = { NONE: 0, TORUS_SCENE: 1, BLOOM_SCENE: 2, ENTIRE_SCENE: 3 };

const ASSET_BASE_URL = 'https://api.getlayers.ai/storage/v1/object/public/public/assets/ascend-d9857ad1f2';
const PLANET_GLB = `${ASSET_BASE_URL}/planet.glb`;
const PLANET_LIGHTS_GLB = `${ASSET_BASE_URL}/planet-lights.glb`;
const PLANET_CLOUDS_PNG = `${ASSET_BASE_URL}/planet-clouds.png`;

// Helpers
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const hexToVec3 = (hex) => {
  const c = new THREE.Color(hex);
  return new THREE.Vector3(c.r, c.g, c.b);
};

function sample(stops, p) {
  if (p <= stops[0].p) return stops[0].v;
  if (p >= stops[stops.length - 1].p) return stops[stops.length - 1].v;
  for (let i = 0; i < stops.length - 1; i++) {
    const s0 = stops[i];
    const s1 = stops[i + 1];
    if (p >= s0.p && p <= s1.p) {
      const t = (p - s0.p) / (s1.p - s0.p);
      const smoothT = t * t * (3 - 2 * t);
      return lerp(s0.v, s1.v, smoothT);
    }
  }
  return stops[0].v;
}

// Simplex noise GLSL verbatim
const SNOISE = `
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0); const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy)); vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz); vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy); vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + 1.0 * C.xxx; vec3 x2 = x0 - i2 + 2.0 * C.xxx; vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 1.0/7.0; vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z *ns.z);
  vec4 x_ = floor(j * ns.z); vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ *ns.x + ns.yyyy; vec4 y = y_ *ns.x + ns.yyyy; vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy); vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0; vec4 s1 = floor(b1)*2.0 + 1.0; vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy; vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy,h.x); vec3 p1 = vec3(a0.zw,h.y); vec3 p2 = vec3(a1.xy,h.z); vec3 p3 = vec3(a1.zw,h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0); m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
`;

// FinalPass shader verbatim
const FinalPass = {
  uniforms: {
    iTime: { value: 0 },
    tDiffuse: { value: null },
    torusTexture: { value: null },
    bloomTexture: { value: null },
    haloTexture: { value: null },
    uBg: { value: hexToVec3(CONFIG.bgColor) },
    uFlameA: { value: hexToVec3(CONFIG.flameColor) },
    uFlameB: { value: hexToVec3(CONFIG.flameColor2) },
    uFlameAmt: { value: CONFIG.flameAmt },
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position, 1.0); }`,
  fragmentShader: `
    uniform float iTime; uniform sampler2D tDiffuse; uniform sampler2D bloomTexture; uniform sampler2D torusTexture; uniform sampler2D haloTexture;
    uniform vec3 uBg; uniform vec3 uFlameA; uniform vec3 uFlameB; uniform float uFlameAmt;
    varying vec2 vUv;
    vec3 warp3d(vec3 pos, float t){ float curv=.8,a=1.9,b=0.7; pos*=2.;
      pos.x+=curv*sin(t+a*pos.y)+t*b; pos.y+=curv*cos(t+a*pos.x);
      pos.y+=curv*sin(t+a*pos.z)+t*b; pos.z+=curv*cos(t+a*pos.y);
      pos.z+=curv*sin(t+a*pos.x)+t*b; pos.x+=curv*cos(t+a*pos.z);
      return 0.5+0.5*cos(pos.xyz+vec3(1,2,4)); }
    void main(){
      vec2 uv = 2.*vUv - 1.;
      vec3 w = pow(warp3d(vec3(uv.x, sin(uv.y), uv.y), iTime*1.5), vec3(1.5));
      vec3 flame = 1.5*uFlameA*w.x; flame*=w.y; flame += uFlameB*w.z;
      flame *= smoothstep(0.25, 1., abs(uv.y));
      float md = smoothstep(-0.7, 1., -uv.y*uv.x); flame *= md*md;
      vec3 bg = uBg * (1.0 - 0.4 * length(uv));
      vec3 halo = texture2D(haloTexture, vUv).xyz;
      gl_FragColor = vec4(bg + flame*uFlameAmt + texture2D(bloomTexture, vUv).xyz + texture2D(torusTexture, vUv).xyz + texture2D(tDiffuse, vUv).xyz + halo, 1.);
    }
  `,
};

// Canvas & Renderer setup
const canvas = document.getElementById('planetCanvas');
let w = window.innerWidth;
let h = window.innerHeight;

const renderer = new THREE.WebGL1Renderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(w, h);
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.VSMShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 200);
camera.position.set(0, 0, 8);
camera.layers.enable(LAYERS.TORUS_SCENE);
camera.layers.enable(LAYERS.BLOOM_SCENE);
camera.layers.enable(LAYERS.ENTIRE_SCENE);
scene.add(camera);

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 1.8);
ambientLight.layers.set(LAYERS.ENTIRE_SCENE);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(0, 10, 2);
dirLight.layers.set(LAYERS.ENTIRE_SCENE);
scene.add(dirLight);

// Composers
const torusComposer = new EffectComposer(renderer);
torusComposer.renderToScreen = false;
torusComposer.addPass(new RenderPass(scene, camera));
torusComposer.addPass(new ShaderPass(GammaCorrectionShader));
const torusBloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.22, 0.2, 0);
torusComposer.addPass(torusBloomPass);
torusComposer.addPass(new ShaderPass(CopyShader));

const bloomComposer = new EffectComposer(renderer);
bloomComposer.renderToScreen = false;
bloomComposer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.5, 0.6, 0);
bloomComposer.addPass(bloomPass);
bloomComposer.addPass(new ShaderPass(GammaCorrectionShader));

const finalPass = new ShaderPass(FinalPass);
finalPass.needsSwap = true;
const finalComposer = new EffectComposer(renderer);
finalComposer.addPass(new RenderPass(scene, camera));
finalComposer.addPass(finalPass);

finalPass.uniforms.bloomTexture.value = bloomComposer.renderTarget1.texture;
finalPass.uniforms.torusTexture.value = torusComposer.renderTarget1.texture;

// OrbitControls (disabled from pointer interactions as per spec)
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.enablePan = false;
controls.enabled = false;
controls.autoRotate = true;
controls.autoRotateSpeed = CONFIG.autoRotate;
controls.minDistance = 3.5;
controls.maxDistance = 16;
controls.target.set(0, 0, 0);

// Keyframe stops for scroll storytelling
const STOPS_X = [{ p: 0, v: 0 }, { p: 0.32, v: -3.1 }, { p: 0.64, v: 3.2 }, { p: 1, v: 0 }];
const STOPS_Y = [{ p: 0, v: -4.5 }, { p: 0.32, v: 0.55 }, { p: 0.64, v: 0.45 }, { p: 1, v: 0.15 }];
const STOPS_S = [{ p: 0, v: 2.15 }, { p: 0.32, v: 1.0 }, { p: 0.64, v: 0.92 }, { p: 1, v: 1.12 }];

// Scene hierarchy
const worldGroup = new THREE.Group();
worldGroup.position.set(STOPS_X[0].v, STOPS_Y[0].v, 0);
worldGroup.scale.setScalar(STOPS_S[0].v);
scene.add(worldGroup);

const planetGroup = new THREE.Group();
planetGroup.rotation.z = CONFIG.tilt;
worldGroup.add(planetGroup);

const cloudGroup = new THREE.Group();
cloudGroup.rotation.z = CONFIG.tilt;
cloudGroup.visible = false;
worldGroup.add(cloudGroup);

// Atmosphere halo glow (additive billboard plane)
const glowGeo = new THREE.PlaneGeometry(2, 2);
const glowMat = new THREE.ShaderMaterial({
  transparent: true,
  side: THREE.DoubleSide,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  uniforms: {
    uGlow: { value: hexToVec3(CONFIG.glowColor) },
    uIntensity: { value: CONFIG.glowIntensity },
  },
  vertexShader: `
    varying vec2 vUv;
    void main(){
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform vec3 uGlow; uniform float uIntensity; varying vec2 vUv;
    void main(){
      float d = length(vUv - 0.5) * 2.0;
      float a = pow(clamp(1.0 - d, 0.0, 1.0), 2.2);
      gl_FragColor = vec4(uGlow * a * uIntensity, a);
    }
  `,
});
const glowMesh = new THREE.Mesh(glowGeo, glowMat);
glowMesh.scale.setScalar(CONFIG.planetRadius * 2.3);
glowMesh.layers.set(LAYERS.ENTIRE_SCENE);
glowMesh.layers.enable(LAYERS.BLOOM_SCENE);
worldGroup.add(glowMesh);

// Ambient atmosphere motes (Points)
const atmoGeo = new THREE.BufferGeometry();
const atmoPos = new Float32Array(CONFIG.atmoCount * 3);
const atmoSize = new Float32Array(CONFIG.atmoCount);
const atmoSeed = new Float32Array(CONFIG.atmoCount);
for (let i = 0; i < CONFIG.atmoCount; i++) {
  atmoPos[i * 3 + 0] = (Math.random() - 0.5) * 2;
  atmoPos[i * 3 + 1] = (Math.random() - 0.5) * 2;
  atmoPos[i * 3 + 2] = (Math.random() - 0.5) * 2;
  atmoSize[i] = CONFIG.atmoSize * (0.4 + Math.random());
  atmoSeed[i] = Math.random() * 100.0;
}
atmoGeo.setAttribute('position', new THREE.BufferAttribute(atmoPos, 3));
atmoGeo.setAttribute('size', new THREE.BufferAttribute(atmoSize, 1));
atmoGeo.setAttribute('seed', new THREE.BufferAttribute(atmoSeed, 1));

const atmoMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  depthTest: false,
  blending: THREE.AdditiveBlending,
  uniforms: {
    uTime: { value: 0 },
    uRes: { value: new THREE.Vector2(w, h) },
    uColor: { value: hexToVec3(CONFIG.atmoColor) },
  },
  vertexShader: `
    attribute float size; attribute float seed; uniform float uTime; uniform vec2 uRes;
    varying float vA;
    vec3 warp(vec3 p, float t){ float c=0.9,a=1.9,b=0.02,s=0.05; p*=2.;
      p.x+=c*sin(s*t+a*p.y)+t*b; p.y+=c*cos(s*t+a*p.x); p.y+=c*sin(s*t+a*p.z)+t*b;
      p.z+=c*cos(s*t+a*p.y); p.z+=c*sin(s*t+a*p.x)+t*b; p.x+=c*cos(s*t+a*p.z);
      return cos(p+vec3(1,2,4)); }
    void main(){
      vec3 v = position*4.0 + warp(position, uTime)*1.2;
      vec4 mv = modelViewMatrix * vec4(v, 1.0);
      float r = length(v); float farF = 1.0 - smoothstep(5.0, 6.5, r); float nearF = smoothstep(0.0, 0.5, -mv.z);
      vA = farF * nearF;
      gl_PointSize = size * uRes.y / 900.0 / -mv.z; gl_PointSize = max(gl_PointSize, 1.0);
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: `
    uniform vec3 uColor; varying float vA;
    void main(){ vec2 p = gl_PointCoord - 0.5; float l = length(p); if (l > 0.5) discard;
      float tex = smoothstep(0.5, 0.0, l); gl_FragColor = vec4(uColor * tex, tex * vA * 0.55); }
  `,
});
const atmoPoints = new THREE.Points(atmoGeo, atmoMat);
atmoPoints.layers.set(LAYERS.ENTIRE_SCENE);
atmoPoints.onBeforeRender = () => {
  const perfNow = performance.now() * 0.001;
  atmoMat.uniforms.uTime.value = perfNow * CONFIG.atmoSpeed * 8;
  atmoPoints.position.copy(camera.position);
  finalPass.uniforms.iTime.value = perfNow;
};
scene.add(atmoPoints);

// Starfield (Points on sphere, r=90)
const starGeo = new THREE.BufferGeometry();
const starPos = new Float32Array(CONFIG.starCount * 3);
const starSeed = new Float32Array(CONFIG.starCount);
const starBright = new Float32Array(CONFIG.starCount);
for (let i = 0; i < CONFIG.starCount; i++) {
  const u = Math.random();
  const v = Math.random();
  const theta = u * 2.0 * Math.PI;
  const phi = Math.acos(2.0 * v - 1.0);
  const r = 90.0;
  const sinPhi = Math.sin(phi);
  starPos[i * 3 + 0] = r * sinPhi * Math.cos(theta);
  starPos[i * 3 + 1] = r * sinPhi * Math.sin(theta);
  starPos[i * 3 + 2] = r * Math.cos(phi);
  starSeed[i] = Math.random() * 100.0;
  starBright[i] = 0.35 + Math.random() * 0.65;
}
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
starGeo.setAttribute('seed', new THREE.BufferAttribute(starSeed, 1));
starGeo.setAttribute('bright', new THREE.BufferAttribute(starBright, 1));

let starTime = 0;
const starMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  depthTest: true,
  blending: THREE.AdditiveBlending,
  uniforms: {
    uTime: { value: 0 },
    uSize: { value: CONFIG.starSize },
    uFlicker: { value: CONFIG.starFlicker },
    uColor: { value: hexToVec3(CONFIG.starColor) },
    uRes: { value: new THREE.Vector2(w, h) },
  },
  vertexShader: `
    attribute float seed; attribute float bright;
    uniform float uTime; uniform float uSize; uniform float uFlicker; uniform vec2 uRes;
    varying float vTw;
    void main(){
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float tw = 0.6 + 0.4 * sin(uTime * uFlicker + seed);
      vTw = bright * tw;
      gl_PointSize = max(uSize * uRes.y / 900.0 * (90.0 / max(-mv.z, 1.0)), 1.0);
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: `
    uniform vec3 uColor; varying float vTw;
    void main(){
      vec2 p = gl_PointCoord - 0.5; float l = length(p); if (l > 0.5) discard;
      float core = smoothstep(0.5, 0.0, l);
      gl_FragColor = vec4(uColor, core * vTw);
    }
  `,
});
const starMesh = new THREE.Points(starGeo, starMat);
starMesh.frustumCulled = false;
starMesh.layers.set(LAYERS.ENTIRE_SCENE);
scene.add(starMesh);

// Cloud shells
let cloudTime = 0;
const cloudUniformsList = [];
const cloudMeshList = [];
const textureLoader = new THREE.TextureLoader();

textureLoader.load(PLANET_CLOUDS_PNG, (cloudTex) => {
  cloudTex.wrapS = THREE.RepeatWrapping;
  cloudTex.wrapT = THREE.RepeatWrapping;
  cloudTex.repeat.set(5, 5);

  const cloudLayers = [
    { h: CONFIG.cloud1Height, o: CONFIG.cloud1Opacity, s: CONFIG.cloud1Spin, ry: 0.0, phase: 0.0 },
    { h: CONFIG.cloud2Height, o: CONFIG.cloud2Opacity, s: CONFIG.cloud2Spin, ry: 2.2, phase: 13.0 },
    { h: CONFIG.cloud3Height, o: CONFIG.cloud3Opacity, s: CONFIG.cloud3Spin, ry: 4.3, phase: 27.0 },
  ];

  cloudLayers.forEach((layer) => {
    const geo = new THREE.SphereGeometry(CONFIG.planetRadius * layer.h, 64, 64);
    const mat = new THREE.MeshStandardMaterial({
      map: cloudTex,
      transparent: true,
      depthWrite: false,
    });

    const uniforms = {
      uTime: { value: 0 },
      noiseScale: { value: 20 },
      uSpeedX: { value: 1.0 },
      uSpeedY: { value: 2.0 },
      uSpeedZ: { value: 2.0 },
      uOpacity: { value: layer.o },
      uPhase: { value: layer.phase },
    };
    cloudUniformsList.push(uniforms);

    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = `varying vec2 vCloudUv;\n` + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        'void main() {',
        'void main() {\n  vCloudUv = uv;'
      );

      shader.fragmentShader = `
        uniform float uTime;
        uniform float noiseScale;
        uniform float uSpeedX;
        uniform float uSpeedY;
        uniform float uSpeedZ;
        uniform float uOpacity;
        uniform float uPhase;
        varying vec2 vCloudUv;
        ${SNOISE}
      ` + shader.fragmentShader;

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <dithering_fragment>',
        `
        #include <dithering_fragment>
        gl_FragColor.rgb = vec3(1.0);
        float cloudNoise = snoise(vec3(vCloudUv.x * noiseScale + uTime * uSpeedX + uPhase, vCloudUv.y * noiseScale - uTime * uSpeedY + uPhase, uTime * uSpeedZ + uPhase));
        float cloudNdv = max(dot(normalize(vNormal), normalize(vViewPosition)), 0.0);
        float cloudEdge = pow(1.0 - cloudNdv, 3.0);
        float cloudMod = mix(cloudNoise, 1.0, cloudEdge);
        float cloudNdl = dot(normalize(vNormal), normalize(vec3(-0.9, 0.18, 0.4)));
        float cloudDay = 1.0 - smoothstep(0.30, -0.30, cloudNdl) * 0.9;
        gl_FragColor.a *= cloudMod * uOpacity * cloudDay;
        `
      );
    };

    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.y = layer.ry;
    mesh.renderOrder = 2;
    mesh.layers.set(LAYERS.ENTIRE_SCENE);
    cloudGroup.add(mesh);
    cloudMeshList.push({ mesh, spin: layer.s });
  });
});

// Load planet & night lights
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.5/');
const gltfLoader = new GLTFLoader();
gltfLoader.setDRACOLoader(dracoLoader);

let planetTime = 0;
let planetMesh = null;
let planetUniforms = null;
let markerUniforms = null;
let markerTime = 0;
let entryStartTime = 0;

gltfLoader.load(PLANET_LIGHTS_GLB, (gltfLights) => {
  let nightTex = null;
  gltfLights.scene.traverse((child) => {
    if (child.isMesh && child.material && child.material.map) {
      nightTex = child.material.map;
    }
  });

  gltfLoader.load(PLANET_GLB, (gltfPlanet) => {
    let baseMesh = null;
    gltfPlanet.scene.traverse((child) => {
      if (!baseMesh && child.isMesh) {
        baseMesh = child;
      }
    });

    if (!baseMesh) return;
    planetMesh = baseMesh;
    baseMesh.geometry.computeBoundingSphere();
    const sphereRadius = baseMesh.geometry.boundingSphere.radius;
    const meshScale = CONFIG.planetRadius / sphereRadius;
    baseMesh.scale.setScalar(meshScale);

    const mat = baseMesh.material.clone();
    mat.metalness = 0;
    mat.roughness = 1;
    mat.envMapIntensity = 0;

    planetUniforms = {
      time: { value: 0 },
      noiseScale: { value: 30 },
      speedX: { value: 1.5 },
      speedY: { value: 2.0 },
      speedZ: { value: 2.5 },
      rimColor: { value: hexToVec3(CONFIG.rimColor) },
      rimPower: { value: CONFIG.rimPower },
      nightBlendTexture: { value: nightTex },
      nightLights: { value: CONFIG.nightLights },
      terrainDepth: { value: CONFIG.terrainDepth },
      terrainShade: { value: CONFIG.terrainShade },
      oceanGlint: { value: CONFIG.oceanGlint },
      oceanDeep: { value: CONFIG.oceanDeep },
      oceanFlow: { value: CONFIG.oceanFlow },
      oceanFlowSpeed: { value: CONFIG.oceanFlowSpeed },
      oceanFlowScale: { value: CONFIG.oceanFlowScale },
    };

    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, planetUniforms);
      shader.vertexShader = `varying vec2 vCustomUv;\n` + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        'void main() {',
        'void main() {\n  vCustomUv = uv;'
      );

      shader.fragmentShader = `
        uniform float time;
        uniform float noiseScale;
        uniform float speedX;
        uniform float speedY;
        uniform float speedZ;
        uniform vec3 rimColor;
        uniform float rimPower;
        uniform sampler2D nightBlendTexture;
        uniform float nightLights;
        uniform float terrainDepth;
        uniform float terrainShade;
        uniform float oceanGlint;
        uniform float oceanDeep;
        uniform float oceanFlow;
        uniform float oceanFlowSpeed;
        uniform float oceanFlowScale;
        varying vec2 vCustomUv;
        ${SNOISE}
      ` + shader.fragmentShader;

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <dithering_fragment>',
        `
        #include <dithering_fragment>
        vec3 normalizedNormal = normalize(vNormal);
        vec3 viewDir = normalize(vViewPosition);
        float rim = 1.0 - max(dot(viewDir, normalizedNormal), 0.0);
        rim = pow(rim, rimPower); rim = pow(rim, 1.5); rim *= 0.7;
        vec3 currentColor = gl_FragColor.rgb;
        float blueDom = currentColor.b - max(currentColor.r, currentColor.g);
        float waterMask = clamp(smoothstep(-0.005, 0.03, blueDom), 0.0, 1.0);
        float shimmer = snoise(vec3(vCustomUv.x * noiseScale + time * speedX, vCustomUv.y * noiseScale - time * speedY, time * speedZ));
        gl_FragColor.rgb += waterMask * shimmer * 0.025;
        float fT = time * oceanFlowSpeed * 4.0;
        float fS = 4.0 * oceanFlowScale;
        float warp = snoise(vec3(vCustomUv.x * fS - fT * 0.5, vCustomUv.y * fS + fT * 0.4, fT * 0.5));
        float flow = snoise(vec3(vCustomUv.x * fS * 2.0 + fT * 0.6 + warp, vCustomUv.y * fS * 2.0 - fT * 0.5, fT * 0.7));
        flow = warp * 0.6 + flow * 0.4;
        gl_FragColor.rgb += waterMask * flow * 0.12 * oceanFlow;
        gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.01, 0.06, 0.16), waterMask * oceanDeep);
        vec3 finalColor = mix(gl_FragColor.rgb, rimColor, rim);
        gl_FragColor = vec4(finalColor, 1.0);
        vec3 surfPos = -vViewPosition;
        float terrH = dot(texture2D(map, vCustomUv).rgb, vec3(0.299, 0.587, 0.114));
        vec3 sigX = dFdx(surfPos), sigY = dFdy(surfPos);
        vec3 vR1 = cross(sigY, normalizedNormal), vR2 = cross(normalizedNormal, sigX);
        float fDet = dot(sigX, vR1);
        vec3 vGrad = sign(fDet) * (dFdx(terrH) * vR1 + dFdy(terrH) * vR2);
        vec3 bumpedNormal = normalize(abs(fDet) * normalizedNormal - terrainDepth * vGrad);
        vec3 shadeNormal = mix(bumpedNormal, normalizedNormal, waterMask);
        vec3 cityLights = texture2D(nightBlendTexture, vCustomUv).rgb * gl_FragColor.rgb * nightLights;
        vec3 viewSunDir = normalize(vec3(-0.9, 0.18, 0.4));
        float ndl = dot(normalizedNormal, viewSunDir);
        float dayAmt = smoothstep(-0.05, 0.35, ndl);
        float relief = dot(shadeNormal, viewSunDir) - ndl;
        gl_FragColor.rgb *= clamp(1.0 + relief * terrainShade * dayAmt, 0.55, 1.6);
        float nightFactor  = smoothstep(0.18, -0.30, ndl);
        float lightsFactor = smoothstep(0.30, -0.35, ndl);
        gl_FragColor.rgb = mix(gl_FragColor.rgb, gl_FragColor.rgb * 0.08, nightFactor);
        gl_FragColor.rgb += cityLights * lightsFactor;
        vec3 halfDir = normalize(viewSunDir + viewDir);
        float ripple = snoise(vec3(vCustomUv * 240.0, time * 4.0));
        float ndh = max(dot(normalizedNormal, halfDir) + ripple * 0.02, 0.0);
        float glint = pow(ndh, 140.0);
        gl_FragColor.rgb += glint * waterMask * dayAmt * oceanGlint * vec3(1.0, 0.97, 0.88);
        `
      );
    };

    baseMesh.material = mat;
    baseMesh.layers.set(LAYERS.ENTIRE_SCENE);
    baseMesh.layers.enable(LAYERS.BLOOM_SCENE);
    planetGroup.add(baseMesh);

    // Land markers generation
    try {
      createLandMarkers(baseMesh);
    } catch (e) {
      console.warn('Land markers sampling error:', e);
    }

    // Enable clouds and start entry animation
    cloudGroup.visible = true;
    entryStartTime = performance.now();
  });
});

// Sampling land markers on planet mesh
function createLandMarkers(mesh) {
  const geo = mesh.geometry;
  const posAttr = geo.attributes.position;
  const uvAttr = geo.attributes.uv;
  const index = geo.index;
  if (!posAttr || !uvAttr) return;

  const count = CONFIG.markerCount;
  const markerPos = new Float32Array(count * 3);
  const markerSeed = new Float32Array(count);

  let mapImg = null;
  let canvas2d = null;
  let ctx2d = null;
  let imgData = null;

  if (mesh.material.map && mesh.material.map.image) {
    mapImg = mesh.material.map.image;
    canvas2d = document.createElement('canvas');
    canvas2d.width = mapImg.width || 512;
    canvas2d.height = mapImg.height || 256;
    ctx2d = canvas2d.getContext('2d');
    if (ctx2d) {
      ctx2d.drawImage(mapImg, 0, 0, canvas2d.width, canvas2d.height);
      try {
        imgData = ctx2d.getImageData(0, 0, canvas2d.width, canvas2d.height);
      } catch (err) {
        imgData = null;
      }
    }
  }

  const numFaces = index ? index.count / 3 : posAttr.count / 3;
  let placed = 0;
  let attempts = 0;

  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const uvA = new THREE.Vector2();
  const uvB = new THREE.Vector2();
  const uvC = new THREE.Vector2();

  while (placed < count && attempts < 3000) {
    attempts++;
    const faceIdx = Math.floor(Math.random() * numFaces);
    const i0 = index ? index.getX(faceIdx * 3) : faceIdx * 3;
    const i1 = index ? index.getX(faceIdx * 3 + 1) : faceIdx * 3 + 1;
    const i2 = index ? index.getX(faceIdx * 3 + 2) : faceIdx * 3 + 2;

    vA.fromBufferAttribute(posAttr, i0);
    vB.fromBufferAttribute(posAttr, i1);
    vC.fromBufferAttribute(posAttr, i2);

    uvA.fromBufferAttribute(uvAttr, i0);
    uvB.fromBufferAttribute(uvAttr, i1);
    uvC.fromBufferAttribute(uvAttr, i2);

    // Barycentric coordinates
    const r1 = Math.random();
    const r2 = Math.random();
    const sqrtR1 = Math.sqrt(r1);
    const u = 1 - sqrtR1;
    const v = r2 * sqrtR1;
    const w = 1 - u - v;

    const sampleU = u * uvA.x + v * uvB.x + w * uvC.x;
    const sampleV = u * uvA.y + v * uvB.y + w * uvC.y;

    if (imgData) {
      const px = Math.floor(((sampleU % 1 + 1) % 1) * imgData.width);
      const py = Math.floor(((1 - (sampleV % 1 + 1) % 1)) * imgData.height);
      const pIdx = (py * imgData.width + px) * 4;
      const cr = imgData.data[pIdx];
      const cg = imgData.data[pIdx + 1];
      const cb = imgData.data[pIdx + 2];
      // reject if blue dominates: cb > cr+6 && cb > cg+6
      if (cb > cr + 6 && cb > cg + 6) continue;
    }

    const p = new THREE.Vector3()
      .addScaledVector(vA, u)
      .addScaledVector(vB, v)
      .addScaledVector(vC, w)
      .multiplyScalar(1.012);

    markerPos[placed * 3 + 0] = p.x;
    markerPos[placed * 3 + 1] = p.y;
    markerPos[placed * 3 + 2] = p.z;
    markerSeed[placed] = Math.random() * 10.0;
    placed++;
  }

  const markerGeo = new THREE.BufferGeometry();
  markerGeo.setAttribute('position', new THREE.BufferAttribute(markerPos, 3));
  markerGeo.setAttribute('seed', new THREE.BufferAttribute(markerSeed, 1));

  markerUniforms = {
    uTime: { value: 0 },
    uSpeed: { value: CONFIG.markerSpeed },
    uColor: { value: hexToVec3(CONFIG.markerColor) },
    uSize: { value: CONFIG.markerSize },
    uRes: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
  };

  const markerMat = new THREE.ShaderMaterial({
    transparent: true,
    depthTest: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: markerUniforms,
    vertexShader: `
      attribute float seed; uniform float uSize; uniform vec2 uRes;
      varying float vSeed; varying float vFade;
      void main(){
        vSeed = seed;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vec3 vn = normalize(normalMatrix * normalize(position));
        vec3 vd = normalize(-mv.xyz);
        vFade = smoothstep(0.15, 0.5, dot(vn, vd));
        gl_PointSize = max(uSize * uRes.y / 900.0 * (7.0 / max(-mv.z, 1.0)), 2.0);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor; uniform float uTime; uniform float uSpeed;
      varying float vSeed; varying float vFade;
      void main(){
        if (vFade <= 0.001) discard;
        vec2 p = gl_PointCoord - 0.5;
        float d = length(p) * 2.0;
        if (d > 1.0) discard;
        float core = smoothstep(0.30, 0.0, d) * 1.2;
        float ph = fract(uTime * uSpeed + vSeed);
        float ring = smoothstep(0.07, 0.0, abs(d - ph)) * (1.0 - ph);
        gl_FragColor = vec4(uColor, clamp(core + ring, 0.0, 1.0) * vFade);
      }
    `,
  });

  const markerPoints = new THREE.Points(markerGeo, markerMat);
  markerPoints.layers.set(LAYERS.ENTIRE_SCENE);
  mesh.add(markerPoints);
}

// Scroll animation state
let curP = 0;
let curX = STOPS_X[0].v;
let curY = STOPS_Y[0].v;
let curS = STOPS_S[0].v;
let spinPhase = 0;

const ENTRY_DUR = 1.9;
const ENTRY_START_Y = -6.5;

// Animation loop
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);

  const dt = Math.min(clock.getDelta(), 0.05);

  planetTime += dt / 12;
  cloudTime += dt / 20;
  starTime += dt;
  markerTime += dt;

  if (planetUniforms) planetUniforms.time.value = planetTime;
  cloudUniformsList.forEach((u) => { u.uTime.value = cloudTime; });
  starMat.uniforms.uTime.value = starTime;
  if (markerUniforms) markerUniforms.uTime.value = markerTime;

  cloudMeshList.forEach(({ mesh, spin }) => {
    mesh.rotation.y += dt * spin;
  });

  // Scroll tracking
  const scrollMax = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  const pTarget = clamp(window.scrollY / scrollMax, 0, 1);
  curP += (pTarget - curP) * Math.min(1, dt * 4.5);

  const sideScale = clamp(window.innerWidth / 1200, 0.5, 1);
  const targetX = sample(STOPS_X, curP) * sideScale;
  const targetY = sample(STOPS_Y, curP);
  const targetS = sample(STOPS_S, curP);

  const k = Math.min(1, dt * 3.2);
  curX += (targetX - curX) * k;
  curY += (targetY - curY) * k;
  curS += (targetS - curS) * k;

  // One-time float-up entrance
  let entryY = 0;
  if (entryStartTime > 0) {
    const elapsed = (performance.now() - entryStartTime) / 1000;
    const t = clamp(elapsed / ENTRY_DUR, 0, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    entryY = ENTRY_START_Y * (1 - ease);
  } else {
    entryY = ENTRY_START_Y;
  }

  worldGroup.position.set(curX, curY + entryY, 0);
  worldGroup.scale.setScalar(curS);

  planetGroup.rotation.y = CONFIG.initRotation + spinPhase + curP * Math.PI * 1.6;
  spinPhase += dt * CONFIG.spin;

  // Billboard glow to camera
  glowMesh.quaternion.copy(camera.quaternion);

  // Render composer passes
  camera.layers.set(LAYERS.TORUS_SCENE);
  torusComposer.render();

  camera.layers.set(LAYERS.BLOOM_SCENE);
  bloomComposer.render();

  finalPass.uniforms.bloomTexture.value = bloomComposer.readBuffer.texture || bloomComposer.renderTarget1.texture;
  finalPass.uniforms.torusTexture.value = torusComposer.readBuffer.texture || torusComposer.renderTarget1.texture;

  camera.layers.set(LAYERS.ENTIRE_SCENE);
  finalComposer.render();
}

animate();

// Resize handler
function onWindowResize() {
  w = window.innerWidth;
  h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  renderer.setSize(w, h);
  torusComposer.setSize(w, h);
  bloomComposer.setSize(w, h);
  finalComposer.setSize(w, h);

  const resVec = new THREE.Vector2(w, h);
  atmoMat.uniforms.uRes.value.copy(resVec);
  starMat.uniforms.uRes.value.copy(resVec);
  if (markerUniforms) markerUniforms.uRes.value.copy(resVec);
  torusBloomPass.resolution.copy(resVec);
  bloomPass.resolution.copy(resVec);
}
window.addEventListener('resize', onWindowResize, false);

// Lenis smooth scroll
const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (!prefersReduced && typeof Lenis !== 'undefined') {
  try {
    const lenis = new Lenis({
      duration: 1.15,
      smoothWheel: true,
      touchMultiplier: 1.5,
    });
    function lenisRaf(time) {
      lenis.raf(time);
      requestAnimationFrame(lenisRaf);
    }
    requestAnimationFrame(lenisRaf);
  } catch (e) {
    console.warn('Lenis initialization note:', e);
  }
}

// Scroll reveal observer
const revealElements = document.querySelectorAll('[data-reveal]');
if (prefersReduced) {
  revealElements.forEach((el) => el.classList.add('in'));
} else {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in');
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.18,
    rootMargin: '0px 0px -8% 0px',
  });
  revealElements.forEach((el) => observer.observe(el));
}
