// Procedural sky for the CDU-100 twin: gradient atmosphere + sun disc/halo + drifting fbm clouds lit by the sun.
// Presets: "sunset" (golden hour, like a refinery photo at dusk), "day", "night".
// The same sky is captured into an environment map so steel and cladding reflect its colours.
import * as THREE from "three";

const PRESETS = {
  sunset: {
    sunElev: 5, sunAzim: 52,                       // degrees; low sun, behind/side of the default view
    zenith: "#26314a", mid: "#7f7a8e", horizon: "#ffa851", sun: "#ffbd66",
    cloudLit: "#ff9a48", cloudMid: "#5a5a6e", cloudShadow: "#1f2534", cover: 0.38, cloudDensity: 1.0,
    keyColor: "#ffb067", keyIntensity: 2.6, hemiSky: "#8d9ab4", hemiGround: "#4a3426", hemiIntensity: 0.75,
    envIntensity: 0.6, fog: "#544b55", fogNear: 220, fogFar: 760, exposure: 0.85, ground: "#2e2a29",
  },
  day: {
    sunElev: 38, sunAzim: 140,
    zenith: "#3d74b8", mid: "#86aed6", horizon: "#d6e4ef", sun: "#fff3dc",
    cloudLit: "#ffffff", cloudMid: "#dfe5ee", cloudShadow: "#7d8a9c", cover: 0.47, cloudDensity: 0.95,
    keyColor: "#fff1dc", keyIntensity: 2.2, hemiSky: "#b9d2ec", hemiGround: "#4a443c", hemiIntensity: 0.8,
    envIntensity: 0.55, fog: "#b3c4d4", fogNear: 260, fogFar: 900, exposure: 0.85, ground: "#4a4844",
  },
  night: {
    sunElev: 30, sunAzim: 220,
    zenith: "#04070c", mid: "#0a111b", horizon: "#16202d", sun: "#9fb4d6",
    cloudLit: "#2a3446", cloudMid: "#141c28", cloudShadow: "#0b1018", cover: 0.55, cloudDensity: 0.7,
    keyColor: "#9fb4d6", keyIntensity: 0.55, hemiSky: "#5a6f8f", hemiGround: "#1a1612", hemiIntensity: 0.9,
    envIntensity: 0.35, fog: "#0d1218", fogNear: 230, fogFar: 620, exposure: 1.05, ground: "#1c1d20",
  },
};

const vert = /* glsl */`
  varying vec3 vDir;
  void main() {
    vDir = normalize(position);
    vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_Position = p.xyww;            // keep the dome at the far plane
  }`;

const frag = /* glsl */`
  uniform vec3 sunDir, zenith, mid, horizon, sunCol, cloudLit, cloudMid, cloudShadow;
  uniform float time, cover, density, sunBoost;
  varying vec3 vDir;

  float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
  float noise(vec3 x){
    vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
  }
  float fbm(vec3 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 6; i++) { s += a * noise(p); p = p * 2.03 + 11.7; a *= 0.5; } return s; }

  void main() {
    vec3 d = normalize(vDir);
    float h = d.y;
    float sd = max(dot(d, sunDir), 0.0);

    // atmosphere gradient: horizon glow -> mid -> zenith, warmer toward the sun
    float t = smoothstep(-0.02, 0.45, h);
    vec3 col = mix(horizon, mid, smoothstep(0.0, 0.25, t));
    col = mix(col, zenith, smoothstep(0.3, 1.0, t));
    col += sunCol * (pow(sd, 8.0) * 0.35 + pow(sd, 64.0) * 0.6) * sunBoost;

    // clouds on a curved layer above the plant (domain-warped fbm -> billowy cumulus masses)
    if (h > -0.02) {
      vec2 uv = d.xz / (h + 0.12) * 0.55;
      vec3 p = vec3(uv * 2.4 + vec2(time * 0.010, time * 0.003), time * 0.01);
      vec3 w = vec3(fbm(p + 3.1), fbm(p + 7.7), 0.0);
      p += w * 0.9;
      float n = fbm(p);
      float c = smoothstep(cover, cover + 0.16, n) * density;
      // self-shadowing: denser toward the sun = darker underside, thinner = lit rim
      vec2 toSun = normalize(sunDir.xz + 1e-4) * 0.09;
      float n2 = fbm(p + vec3(toSun, 0.0));
      float lit = clamp((n - n2) * 6.0 + 0.35, 0.0, 1.0);
      float facing = pow(sd, 1.6);
      float thick = smoothstep(cover, cover + 0.45, n);
      vec3 cc = mix(cloudMid, cloudShadow, thick);              // thick cores go dark slate
      cc = mix(cc, cloudLit, lit * (0.08 + 0.7 * facing) * (1.0 - 0.85 * thick));
      cc += sunCol * pow(sd, 14.0) * (1.0 - thick) * 1.2 * sunBoost;   // glowing edges near the sun
      c *= smoothstep(-0.015, 0.035, h);
      col = mix(col, cc, c);
    }

    // sun disc (drawn last so clouds can partly cover it via the halo above)
    col += sunCol * smoothstep(0.99955, 0.99985, sd) * 6.0 * sunBoost;
    // below horizon: dark distant ground haze
    col = mix(col, horizon * 0.35, smoothstep(0.0, -0.08, h));
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }`;

export function createSky(renderer, scene, { key, hemi, groundMesh } = {}) {
  const uniforms = {
    sunDir: { value: new THREE.Vector3() }, time: { value: 0 }, cover: { value: 0.45 }, density: { value: 0.9 }, sunBoost: { value: 1 },
    zenith: { value: new THREE.Color() }, mid: { value: new THREE.Color() }, horizon: { value: new THREE.Color() }, sunCol: { value: new THREE.Color() },
    cloudLit: { value: new THREE.Color() }, cloudMid: { value: new THREE.Color() }, cloudShadow: { value: new THREE.Color() },
  };
  const dome = new THREE.Mesh(new THREE.SphereGeometry(1000, 64, 32),
    new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag, side: THREE.BackSide, depthWrite: false, fog: false }));
  dome.frustumCulled = false; dome.renderOrder = -1;
  scene.add(dome);

  const pmrem = new THREE.PMREMGenerator(renderer);
  let envRT = null, current = null;

  function bakeEnv() {
    const envScene = new THREE.Scene();
    const copy = new THREE.Mesh(dome.geometry, dome.material);
    copy.scale.setScalar(0.05);                     // inside the cube camera's far plane
    envScene.add(copy);
    envRT?.dispose();
    envRT = pmrem.fromScene(envScene, 0.03);
    scene.environment = envRT.texture;
  }

  function apply(name) {
    const p = PRESETS[name] || PRESETS.sunset;
    current = name;
    const el = THREE.MathUtils.degToRad(p.sunElev), az = THREE.MathUtils.degToRad(p.sunAzim);
    uniforms.sunDir.value.set(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)).normalize();
    for (const k of ["zenith", "mid", "horizon", "cloudLit", "cloudMid", "cloudShadow"]) uniforms[k].value.set(p[k]);
    uniforms.sunCol.value.set(p.sun);
    uniforms.cover.value = p.cover; uniforms.density.value = p.cloudDensity;
    uniforms.sunBoost.value = name === "night" ? 0.15 : 1.0;
    if (key) {
      key.color.set(p.keyColor); key.intensity = p.keyIntensity;
      key.position.copy(uniforms.sunDir.value).multiplyScalar(260);
      // low sun -> keep shadows from stretching off the map
      key.position.y = Math.max(key.position.y, 40);
    }
    if (hemi) { hemi.color.set(p.hemiSky); hemi.groundColor.set(p.hemiGround); hemi.intensity = p.hemiIntensity; }
    if (groundMesh) groundMesh.material.color.set(p.ground);
    scene.background = null;
    scene.fog = new THREE.Fog(p.fog, p.fogNear, p.fogFar);
    scene.environmentIntensity = p.envIntensity;
    renderer.toneMappingExposure = p.exposure;
    bakeEnv();
  }

  return {
    apply,
    get preset() { return current; },
    update(t, camera) { uniforms.time.value = t; if (camera) dome.position.copy(camera.position); },
    presets: Object.keys(PRESETS),
  };
}
