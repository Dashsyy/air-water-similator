/** WGSL for the render passes. Every pass shares one scene uniform and one bind group layout. */

export const SCENE_FLOATS = 80;

const COMMON = /* wgsl */ `
struct Scene {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  camPos: vec4<f32>,
  dims: vec4<f32>,      // nx, ny, nz, inflow speed
  viewport: vec4<f32>,  // width, height, line half-width in pixels, unused
  slice: vec4<f32>,     // axis, position in cells, field (0 speed, 1 pressure, 2 vorticity), opacity
  volume: vec4<f32>,    // smoke on, vortex on, Q threshold, smoke density
  mesh: vec4<f32>,      // pressure map on, Cp range, speed range, line colour mode
  streak: vec4<f32>,    // particles per seed, seed count, particle life, unused
  tint: vec4<f32>,      // colour for helper lines
};
@group(0) @binding(0) var<uniform> S: Scene;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var macroTex: texture_3d<f32>;
@group(0) @binding(3) var solidTex: texture_3d<f32>;
@group(0) @binding(4) var smokeTex: texture_3d<f32>;
@group(0) @binding(5) var vortTex: texture_3d<f32>;
@group(0) @binding(6) var<storage, read> parts: array<vec4<f32>>;

fn turbo(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  let v4 = vec4<f32>(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  return clamp(vec3<f32>(
    dot(v4, vec4<f32>(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2<f32>(-152.94239396, 59.28637943)),
    dot(v4, vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2<f32>(4.27729857, 2.82956604)),
    dot(v4, vec4<f32>(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2<f32>(-89.90310912, 27.34824973)),
  ), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Blue for suction, pale for free-stream pressure, red for high pressure. t in [-1, 1].
fn diverging(t: f32) -> vec3<f32> {
  let x = clamp(t, -1.0, 1.0);
  let mid = vec3<f32>(0.93, 0.92, 0.90);
  let cold = vec3<f32>(0.13, 0.30, 0.78);
  let hot = vec3<f32>(0.80, 0.10, 0.12);
  return mix(mid, select(hot, cold, x < 0.0), pow(abs(x), 0.8));
}

fn pressureCoefficient(rho1: f32) -> f32 {
  return (rho1 / 3.0) / (0.5 * S.dims.w * S.dims.w);
}
`;

export const MESH_SHADER = COMMON + /* wgsl */ `
struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) color: vec3<f32>) -> VOut {
  var out: VOut;
  let world = S.model * vec4<f32>(pos, 1.0);
  out.clip = S.viewProj * world;
  out.world = world.xyz;
  out.normal = (S.model * vec4<f32>(normal, 0.0)).xyz;
  out.color = color;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let view = normalize(S.camPos.xyz - in.world);
  // Face normal from screen-space derivatives: reliable even when the file's normals are not.
  var face = normalize(cross(dpdx(in.world), dpdy(in.world)));
  if (dot(face, view) < 0.0) { face = -face; }
  var n = normalize(in.normal);
  if (dot(n, face) < 0.0) { n = -n; }

  var base = in.color;
  var shine = 0.25;
  if (S.mesh.x > 0.5) {
    // Nine taps just above the surface smooth out the cell-to-cell ripple of a voxelised wall.
    let t1 = normalize(cross(face, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(face.y) > 0.9)));
    let t2 = cross(face, t1);
    let c = in.world + face * 1.5;
    var rho1 = textureSampleLevel(macroTex, samp, c / S.dims.xyz, 0.0).w * 2.0;
    for (var i = 0; i < 8; i++) {
      let angle = f32(i) * 0.78539816;
      let reach = select(1.4, 2.4, i % 2 == 1);
      let tap = c + (t1 * cos(angle) + t2 * sin(angle)) * reach;
      rho1 += textureSampleLevel(macroTex, samp, tap / S.dims.xyz, 0.0).w;
    }
    rho1 /= 10.0;
    let cp = pressureCoefficient(rho1);
    base = diverging(cp / S.mesh.y);
    shine = 0.08;
  }
  let key = normalize(vec3<f32>(-0.45, 0.8, 0.5));
  let fill = normalize(vec3<f32>(0.6, 0.25, -0.7));
  let diffuse = 0.34 + 0.62 * max(dot(n, key), 0.0) + 0.22 * max(dot(n, fill), 0.0);
  let spec = pow(max(dot(n, normalize(key + view)), 0.0), 48.0) * shine;
  let rim = pow(1.0 - max(dot(n, view), 0.0), 3.0) * 0.12;
  return vec4<f32>(base * diffuse + vec3<f32>(spec + rim), 1.0);
}
`;

export const LINE_HELPER_SHADER = COMMON + /* wgsl */ `
@vertex
fn vs(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
  return S.viewProj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return S.tint;
}
`;

export const STREAK_SHADER = COMMON + /* wgsl */ `
struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) side: f32,
};

const CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
);

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  out.color = vec4<f32>(0.0);
  out.side = 0.0;

  let per = u32(S.streak.x);
  let k = ii % per;
  let a = parts[ii];
  let b = parts[ii - k + (k + 1u) % per];
  let life = S.streak.z;
  let gap = b.w - a.w;
  if (a.x < -100.0 || b.x < -100.0 || gap <= 0.0 || gap > 3.0 * life / f32(per)) { return out; }
  if (distance(a.xyz, b.xyz) > 8.0) { return out; }

  let ca = S.viewProj * vec4<f32>(a.xyz, 1.0);
  let cb = S.viewProj * vec4<f32>(b.xyz, 1.0);
  if (ca.w <= 0.01 || cb.w <= 0.01) { return out; }

  let corner = CORNERS[vi];
  let sa = ca.xy / ca.w * S.viewport.xy;
  let sb = cb.xy / cb.w * S.viewport.xy;
  var dir = sb - sa;
  let len = length(dir);
  dir = select(vec2<f32>(1.0, 0.0), dir / max(len, 1e-6), len > 1e-6);
  let normal = vec2<f32>(-dir.y, dir.x);
  let c = mix(ca, cb, corner.x);
  let offset = normal * corner.y * S.viewport.z * 2.0 / S.viewport.xy;
  out.clip = vec4<f32>(c.xy + offset * c.w, c.zw);

  let p = mix(a.xyz, b.xyz, corner.x);
  let age = mix(a.w, b.w, corner.x) / life;
  let fade = smoothstep(0.0, 0.01, age) * (1.0 - smoothstep(0.85, 1.0, age));
  var rgb = vec3<f32>(0.92, 0.95, 1.0);
  if (S.mesh.w > 0.5) {
    let speed = length(textureSampleLevel(macroTex, samp, p / S.dims.xyz, 0.0).xyz) / S.dims.w;
    rgb = turbo(speed / S.mesh.z);
  }
  out.color = vec4<f32>(rgb, fade);
  out.side = corner.y;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let soft = 1.0 - in.side * in.side * 0.75;
  return vec4<f32>(in.color.rgb, in.color.a * soft * 0.9);
}
`;

export const SLICE_SHADER = COMMON + /* wgsl */ `
struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) world: vec3<f32>,
};

const QUAD = array<vec2<f32>, 6>(
  vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
);

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let q = QUAD[vi];
  let d = S.dims.xyz;
  let at = S.slice.y;
  var p: vec3<f32>;
  if (S.slice.x < 0.5) { p = vec3<f32>(at, q.x * d.y, q.y * d.z); }
  else if (S.slice.x < 1.5) { p = vec3<f32>(q.x * d.x, at, q.y * d.z); }
  else { p = vec3<f32>(q.x * d.x, q.y * d.y, at); }
  var out: VOut;
  out.clip = S.viewProj * vec4<f32>(p, 1.0);
  out.world = p;
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let uvw = in.world / S.dims.xyz;
  if (textureSampleLevel(solidTex, samp, uvw, 0.0).r > 0.5) { discard; }
  let m = textureSampleLevel(macroTex, samp, uvw, 0.0);
  var rgb: vec3<f32>;
  if (S.slice.z < 0.5) {
    rgb = turbo(length(m.xyz) / S.dims.w / S.mesh.z);
  } else if (S.slice.z < 1.5) {
    rgb = diverging(pressureCoefficient(m.w) / S.mesh.y);
  } else {
    let w = textureSampleLevel(vortTex, samp, uvw, 0.0).g;
    rgb = turbo(w / 12.0);
  }
  return vec4<f32>(rgb, S.slice.w);
}
`;

export const VOLUME_SHADER = COMMON + /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

fn hash(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn qAt(p: vec3<f32>) -> f32 {
  return textureSampleLevel(vortTex, samp, p / S.dims.xyz, 0.0).r;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let ndc = vec2<f32>(frag.x / S.viewport.x * 2.0 - 1.0, 1.0 - frag.y / S.viewport.y * 2.0);
  let near = S.invViewProj * vec4<f32>(ndc, 0.0, 1.0);
  let far = S.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  let origin = near.xyz / near.w;
  let dir = normalize(far.xyz / far.w - origin);

  let inv = 1.0 / dir;
  let t0 = (vec3<f32>(0.5) - origin) * inv;
  let t1 = (S.dims.xyz - vec3<f32>(0.5) - origin) * inv;
  let lo = min(t0, t1);
  let hi = max(t0, t1);
  let tEnter = max(max(max(lo.x, lo.y), lo.z), 0.0);
  let tExit = min(min(hi.x, hi.y), hi.z);
  if (tEnter >= tExit) { discard; }

  let smokeOn = S.volume.x > 0.5;
  let vortexOn = S.volume.y > 0.5;
  let threshold = S.volume.z;
  let stepLen = max((tExit - tEnter) / 200.0, 0.7);
  let jitter = hash(frag.xy);
  let light = normalize(vec3<f32>(-0.45, 0.8, 0.5));

  var color = vec3<f32>(0.0);
  var alpha = 0.0;
  var prevQ = 1e9;
  for (var i = 0; i < 200; i++) {
    let t = tEnter + (f32(i) + jitter) * stepLen;
    if (t > tExit || alpha > 0.985) { break; }
    let p = origin + dir * t;
    let uvw = p / S.dims.xyz;
    if (textureSampleLevel(solidTex, samp, uvw, 0.0).r > 0.5) { break; }

    if (vortexOn) {
      let q = qAt(p);
      if (q > threshold && prevQ <= threshold) {
        let hit = p - dir * stepLen * (q - threshold) / max(q - prevQ, 1e-6);
        let e = vec2<f32>(0.75, 0.0);
        let grad = vec3<f32>(
          qAt(hit + e.xyy) - qAt(hit - e.xyy),
          qAt(hit + e.yxy) - qAt(hit - e.yxy),
          qAt(hit + e.yyx) - qAt(hit - e.yyx),
        );
        let n = -normalize(grad + vec3<f32>(1e-9));
        let speed = textureSampleLevel(vortTex, samp, hit / S.dims.xyz, 0.0).b;
        let shade = 0.38 + 0.62 * max(dot(n, light), 0.0) + 0.15 * max(dot(n, -dir), 0.0);
        let a = 0.93;
        color += (1.0 - alpha) * a * turbo(speed / S.mesh.z) * shade;
        alpha += (1.0 - alpha) * a;
        break;
      }
      prevQ = q;
    }

    if (smokeOn) {
      // Drop the faint haze that numerical diffusion spreads around each plume.
      let d = max(textureSampleLevel(smokeTex, samp, uvw, 0.0).r - 0.04, 0.0);
      if (d > 0.0) {
        // Cheap self-shadowing: denser smoke towards the light darkens this sample.
        let towards = textureSampleLevel(smokeTex, samp, (p + light * 2.0) / S.dims.xyz, 0.0).r;
        let lit = mix(1.0, 0.55, clamp(towards * 1.5, 0.0, 1.0));
        let a = 1.0 - exp(-d * S.volume.w * stepLen);
        color += (1.0 - alpha) * a * vec3<f32>(0.90, 0.93, 0.97) * lit;
        alpha += (1.0 - alpha) * a;
      }
    }
  }
  if (alpha < 0.002) { discard; }
  return vec4<f32>(color, alpha);
}
`;
