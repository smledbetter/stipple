// WGSL inlined as TS string literals. Scatter pipeline:
//   - one instanced unit-quad per point
//   - antialiased disk fragment
//   - view transform (world → clip) via translate + scale uniforms
//   - per-instance color index → palette lookup

export const SCATTER_WGSL = /* wgsl */ `
struct Uniforms {
  viewport: vec2f,
  point_size_px: f32,
  palette_n: f32,
  view_translate: vec2f,
  view_scale: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> color_idx: array<u32>;
@group(0) @binding(3) var<storage, read> palette: array<vec4f>;

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) tint: vec3f,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
  var quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
  );

  let world = positions[iid];
  let clip_center = (world - u.view_translate) * u.view_scale;

  let offset_px = quad[vid] * u.point_size_px;
  let offset_ndc = offset_px * 2.0 / u.viewport;

  // Wrap-around safe palette lookup
  let pn = max(1u, u32(u.palette_n));
  let ci = color_idx[iid] % pn;

  var out: VSOut;
  out.clip = vec4f(clip_center + offset_ndc, 0.0, 1.0);
  out.uv = quad[vid];
  out.tint = palette[ci].rgb;
  return out;
}

@fragment
fn fs(@location(0) uv: vec2f, @location(1) tint: vec3f) -> @location(0) vec4f {
  let d = length(uv);
  if (d > 1.0) { discard; }
  let edge = smoothstep(1.0, 0.78, d);
  return vec4f(tint, edge);
}
`;

// tab10 (matplotlib default categorical), normalized to [0, 1] RGB
export const TAB10: ReadonlyArray<[number, number, number]> = [
  [0.122, 0.467, 0.706],
  [1.0, 0.498, 0.055],
  [0.173, 0.627, 0.173],
  [0.839, 0.153, 0.157],
  [0.580, 0.404, 0.741],
  [0.549, 0.337, 0.294],
  [0.890, 0.467, 0.761],
  [0.498, 0.498, 0.498],
  [0.737, 0.741, 0.133],
  [0.090, 0.745, 0.812],
];

export const BRAND_BLUE: [number, number, number] = [0.20, 0.55, 0.95];

// Compute shader: point-in-polygon test via even-odd ray casting in world
// space. Each invocation handles one point. Hits append their original
// index to the output buffer via an atomic counter.
export const LASSO_COMPUTE_WGSL = /* wgsl */ `
struct LassoUniforms {
  n_points: u32,
  poly_n: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> polygon: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(4) var<uniform> lu: LassoUniforms;

fn point_in_poly(p: vec2f) -> bool {
  var inside: bool = false;
  let n = lu.poly_n;
  if (n < 3u) { return false; }
  var j: u32 = n - 1u;
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let pi = polygon[i];
    let pj = polygon[j];
    let cond1 = (pi.y > p.y) != (pj.y > p.y);
    if (cond1) {
      let xCross = (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x;
      if (p.x < xCross) {
        inside = !inside;
      }
    }
    j = i;
  }
  return inside;
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= lu.n_points) { return; }
  let p = positions[idx];
  if (point_in_poly(p)) {
    let out_i = atomicAdd(&counter, 1u);
    out_indices[out_i] = idx;
  }
}
`;
