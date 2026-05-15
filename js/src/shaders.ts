// WGSL inlined as TS string literals. Scatter pipeline:
//   - one instanced unit-quad per point
//   - antialiased disk fragment
//   - view transform (world → clip) via translate + scale uniforms
//   - per-instance color index → palette lookup
//   - per-instance selection mask → alpha modulation

export const SCATTER_WGSL = /* wgsl */ `
struct Uniforms {
  viewport: vec2f,
  point_size_px: f32,
  palette_n: f32,
  view_translate: vec2f,
  view_scale: vec2f,
  selection_dim: f32,
  has_selection: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read> color_idx: array<u32>;
@group(0) @binding(3) var<storage, read> palette: array<vec4f>;
@group(0) @binding(4) var<storage, read> selection_mask: array<u32>;

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) tint: vec3f,
  @location(2) alpha_mul: f32,
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

  let pn = max(1u, u32(u.palette_n));
  let ci = color_idx[iid] % pn;

  // alpha modulation: full when no selection is active OR when this point
  // is in the selection; otherwise dim by selection_dim.
  let sel = selection_mask[iid];
  let dim_others = u.has_selection > 0.5;
  var alpha_mul: f32 = 1.0;
  if (dim_others && sel == 0u) {
    alpha_mul = u.selection_dim;
  }

  var out: VSOut;
  out.clip = vec4f(clip_center + offset_ndc, 0.0, 1.0);
  out.uv = quad[vid];
  out.tint = palette[ci].rgb;
  out.alpha_mul = alpha_mul;
  return out;
}

@fragment
fn fs(@location(0) uv: vec2f, @location(1) tint: vec3f, @location(2) alpha_mul: f32) -> @location(0) vec4f {
  let d = length(uv);
  if (d > 1.0) { discard; }
  let edge = smoothstep(1.0, 0.78, d);
  // Modulate the COLOR (not just alpha) by alpha_mul so dense unselected
  // regions don't saturate to full brightness via overlapping points.
  return vec4f(tint * alpha_mul, edge);
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
// index to the output buffer via an atomic counter AND write a 1 to the
// selection mask buffer (used by the scatter shader to brighten selected
// points + dim unselected ones).
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
@group(0) @binding(5) var<storage, read_write> selection_mask: array<u32>;

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

@compute @workgroup_size(256)
fn cs(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(num_workgroups) ng: vec3u,
) {
  let idx = gid.x + gid.y * (ng.x * 256u);
  if (idx >= lu.n_points) { return; }
  let p = positions[idx];
  if (point_in_poly(p)) {
    let out_i = atomicAdd(&counter, 1u);
    out_indices[out_i] = idx;
    selection_mask[idx] = 1u;
  }
}
`;

// Compute shader: zero a `mask` array of u32. Used to reset the selection
// mask before each lasso run.
export const MASK_CLEAR_WGSL = /* wgsl */ `
struct ClearUniforms { n: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> mask: array<u32>;
@group(0) @binding(1) var<uniform> cu: ClearUniforms;

@compute @workgroup_size(256)
fn cs(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(num_workgroups) ng: vec3u,
) {
  let i = gid.x + gid.y * (ng.x * 256u);
  if (i >= cu.n) { return; }
  mask[i] = 0u;
}
`;

// Hover index: world-space uniform grid (CELL_N × CELL_N cells). Built once
// per upload. Two compute passes (count + scatter) bracket a CPU prefix-sum
// readback. The hover query later walks a 3×3 cell neighborhood around the
// cursor and reduces to the nearest point with a workgroup-local tournament.

export const GRID_COUNT_WGSL = /* wgsl */ `
struct GridUniforms {
  n_points: u32,
  cell_n: u32,
  _p0: u32,
  _p1: u32,
  origin: vec2f,
  inv_cell: vec2f,
};

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> gu: GridUniforms;

fn cell_of(p: vec2f) -> u32 {
  let f = (p - gu.origin) * gu.inv_cell;
  let cx = clamp(u32(max(0.0, f.x)), 0u, gu.cell_n - 1u);
  let cy = clamp(u32(max(0.0, f.y)), 0u, gu.cell_n - 1u);
  return cy * gu.cell_n + cx;
}

@compute @workgroup_size(256)
fn cs(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(num_workgroups) ng: vec3u,
) {
  let i = gid.x + gid.y * (ng.x * 256u);
  if (i >= gu.n_points) { return; }
  atomicAdd(&cell_count[cell_of(positions[i])], 1u);
}
`;

export const GRID_SCATTER_WGSL = /* wgsl */ `
struct GridUniforms {
  n_points: u32,
  cell_n: u32,
  _p0: u32,
  _p1: u32,
  origin: vec2f,
  inv_cell: vec2f,
};

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> cell_start: array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_cursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> point_ids: array<u32>;
@group(0) @binding(4) var<uniform> gu: GridUniforms;

fn cell_of(p: vec2f) -> u32 {
  let f = (p - gu.origin) * gu.inv_cell;
  let cx = clamp(u32(max(0.0, f.x)), 0u, gu.cell_n - 1u);
  let cy = clamp(u32(max(0.0, f.y)), 0u, gu.cell_n - 1u);
  return cy * gu.cell_n + cx;
}

@compute @workgroup_size(256)
fn cs(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(num_workgroups) ng: vec3u,
) {
  let i = gid.x + gid.y * (ng.x * 256u);
  if (i >= gu.n_points) { return; }
  let c = cell_of(positions[i]);
  let off = atomicAdd(&cell_cursor[c], 1u);
  point_ids[cell_start[c] + off] = i;
}
`;

// Hover query: one workgroup of 64 threads scans a (2r+1)² cell window
// around the cursor cell, each thread tracks its own best (dist², idx),
// then a shared-memory tournament reduces to a single winner.
export const HOVER_QUERY_WGSL = /* wgsl */ `
struct QueryUniforms {
  cursor: vec2f,
  cell_n: u32,
  radius_cells: u32,
  origin: vec2f,
  inv_cell: vec2f,
  _p0: u32,
  _p1: u32,
  _p2: u32,
  _p3: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> cell_start: array<u32>;
@group(0) @binding(2) var<storage, read> cell_count: array<u32>;
@group(0) @binding(3) var<storage, read> point_ids: array<u32>;
@group(0) @binding(4) var<uniform> qu: QueryUniforms;
@group(0) @binding(5) var<storage, read_write> out_result: array<u32>;

var<workgroup> wg_dist: array<f32, 64>;
var<workgroup> wg_idx: array<u32, 64>;

@compute @workgroup_size(64)
fn cs(@builtin(local_invocation_id) lid: vec3u) {
  let cf = (qu.cursor - qu.origin) * qu.inv_cell;
  let cell_n_i = i32(qu.cell_n);
  let cx0 = i32(floor(cf.x));
  let cy0 = i32(floor(cf.y));
  let r = i32(qu.radius_cells);
  let cl = max(cx0 - r, 0);
  let ch = min(cx0 + r, cell_n_i - 1);
  let rl = max(cy0 - r, 0);
  let rh = min(cy0 + r, cell_n_i - 1);

  var best_dist: f32 = 3.4e38;
  var best_idx: u32 = 0xFFFFFFFFu;

  for (var cy = rl; cy <= rh; cy = cy + 1) {
    for (var cx = cl; cx <= ch; cx = cx + 1) {
      let c = u32(cy) * qu.cell_n + u32(cx);
      let start = cell_start[c];
      let count = cell_count[c];
      var k = lid.x;
      loop {
        if (k >= count) { break; }
        let idx = point_ids[start + k];
        let p = positions[idx];
        let dx = p.x - qu.cursor.x;
        let dy = p.y - qu.cursor.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < best_dist) {
          best_dist = d2;
          best_idx = idx;
        }
        k = k + 64u;
      }
    }
  }

  wg_dist[lid.x] = best_dist;
  wg_idx[lid.x] = best_idx;
  workgroupBarrier();

  var stride: u32 = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      let da = wg_dist[lid.x];
      let db = wg_dist[lid.x + stride];
      if (db < da) {
        wg_dist[lid.x] = db;
        wg_idx[lid.x] = wg_idx[lid.x + stride];
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    out_result[0] = bitcast<u32>(wg_dist[0]);
    out_result[1] = wg_idx[0];
  }
}
`;

export const HOVER_GRID_CELL_N = 256;

// RasterScan density primitive. Build pass: one thread per point, atomicAdd
// into the bin's slot in a CELL_N × CELL_N grid (default 1024² = 1M bins).
// Render pass: full-screen triangle, fragment back-projects pixel → world →
// bin, samples the bin count, runs a log color map through the palette LUT.
// Frame cost scales with rendered pixels (≤ canvas size), not with N.

export const DENSITY_BUILD_WGSL = /* wgsl */ `
struct DensityUniforms {
  n_points: u32,
  bin_n: u32,
  _p0: u32,
  _p1: u32,
  origin: vec2f,
  inv_cell: vec2f,
};

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> bin_count: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> du: DensityUniforms;

@compute @workgroup_size(256)
fn cs(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(num_workgroups) ng: vec3u,
) {
  let i = gid.x + gid.y * (ng.x * 256u);
  if (i >= du.n_points) { return; }
  let p = positions[i];
  let f = (p - du.origin) * du.inv_cell;
  let bn = f32(du.bin_n);
  let cx = u32(clamp(f.x, 0.0, bn - 1.0));
  let cy = u32(clamp(f.y, 0.0, bn - 1.0));
  atomicAdd(&bin_count[cy * du.bin_n + cx], 1u);
}
`;

export const DENSITY_RENDER_WGSL = /* wgsl */ `
struct RenderUniforms {
  viewport: vec2f,
  bin_n: u32,
  log_max: f32,
  view_translate: vec2f,
  view_scale: vec2f,
  origin: vec2f,
  inv_cell: vec2f,
  palette_n: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<uniform> u: RenderUniforms;
@group(0) @binding(1) var<storage, read> bin_count: array<u32>;
@group(0) @binding(2) var<storage, read> palette: array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
  // Oversized triangle covering the viewport — the clip stage trims it.
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  return vec4f(pos[vid], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
  // Pixel → clip (WebGPU fragment Y is top-down; clip Y is bottom-up).
  let clip_x = (frag_coord.x / u.viewport.x) * 2.0 - 1.0;
  let clip_y = 1.0 - (frag_coord.y / u.viewport.y) * 2.0;
  // Clip → world.
  let wx = clip_x / u.view_scale.x + u.view_translate.x;
  let wy = clip_y / u.view_scale.y + u.view_translate.y;
  // World → bin.
  let fx = (wx - u.origin.x) * u.inv_cell.x;
  let fy = (wy - u.origin.y) * u.inv_cell.y;
  let bn_f = f32(u.bin_n);
  if (fx < 0.0 || fy < 0.0 || fx >= bn_f || fy >= bn_f) {
    discard;
  }
  let bx = u32(fx);
  let by = u32(fy);
  let count = bin_count[by * u.bin_n + bx];
  if (count == 0u) {
    discard;
  }
  let t = log(f32(count) + 1.0) / max(u.log_max, 1e-6);
  let pn = max(1u, u.palette_n);
  let idx = u32(clamp(t * f32(pn - 1u), 0.0, f32(pn - 1u)));
  let rgb = palette[idx].rgb;
  return vec4f(rgb, 1.0);
}
`;

export const DENSITY_BIN_N = 1024;

// Single-workgroup max-reduction over the bin_count buffer. Used to get a
// fast (~0.5 ms) log_max for progressive renders during chunked uploads,
// without paying the ~10 ms CPU readback + linear scan over 4 MB.
export const MAX_REDUCE_WGSL = /* wgsl */ `
struct MaxUniforms { n: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read> bin_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_max: array<u32>;
@group(0) @binding(2) var<uniform> mu: MaxUniforms;

var<workgroup> wg_max: array<u32, 256>;

@compute @workgroup_size(256)
fn cs(@builtin(local_invocation_id) lid: vec3u) {
  var m: u32 = 0u;
  var i: u32 = lid.x;
  loop {
    if (i >= mu.n) { break; }
    let v = bin_count[i];
    if (v > m) { m = v; }
    i = i + 256u;
  }
  wg_max[lid.x] = m;
  workgroupBarrier();

  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      let a = wg_max[lid.x];
      let b = wg_max[lid.x + stride];
      if (b > a) { wg_max[lid.x] = b; }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    out_max[0] = wg_max[0];
  }
}
`;
