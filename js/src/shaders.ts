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

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
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

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
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

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
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

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
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
