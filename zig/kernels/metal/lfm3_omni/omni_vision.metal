// LFM3-Omni's pictures: the LFM3-VL tower's own operators (zig/kernels/metal/lfm3/vision.metal on the LFM3-VL Metal
// port, the same tower and weights shapes), bf16 operands and fp32 sums, rounded where the reference rounds. Rows are
// patches in merge order (each 2 x 2 block's four patches together), 1024 wide; the products go through the family's
// GEMM (omni_nax.metal or the simdgroup-matrix product).
#include <metal_stdlib>
using namespace metal;

constant constexpr uint L3VW = 1024;

inline float l3v_sum(float x, threadgroup float* partial, uint sg, uint lane) {
  x = simd_sum(x);
  if (lane == 0) partial[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
    if (lane == 0) partial[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = partial[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// RMSNorm over a 1024-wide row (eps 1e-5): the normalized value rounds before the weight multiply.
[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_rms(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]], device bfloat* y [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * L3VW;
  float v[4];
  for (uint j = 0; j < 4; ++j) v[j] = float(x[base + tid + 256 * j]);
  float square = 0.0f;
  for (uint j = 0; j < 4; ++j) square += v[j] * v[j];
  threadgroup float part[8];
  const float inv = metal::precise::rsqrt(l3v_sum(square, part, sg, lane) / float(L3VW) + 1e-5f);
  for (uint j = 0; j < 4; ++j) {
    const uint col = tid + 256 * j;
    y[base + col] = bfloat(float(bfloat(v[j] * inv)) * float(weight[col]));
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_add(
    const device bfloat* branch [[buffer(0)]], device bfloat* residual [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count) residual[i] = bfloat(float(residual[i]) + float(branch[i]));
}

// A merge-order row's place on the patch grid, and back.
inline uint2 l3v_place(uint row, uint gw) {
  return uint2(((row / 4) % (gw / 2)) * 2 + row % 2, (row / (2 * gw)) * 2 + (row / 2) % 2);
}

inline uint l3v_index(uint y, uint x, uint gw) {
  return ((y / 2) * (gw / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
}

struct L3vGrid { uint gh, gw; };

// The gated 3 x 3 depthwise convolution over one picture's grid: in_proj's (B, C, X) thirds, B * X convolved (zero
// padded), times C.
[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_conv(
    const device bfloat* bcx [[buffer(0)]], const device bfloat* weights [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant L3vGrid& grid [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const uint channel = at.x, row = at.y;
  if (channel >= L3VW || row >= grid.gh * grid.gw) return;
  const uint2 pos = l3v_place(row, grid.gw);
  float acc = 0.0f;
  for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
    const int sy = int(pos.y) + dy, sx = int(pos.x) + dx;
    bfloat bx = bfloat(0.0f);
    if (sy >= 0 && sx >= 0 && sy < int(grid.gh) && sx < int(grid.gw)) {
      const size_t src = size_t(l3v_index(uint(sy), uint(sx), grid.gw)) * 3 * L3VW + channel;
      bx = bfloat(float(bcx[src]) * float(bcx[src + 2 * L3VW]));
    }
    acc += float(bx) * float(weights[channel * 9 + (dy + 1) * 3 + dx + 1]);
  }
  const bfloat conv = bfloat(acc);
  y[size_t(row) * L3VW + channel] = bfloat(float(conv) * float(bcx[size_t(row) * 3 * L3VW + L3VW + channel]));
}

// Q/K head norms (round before the weight multiply), then the axial RoPE: the first half of each head's pairs by the
// patch's row, the second by its column (theta 1e6, 16 frequencies each), rotated in fp32. 16 query and 8 key heads.
[[kernel, max_total_threads_per_threadgroup(32)]] void l3v_qk(
    device bfloat* q [[buffer(0)]], device bfloat* k [[buffer(1)]], const device bfloat* qn [[buffer(2)]],
    const device bfloat* kn [[buffer(3)]], constant uint& gw [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  const bool query = head < 16;
  device bfloat* src = query ? q + (size_t(row) * 16 + head) * 64 : k + (size_t(row) * 8 + head - 16) * 64;
  const device bfloat* gain = query ? qn : kn;
  const float a = float(src[lane]), b = float(src[lane + 32]);
  const float inv = metal::precise::rsqrt(simd_sum(a * a + b * b) / 64.0f + 1e-5f);
  const float na = float(bfloat(float(bfloat(a * inv)) * float(gain[lane])));
  const float nb = float(bfloat(float(bfloat(b * inv)) * float(gain[lane + 32])));
  const uint2 pos = l3v_place(row, gw);
  const float coord = float(lane < 16 ? pos.y : pos.x);
  const float angle = coord * metal::exp2(-float(lane % 16) / 16.0f * 19.931568569324174f);
  const float cs = metal::precise::cos(angle), sn = metal::precise::sin(angle);
  src[lane] = bfloat(na * cs - nb * sn);
  src[lane + 32] = bfloat(nb * cs + na * sn);
}

// Bidirectional grouped attention over one picture's patches (scale 1/8): 8 rows x a KV head's 2 query heads share
// each 32 keys staged once; an online softmax in fp32, one bf16 rounding of the output.
[[kernel, max_total_threads_per_threadgroup(512)]] void l3v_attention_tiled(
    const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant uint& patches [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat ks[32 * 64];
  threadgroup bfloat vs[32 * 64];
  const uint kh = tg.y, row = tg.x * 8 + sg / 2, head = kh * 2 + sg % 2;
  const bool live = row < patches;
  const size_t qi = (size_t(row) * 16 + head) * 64 + lane;
  const float qa = live ? float(q[qi]) : 0.0f, qb = live ? float(q[qi + 32]) : 0.0f;
  float maximum = -INFINITY;
  float total = 0.0f, ya = 0.0f, yb = 0.0f;
  for (uint t0 = 0; t0 < patches; t0 += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const uint r = tid / 16, c = (tid % 16) * 4, key = t0 + r;
      uint2 kk = uint2(0), vv = uint2(0);
      if (key < patches) {
        const size_t at = (size_t(key) * 8 + kh) * 64 + c;
        kk = *(const device uint2*)(k + at);
        vv = *(const device uint2*)(v + at);
      }
      *(threadgroup uint2*)(ks + r * 64 + c) = kk;
      *(threadgroup uint2*)(vs + r * 64 + c) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) continue;
    for (uint r = 0; r < min(32u, patches - t0); ++r) {
      float dot = qa * float(ks[r * 64 + lane]);
      dot += qb * float(ks[r * 64 + lane + 32]);
      const float score = simd_sum(dot) * 0.125f;
      const float next = max(maximum, score);
      const float alpha = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
      const float probability = score == -INFINITY ? 0.0f : metal::precise::exp(score - next);
      total = total * alpha + probability;
      ya = ya * alpha + probability * float(vs[r * 64 + lane]);
      yb = yb * alpha + probability * float(vs[r * 64 + lane + 32]);
      maximum = next;
    }
  }
  if (!live) return;
  y[qi] = bfloat(ya / total);
  y[qi + 32] = bfloat(yb / total);
}

// SiLU(gate) rounded, times up, rounded.
[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_swiglu(
    const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]], device bfloat* out [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const float g = float(gate[i]);
  const float sigmoid = 1.0f / (1.0f + metal::precise::exp(-g));
  const bfloat silu = bfloat(g * sigmoid);
  out[i] = bfloat(float(silu) * float(up[i]));
}

// The projector's per-patch LayerNorm (eps 1e-6, with bias) before four adjacent merge-order patches make one row.
[[kernel, max_total_threads_per_threadgroup(256)]] void l3v_layer_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]], const device bfloat* bias [[buffer(2)]],
    device bfloat* y [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * L3VW;
  float v[4], sum = 0.0f;
  for (uint j = 0; j < 4; ++j) { v[j] = float(x[base + tid + 256 * j]); sum += v[j]; }
  threadgroup float part[8];
  const float mean = l3v_sum(sum, part, sg, lane) / float(L3VW);
  float sq = 0.0f;
  for (uint j = 0; j < 4; ++j) { v[j] -= mean; sq += v[j] * v[j]; }
  const float inv = metal::precise::rsqrt(l3v_sum(sq, part, sg, lane) / float(L3VW) + 1e-6f);
  for (uint j = 0; j < 4; ++j) {
    const uint col = tid + 256 * j;
    y[base + col] = bfloat(v[j] * inv * float(weight[col]) + float(bias[col]));
  }
}
