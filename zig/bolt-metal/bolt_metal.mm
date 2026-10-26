// libbolt on Metal: Nemotron 3.5 Bolt 74B-A9B (NVIDIA's packed 24GB checkpoint) and its LiLiCorr drafter on Apple
// GPUs, behind the C ABI of MiaAI-Lab's CUDA libbolt (Nemotron-3.5-Bolt-TensorFold-DGX-Spark, engine/bolt_host.cu and
// engine/bolt_draft.cu), so TensorFold's Bolt lane backend drives it unchanged: verify windows, keep, drafts, prompt
// snapshots. The block forward, state layout and rollback follow that engine; the kernels are in bolt.metal.
//
// Weights stay packed in unified memory (IQ2_XXS experts, NVFP4 Mamba projections, Q8_0 attention, latent
// projections and head) and are decoded in registers. A block of up to TMAX rows is one command buffer. Several
// sequences keep their own state; a shared decode round carries a segment of rows from each. Mamba state: two SSM
// slots per sequence (a block reads one and writes the other) and a ring of conv slots.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "bolt_args.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char* kSource =
#include "bolt_src.inc"
    ;

namespace {

constexpr int TMAX = BOLT_TMAX, BATCH_MAX = BOLT_BATCH_MAX, LOGIT_ROWS = BOLT_LOGIT_ROWS, SEG_MAX = BOLT_SEG_MAX;
constexpr int VERIFY_MAX = BOLT_VERIFY_MAX, CONV_SLOTS = BOLT_CONV_SLOTS, HID = BOLT_HID, VOCAB = BOLT_VOCAB;
constexpr int TOPN = BOLT_TOPN, CAND = BOLT_CAND, PROJ = BOLT_PROJ;
constexpr size_t SSM_ELEMS = BOLT_SSM_ELEMS, CONV_ELEMS = BOLT_CONV_ELEMS;
constexpr int AUX_LAYERS[6] = {1, 18, 35, 51, 68, 85};

int aux_slot(int layer) {
    for (int i = 0; i < 6; i++) if (AUX_LAYERS[i] == layer) return i;
    return -1;
}

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------------------------------------------------
// Metal: one device, one queue, an arena of shared buffers addressed by GPU address, one encoder at a time.
// ---------------------------------------------------------------------------------------------------------

struct DPtr {
    uint64_t gpu = 0;
    uint8_t* cpu = nullptr;
    DPtr at(size_t bytes) const { return {gpu ? gpu + bytes : 0, cpu ? cpu + bytes : nullptr}; }
    float* f() const { return (float*)cpu; }
    explicit operator bool() const { return gpu != 0; }
};

struct Gpu {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLLibrary> lib = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pso;
    std::vector<id<MTLBuffer>> bufs;
    id<MTLBuffer> chunk = nil;
    size_t chunk_used = 0, chunk_size = 0;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    size_t total = 0;
    int dispatches = 0;
    int par = 0;             // inside a Par scope: no barrier between its dispatches
    bool concurrent = true;  // BOLT_CONCURRENT=0: a serial encoder (every dispatch after the last, as with Par off)
    int prof = 0;            // BOLT_PROF=1: GPU time of every block; 2: every dispatch on its own, summed by kernel
    double gpu_ms = 0;
    std::unordered_map<std::string, std::pair<double, int>> by_kernel;
};
Gpu G;

bool gpu_init() {
    if (G.dev) return true;
    @autoreleasepool {
        G.dev = MTLCreateSystemDefaultDevice();
        if (!G.dev) { fprintf(stderr, "bolt: no Metal device\n"); return false; }
        G.q = [G.dev newCommandQueueWithMaxCommandBufferCount:64];
        NSError* err = nil;
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.languageVersion = MTLLanguageVersion3_2;
        if (getenv("BOLT_SAFE_MATH")) opt.mathMode = MTLMathModeSafe;
        G.prof = getenv("BOLT_PROF") ? atoi(getenv("BOLT_PROF")) : 0;
        G.concurrent = !(getenv("BOLT_CONCURRENT") && atoi(getenv("BOLT_CONCURRENT")) == 0);
        double t0 = now_ms();
        G.lib = [G.dev newLibraryWithSource:[NSString stringWithUTF8String:kSource] options:opt error:&err];
        if (!G.lib) { fprintf(stderr, "bolt: Metal compile failed: %s\n", err.localizedDescription.UTF8String); return false; }
        fprintf(stderr, "bolt: %s, kernels compiled in %.0f ms\n", G.dev.name.UTF8String, now_ms() - t0);
    }
    return true;
}

id<MTLComputePipelineState> pso(const char* name) {
    auto it = G.pso.find(name);
    if (it != G.pso.end()) return it->second;
    NSError* err = nil;
    id<MTLFunction> fn = [G.lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "bolt: no kernel %s\n", name); abort(); }
    id<MTLComputePipelineState> p = [G.dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "bolt: pipeline %s: %s\n", name, err.localizedDescription.UTF8String); abort(); }
    G.pso[name] = p;
    return p;
}

// Shared (CPU and GPU visible) memory, carved from 1 GiB buffers (or one of its own when bigger), 256-byte aligned.
DPtr alloc(size_t bytes, bool zero = true) {
    bytes = (bytes + 255) & ~(size_t)255;
    if (bytes == 0) bytes = 256;
    const size_t CH = (size_t)1 << 30;
    if (!G.chunk || G.chunk_used + bytes > G.chunk_size) {
        size_t sz = std::max(CH, bytes);
        id<MTLBuffer> b = [G.dev newBufferWithLength:sz options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (!b) { fprintf(stderr, "bolt: cannot allocate %.2f GiB\n", sz / 1073741824.0); abort(); }
        G.bufs.push_back(b);
        G.total += sz;
        if (bytes >= CH) {   // a big tensor keeps its own buffer; the current chunk stays open
            DPtr p{b.gpuAddress, (uint8_t*)b.contents};
            if (zero) memset(p.cpu, 0, bytes);
            return p;
        }
        G.chunk = b;
        G.chunk_used = 0;
        G.chunk_size = sz;
    }
    DPtr p{G.chunk.gpuAddress + G.chunk_used, (uint8_t*)G.chunk.contents + G.chunk_used};
    G.chunk_used += bytes;
    if (zero) memset(p.cpu, 0, bytes);
    return p;
}

void begin() {
    if (G.enc) return;
    G.cb = [G.q commandBufferWithUnretainedReferences];
    G.enc = [G.cb computeCommandEncoderWithDispatchType:G.concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    [G.enc useResources:G.bufs.data() count:G.bufs.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
}

// Finish the work encoded so far and wait for it.
bool finish() {
    if (!G.enc) return true;
    [G.enc endEncoding];
    [G.cb commit];
    [G.cb waitUntilCompleted];
    bool ok = G.cb.status == MTLCommandBufferStatusCompleted;
    if (G.prof) G.gpu_ms += (G.cb.GPUEndTime - G.cb.GPUStartTime) * 1000.0;
    if (!ok) fprintf(stderr, "bolt: command buffer failed: %s\n", G.cb.error.localizedDescription.UTF8String);
    G.enc = nil;
    G.cb = nil;
    return ok;
}

template <class A>
void disp(const char* name, const A& args, unsigned gx, unsigned gy, unsigned gz, unsigned threads) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    begin();
    [G.enc setComputePipelineState:pso(name)];
    [G.enc setBytes:&args length:sizeof(A) atIndex:0];
    [G.enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    // the kernels reach memory by GPU address, which Metal does not track: order every dispatch after the last, except
    // between the independent dispatches of a Par scope
    if (G.par == 0 || !G.concurrent) [G.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    G.dispatches++;
    if (G.prof == 2) {
        const double before = G.gpu_ms;
        finish();
        auto& e = G.by_kernel[name];
        e.first += G.gpu_ms - before;
        e.second++;
    }
}

// Dispatches that do not depend on each other run concurrently; the scope's end orders what follows after all of them.
struct Par {
    Par() { G.par++; }
    ~Par() {
        if (--G.par == 0 && G.enc) [G.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
};

void copyf(DPtr dst, DPtr src, size_t n, int add = 0) {
    CopyArgs a{src.gpu, dst.gpu, (uint32_t)n, add};
    disp("copy_f32", a, (unsigned)((n + 255) / 256), 1, 1, 256);
}

// ---------------------------------------------------------------------------------------------------------
// Weights: safetensors read through mmap, repacked into the planar layouts the kernels read.
// ---------------------------------------------------------------------------------------------------------

struct Info { std::string dtype; std::vector<long> shape; size_t off = 0, len = 0; };

std::string slurp(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s(n, 0);
    if (fread(s.data(), 1, n, f) != (size_t)n) s.clear();
    fclose(f);
    return s;
}

bool parse_header(const uint8_t* file, size_t n, std::unordered_map<std::string, Info>& out) {
    if (n < 8) return false;
    uint64_t hlen;
    memcpy(&hlen, file, 8);
    if (8 + hlen > n) return false;
    std::string h((const char*)file + 8, hlen);
    size_t i = 0;
    while (true) {
        size_t k = h.find("\"dtype\"", i);
        if (k == std::string::npos) break;
        size_t name_end = h.rfind('"', k - 2);
        size_t name_start = h.rfind('"', name_end - 1);
        if (name_start == std::string::npos) break;
        std::string name = h.substr(name_start + 1, name_end - name_start - 1);
        Info info;
        size_t dq = h.find('"', h.find("dtype", k) + 6);
        size_t dq2 = h.find('"', dq + 1);
        info.dtype = h.substr(dq + 1, dq2 - dq - 1);
        size_t sb = h.find('[', dq2);
        size_t se = h.find(']', sb);
        std::string sh = h.substr(sb + 1, se - sb - 1);
        long v = 0;
        bool any = false;
        for (char c : sh) {
            if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); any = true; }
            else if (any) { info.shape.push_back(v); v = 0; any = false; }
        }
        if (any) info.shape.push_back(v);
        size_t ob = h.find("data_offsets", se);
        size_t br = h.find('[', ob);
        unsigned long long a = 0, b = 0;
        sscanf(h.c_str() + br, "[%llu, %llu]", &a, &b);
        info.off = a;
        info.len = b - a;
        out[name] = info;
        i = br;
    }
    return !out.empty();
}

struct Store {
    std::unordered_map<std::string, std::string> shard;   // tensor -> file
    std::unordered_map<std::string, std::unordered_map<std::string, Info>> headers;
    std::unordered_map<std::string, std::pair<const uint8_t*, size_t>> files;
    std::string dir;
    std::vector<std::pair<void*, size_t>> maps;
    ~Store() { for (auto& m : maps) munmap(m.first, m.second); }
};

// A directory with model.safetensors.index.json (names under `prefix`), or a single model.safetensors.
Store* open_store(const char* dir, const char* prefix) {
    auto* s = new Store();
    s->dir = dir;
    std::string idx = slurp(std::string(dir) + "/model.safetensors.index.json");
    if (idx.empty()) {
        s->shard["*"] = "model.safetensors";
        return s;
    }
    std::string pre = std::string("\"") + prefix;
    size_t p = 0;
    while (true) {
        size_t a = idx.find(pre, p);
        if (a == std::string::npos) break;
        size_t b = idx.find('"', a + 1);
        std::string name = idx.substr(a + 1, b - a - 1);
        size_t c = idx.find('"', b + 1);
        size_t d = idx.find('"', c + 1);
        s->shard[name] = idx.substr(c + 1, d - c - 1);
        p = d + 1;
    }
    fprintf(stderr, "bolt: index %zu tensors\n", s->shard.size());
    return s;
}

bool lookup(Store* s, const std::string& name, Info& info, const uint8_t*& data) {
    auto it = s->shard.find(name);
    if (it == s->shard.end()) it = s->shard.find("*");
    if (it == s->shard.end()) return false;
    const std::string& shard = it->second;
    if (!s->headers.count(shard)) {
        std::string path = s->dir + "/" + shard;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); return false; }
        void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) return false;
        s->maps.push_back({m, (size_t)st.st_size});
        s->files[shard] = {(const uint8_t*)m, (size_t)st.st_size};
        parse_header((const uint8_t*)m, st.st_size, s->headers[shard]);
    }
    auto jt = s->headers[shard].find(name);
    if (jt == s->headers[shard].end()) return false;
    info = jt->second;
    uint64_t hlen;
    memcpy(&hlen, s->files[shard].first, 8);
    data = s->files[shard].first + 8 + hlen + info.off;
    return true;
}

size_t numel(const Info& info) { size_t n = 1; for (long d : info.shape) n *= (size_t)d; return n; }

float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

// An fp32 copy of an F32 or BF16 tensor.
DPtr dev_f32(Store* s, const std::string& name, size_t expect = 0) {
    Info info; const uint8_t* data;
    if (!lookup(s, name, info, data)) { fprintf(stderr, "bolt: missing %s\n", name.c_str()); return {}; }
    size_t n = numel(info);
    if (expect && n != expect) fprintf(stderr, "bolt: %s has %zu values, wanted %zu\n", name.c_str(), n, expect);
    DPtr p = alloc(n * 4, false);
    float* h = p.f();
    if (info.dtype == "F32") memcpy(h, data, n * 4);
    else if (info.dtype == "BF16") for (size_t i = 0; i < n; i++) { uint16_t v; memcpy(&v, data + i * 2, 2); h[i] = bf2f(v); }
    else { fprintf(stderr, "bolt: %s dtype %s\n", name.c_str(), info.dtype.c_str()); return {}; }
    return p;
}

DPtr dev_raw(Store* s, const std::string& name, size_t* len = nullptr) {
    Info info; const uint8_t* data;
    if (!lookup(s, name, info, data)) { fprintf(stderr, "bolt: missing %s\n", name.c_str()); return {}; }
    DPtr p = alloc(info.len, false);
    memcpy(p.cpu, data, info.len);
    if (len) *len = info.len;
    return p;
}

enum Fmt { Q8 = 0, NV4 = 1, IQ2 = 2, BF = 3 };
struct Mat { int fmt = 0; DPtr w, s; float s2 = 1.f; int n = 0, k = 0; };
struct Slab { DPtr qs, d; size_t qs_stride = 0, d_stride = 0; };

// Q8_0 [N][K/32] blocks of (half d, 32 int8) into planar int8 [N][K] and half [N][K/32].
Mat load_q8(Store* s, const std::string& name) {
    Mat m; m.fmt = Q8;
    Info w; const uint8_t* data;
    if (!lookup(s, name + ".weight", w, data) || w.shape.size() != 3 || w.shape[2] != 34) {
        fprintf(stderr, "bolt: %s is not a Q8_0 tensor\n", name.c_str());
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = (int)w.shape[1] * 32;
    const size_t nb = (size_t)m.n * (m.k / 32);
    m.w = alloc((size_t)m.n * m.k, false);
    m.s = alloc(nb * 2, false);
    for (size_t i = 0; i < nb; i++) {
        memcpy(m.s.cpu + i * 2, data + i * 34, 2);
        memcpy(m.w.cpu + i * 32, data + i * 34 + 2, 32);
    }
    return m;
}

Mat load_nv4(Store* s, const std::string& name) {
    Mat m; m.fmt = NV4;
    Info w; const uint8_t* data;
    if (!lookup(s, name + ".weight", w, data) || w.shape.size() != 2) {
        fprintf(stderr, "bolt: %s is not an NVFP4 tensor\n", name.c_str());
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = (int)w.shape[1] * 2;
    m.w = alloc(w.len, false);
    memcpy(m.w.cpu, data, w.len);
    m.s = dev_raw(s, name + ".weight_scale");
    Info g; const uint8_t* gd;
    if (lookup(s, name + ".weight_scale_2", g, gd)) memcpy(&m.s2, gd, 4);
    return m;
}

// IQ2_XXS [N][K/256] blocks of (half d, 64 bytes) into planar bytes [N][K/256][64] and half [N][K/256].
void repack_iq2(const uint8_t* src, uint8_t* qs, uint8_t* d, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        memcpy(d + i * 2, src + i * 66, 2);
        memcpy(qs + i * 64, src + i * 66 + 2, 64);
    }
}

Mat load_iq2(Store* s, const std::string& name) {
    Mat m; m.fmt = IQ2;
    Info w; const uint8_t* data;
    if (!lookup(s, name + ".weight", w, data) || w.shape.size() != 3 || w.shape[2] != 66) {
        fprintf(stderr, "bolt: %s is not an IQ2_XXS tensor\n", name.c_str());
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = (int)w.shape[1] * 256;
    const size_t nb = (size_t)m.n * (m.k / 256);
    m.w = alloc(nb * 64, false);
    m.s = alloc(nb * 2, false);
    repack_iq2(data, m.w.cpu, m.s.cpu, nb);
    return m;
}

Mat load_bf16(Store* s, const std::string& name) {
    Mat m; m.fmt = BF;
    Info w; const uint8_t* data;
    if (!lookup(s, name, w, data) || w.shape.size() < 1) {
        fprintf(stderr, "bolt: %s is not a bf16 matrix\n", name.c_str());
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = w.shape.size() > 1 ? (int)w.shape[1] : 1;
    m.w = alloc(w.len, false);
    memcpy(m.w.cpu, data, w.len);
    return m;
}

// All experts of one projection in one slab.
bool load_slab(Store* s, const std::string& layer_prefix, const char* proj, int experts, int rows, int k, Slab& slab) {
    const size_t nb = (size_t)rows * (k / 256);
    slab.qs = alloc(nb * 64 * experts, false);
    slab.d = alloc(nb * 2 * experts, false);
    slab.qs_stride = nb * 64;
    slab.d_stride = nb;
    for (int e = 0; e < experts; e++) {
        Info w; const uint8_t* data;
        std::string name = layer_prefix + ".mixer.experts." + std::to_string(e) + "." + proj + ".weight";
        if (!lookup(s, name, w, data) || w.len != nb * 66) {
            fprintf(stderr, "bolt: expert tensor %s missing or the wrong size\n", name.c_str());
            return false;
        }
        repack_iq2(data, slab.qs.cpu + e * nb * 64, slab.d.cpu + e * nb * 2, nb);
    }
    return true;
}

std::vector<std::string> block_types(const std::string& cfg) {
    std::vector<std::string> t;
    size_t p = cfg.find("\"block_configs\"");
    if (p == std::string::npos) return t;
    size_t end = cfg.find("\"blockwise_members\"", p);
    if (end == std::string::npos) end = cfg.size();
    size_t i = p;
    while (i < end) {
        size_t a = cfg.find("\"block_type\"", i);
        if (a == std::string::npos || a > end) break;
        size_t q1 = cfg.find('"', cfg.find(':', a) + 1);
        size_t q2 = cfg.find('"', q1 + 1);
        t.push_back(cfg.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return t;
}

void moe_shape(const std::string& cfg, int index, int& experts, int& width) {
    size_t i = cfg.find("\"block_configs\"");
    int seen = 0;
    while (true) {
        size_t a = cfg.find("\"block_type\"", i);
        if (a == std::string::npos) break;
        size_t q1 = cfg.find('"', cfg.find(':', a) + 1);
        size_t q2 = cfg.find('"', q1 + 1);
        if (cfg.substr(q1 + 1, q2 - q1 - 1) == "moe") {
            if (seen == index) {
                size_t e = cfg.find("\"n_routed_experts\"", a);
                size_t w = cfg.find("\"moe_intermediate_size\"", a);
                experts = atoi(cfg.c_str() + cfg.find(':', e) + 1);
                width = atoi(cfg.c_str() + cfg.find(':', w) + 1);
                return;
            }
            seen++;
        }
        i = q2 + 1;
    }
}

// ---------------------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------------------

// y[t] = epi(W x[t] + bias), T rows.
void gemv(const Mat& m, DPtr x, int ldx, DPtr y, int ldy, int T, int epi = 0, DPtr bias = {}, int accum = 0) {
    if (T <= 0) return;
    static const char* names[4][4] = {{"gemv_q8_1", "gemv_q8_2", "gemv_q8_4", "gemv_q8_8"},
                                      {"gemv_nv4_1", "gemv_nv4_2", "gemv_nv4_4", "gemv_nv4_8"},
                                      {"gemv_iq2_1", "gemv_iq2_2", "gemv_iq2_4", "gemv_iq2_8"},
                                      {"gemv_bf_1", "gemv_bf_2", "gemv_bf_4", "gemv_bf_8"}};
    static const int gemm_min = getenv("BOLT_GEMM_MIN") ? atoi(getenv("BOLT_GEMM_MIN")) : 33;
    if (T >= gemm_min) {   // prompt chunks: tiles of 64 rows x 32 tokens on simdgroup matrices
        static const char* gnames[4] = {"gemm_q8", "gemm_nv4", "gemm_iq2", "gemm_bf"};
        GemvArgs a{m.w.gpu, m.s.gpu, x.gpu, y.gpu, bias.gpu, m.s2, m.n, m.k, ldx, ldy, T, epi, accum};
        disp(gnames[m.fmt], a, (unsigned)((m.n + 63) / 64), (unsigned)((T + 31) / 32), 1, 128);
        return;
    }
    const int v = T == 1 ? 0 : T == 2 ? 1 : T <= 4 ? 2 : 3;
    GemvArgs a{m.w.gpu, m.s.gpu, x.gpu, y.gpu, bias.gpu, m.s2, m.n, m.k, ldx, ldy, T, epi, accum};
    // BOLT_WIDE_R: weight rows a simdgroup takes for 3 to 32 activation rows (2, 4 or 8); more rows read the
    // activations fewer times. Every row's sum keeps its order whatever this is.
    static const int wide_r = getenv("BOLT_WIDE_R") ? atoi(getenv("BOLT_WIDE_R")) : 2;
    if (v >= 2 && wide_r > 2) {
        static const char* fmts[4] = {"q8", "nv4", "iq2", "bf"};
        char name[32];
        const int r = (v == 2 || wide_r < 8) ? 4 : 8;
        snprintf(name, sizeof name, "gemv_%s_%dr%d", fmts[m.fmt], v == 2 ? 4 : 8, r);
        disp(name, a, (unsigned)((m.n + 4 * r - 1) / (4 * r)), 1, 1, 128);
        return;
    }
    const int rows_per_tg = v <= 1 ? 16 : 8;
    disp(names[m.fmt][v], a, (unsigned)((m.n + rows_per_tg - 1) / rows_per_tg), 1, 1, 128);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// The LiLiCorr drafter
// ---------------------------------------------------------------------------------------------------------

namespace {

constexpr int DH = 4096, DHD = 128, DHEADS = 32, DKVH = 2, DINTER = 12288;
constexpr int DBLOCK = 8, DSLOTS = 7, DTOPK = 8, DNODES = DSLOTS * DTOPK;
constexpr int DLH = 1024, DLHEADS = 8;
constexpr int DL = 5, DRING = 2048;
constexpr float DEPS = 1e-6f, VEPS = 1e-4f;

struct DLayer {
    DPtr in_norm, post_norm, qn, kn, sink, base_a, base_m;
    Mat q, k, v, o, att_kp, mlp_kp, gate, up, down;
};
struct LLayer {
    DPtr an, mn, qkv_b, out_b, up_b, down_b;
    Mat qkv, out, up, down;
};

}  // namespace

struct Draft {
    DPtr embed;   // the target's
    Mat head;    // the target's
    Mat fc;
    DPtr hid_norm, final_norm;
    DLayer L[DL];
    LLayer ll[2];
    Mat tok_p, pass_p, ctx_p, factor, out_h, in_h, anc_h, feat_down;
    DPtr tok_b, pass_b, ctx_b, factor_b, out_b, in_b, anc_b, feat_up, feat_up_b, feat_down_b;
    DPtr feat_nw, feat_nb, out_norm, anc_norm, slot, rank, abias;
    int nseq = 1;
    DPtr ringk, ringv;           // [seq][layer][RING][2][128]
    DPtr anchor;                 // [seq][H]
    std::vector<int> clen;      // per sequence
    // scratch
    DPtr x, r, h, z, y, q, k, v, g, u, hid, dlog, ictx, ik, iv, csv, csi, lse, tokstate, passstate, feat, fh;
    DPtr nodes, ln, lqkv, lo, cat, edges_out, edges_in, aout, astate, acat, gathered, tmp, cvals, cids, outp;
    DPtr seqs_d, clen_d, pos_d, anc_d, irow_seq, irow_pos;
};

namespace {

void rms(DPtr x, DPtr w, DPtr y, int n, int rows, float eps = DEPS) {
    RmsArgs a{x.gpu, w.gpu, y.gpu, n, eps};
    disp("d_rms", a, rows, 1, 1, 256);
}

void misc(const char* k, DraftMiscArgs a, unsigned gx, unsigned gy, unsigned threads) { disp(k, a, gx, gy, 1, threads); }

void candidates(DPtr logits, int T, int topk, DPtr vals, DPtr ids, DPtr sv, DPtr si, DPtr sel = {}, DPtr masks = {}, int words = 0) {
    if (topk < 1) topk = 1;
    if (topk > CAND) topk = CAND;
    CandArgs a{logits.gpu, sel.gpu, masks.gpu, sv.gpu, si.gpu, vals.gpu, ids.gpu, topk, words};
    disp("cand1", a, BOLT_CAND_BLOCKS, T, 1, 256);
    disp("cand2", a, T, 1, 1, 256);
}

Mat dbf(Store* f, const std::string& n) { return load_bf16(f, n); }

}  // namespace

static Draft* draft_open(const char* dir, DPtr embed, const Mat& head, int nseq) {
    std::string path = std::string(dir) + "/model.safetensors";
    if (access(path.c_str(), R_OK) != 0) { fprintf(stderr, "bolt: LiLiCorr weights not found at %s\n", path.c_str()); return nullptr; }
    Store* f = open_store(dir, "");
    Draft* d = new Draft();
    d->embed = embed;
    d->head = head;
    d->fc = load_nv4(f, "fc");
    d->hid_norm = dev_f32(f, "hidden_norm.weight", DH);
    d->final_norm = dev_f32(f, "norm.weight", DH);
    for (int l = 0; l < DL; l++) {
        std::string p = "layers." + std::to_string(l);
        DLayer& a = d->L[l];
        a.in_norm = dev_f32(f, p + ".input_layernorm.weight", DH);
        a.post_norm = dev_f32(f, p + ".post_attention_layernorm.weight", DH);
        a.q = dbf(f, p + ".self_attn.q_proj.weight");
        a.k = dbf(f, p + ".self_attn.k_proj.weight");
        a.v = dbf(f, p + ".self_attn.v_proj.weight");
        a.o = dbf(f, p + ".self_attn.o_proj.weight");
        a.qn = dev_f32(f, p + ".self_attn.q_norm.weight", DHD);
        a.kn = dev_f32(f, p + ".self_attn.k_norm.weight", DHD);
        a.sink = dev_f32(f, p + ".self_attn.attention_sink_bias", DHEADS);
        a.gate = load_nv4(f, p + ".mlp.gate_proj");
        a.up = load_nv4(f, p + ".mlp.up_proj");
        a.down = load_nv4(f, p + ".mlp.down_proj");
        a.base_a = dev_f32(f, p + ".attention_conv.base_kernel", 4 * DH);
        a.base_m = dev_f32(f, p + ".mlp_conv.base_kernel", 4 * DH);
        a.att_kp = dbf(f, p + ".attention_conv.kernel_projection.weight");
        a.mlp_kp = dbf(f, p + ".mlp_conv.kernel_projection.weight");
    }
    d->nseq = nseq < 1 ? 1 : nseq;
    d->ringk = alloc((size_t)d->nseq * DL * DRING * DKVH * DHD * 4);
    d->ringv = alloc((size_t)d->nseq * DL * DRING * DKVH * DHD * 4);
    d->anchor = alloc((size_t)d->nseq * DH * 4);
    d->clen.assign(d->nseq, 0);
    d->tok_p = load_nv4(f, "lilicorr.token_proj");
    d->pass_p = load_nv4(f, "lilicorr.pass_hidden_proj");
    d->ctx_p = load_nv4(f, "lilicorr.context_proj");
    d->tok_b = dev_f32(f, "lilicorr.token_proj.bias", DLH);
    d->pass_b = dev_f32(f, "lilicorr.pass_hidden_proj.bias", DLH);
    d->ctx_b = dev_f32(f, "lilicorr.context_proj.bias", DLH);
    d->out_norm = dev_f32(f, "lilicorr.output_norm.weight", DLH);
    d->anc_norm = dev_f32(f, "lilicorr.anchor_norm.weight", DLH);
    d->factor = dbf(f, "lilicorr.factor_input_proj.weight");
    d->factor_b = dev_f32(f, "lilicorr.factor_input_proj.bias", DLH);
    d->out_h = dbf(f, "lilicorr.out_head.weight");
    d->out_b = dev_f32(f, "lilicorr.out_head.bias", DLH);
    d->in_h = dbf(f, "lilicorr.in_head.weight");
    d->in_b = dev_f32(f, "lilicorr.in_head.bias", DLH);
    d->anc_h = dbf(f, "lilicorr.anchor_out_head.weight");
    d->anc_b = dev_f32(f, "lilicorr.anchor_out_head.bias", DLH);
    d->feat_nw = dev_f32(f, "lilicorr.feature_norm.weight", 5);
    d->feat_nb = dev_f32(f, "lilicorr.feature_norm.bias", 5);
    d->feat_up = dev_raw(f, "lilicorr.feature_mlp.up_proj.weight");
    d->feat_up_b = dev_f32(f, "lilicorr.feature_mlp.up_proj.bias", DLH);
    d->feat_down = dbf(f, "lilicorr.feature_mlp.down_proj.weight");
    d->feat_down_b = dev_f32(f, "lilicorr.feature_mlp.down_proj.bias", DLH);
    for (int i = 0; i < 2; i++) {
        std::string p = "lilicorr.layers." + std::to_string(i);
        LLayer& a = d->ll[i];
        a.an = dev_f32(f, p + ".attn_norm.weight", DLH);
        a.mn = dev_f32(f, p + ".mlp_norm.weight", DLH);
        a.qkv = dbf(f, p + ".attn.in_proj_weight");
        a.qkv_b = dev_f32(f, p + ".attn.in_proj_bias", 3 * DLH);
        a.out = load_nv4(f, p + ".attn.out_proj");
        a.out_b = dev_f32(f, p + ".attn.out_proj.bias", DLH);
        a.up = load_nv4(f, p + ".mlp.up_proj");
        a.up_b = dev_f32(f, p + ".mlp.up_proj.bias", 2 * DLH);
        a.down = load_nv4(f, p + ".mlp.down_proj");
        a.down_b = dev_f32(f, p + ".mlp.down_proj.bias", DLH);
    }
    d->slot = dev_f32(f, "lilicorr.slot_embedding", DSLOTS * DLH);
    d->rank = dev_f32(f, "lilicorr.rank_embedding", DTOPK * DLH);
    {   // attention bias [heads][56][56] from the relative-slot and same-slot tables
        Info ri, si; const uint8_t *rd, *sd;
        lookup(f, "lilicorr.relative_slot_bias", ri, rd);
        lookup(f, "lilicorr.same_slot_bias", si, sd);
        const uint16_t* rb = (const uint16_t*)rd;
        const uint16_t* sb = (const uint16_t*)sd;
        d->abias = alloc((size_t)DLHEADS * DNODES * DNODES * 4);
        float* bias = d->abias.f();
        for (int i = 0; i < DNODES; i++) for (int j = 0; j < DNODES; j++) {
            int s_i = i / DTOPK, s_j = j / DTOPK;
            int rel = std::clamp(s_i - s_j, -7, 7);
            for (int h = 0; h < DLHEADS; h++) {
                float b = bf2f(rb[h * 15 + rel + 7]);
                if (s_i == s_j) b += bf2f(sb[h]);
                bias[((size_t)h * DNODES + i) * DNODES + j] = b;
            }
        }
    }
    delete f;
    const size_t NB = d->nseq, R = NB * DBLOCK, N = NB * DNODES;
    auto fa = [](size_t floats) { return alloc(floats * 4); };
    d->x = fa(R * DH); d->r = fa(R * DH); d->h = fa(R * DH); d->z = fa(R * DH);
    d->y = fa(R * 1024); d->q = fa(R * DH); d->k = fa(R * 256); d->v = fa(R * 256);
    d->g = fa(R * DINTER); d->u = fa(R * DINTER); d->hid = fa(R * DH);
    d->dlog = fa(R * VOCAB);
    d->ictx = fa((size_t)TMAX * DH); d->ik = fa((size_t)TMAX * 256); d->iv = fa((size_t)TMAX * 256);
    d->csv = fa(R * BOLT_CAND_BLOCKS * CAND); d->csi = fa(R * BOLT_CAND_BLOCKS * CAND); d->lse = fa(R);
    d->tokstate = fa(N * DLH); d->passstate = fa(R * DLH); d->feat = fa(N * 8); d->fh = fa(N * DLH);
    d->nodes = fa(N * DLH); d->ln = fa(N * DLH); d->lqkv = fa(N * 3 * DLH);
    d->lo = fa(N * DLH); d->cat = fa(N * 3 * DLH);
    d->edges_out = fa(N * DLH); d->edges_in = fa(N * DLH);
    d->aout = fa(NB * DLH); d->astate = fa(NB * DLH); d->acat = fa(NB * DH); d->gathered = fa(N * DH); d->tmp = fa(N * 2 * DLH);
    d->cvals = fa(R * CAND); d->cids = fa(R * CAND); d->outp = fa(NB * DSLOTS);
    d->seqs_d = fa(NB); d->clen_d = fa(NB); d->pos_d = fa(NB); d->anc_d = fa(NB);
    d->irow_seq = fa(TMAX); d->irow_pos = fa(TMAX);
    fprintf(stderr, "bolt: LiLiCorr resident, fc %d x %d\n", d->fc.n, d->fc.k);
    return d;
}

static size_t draft_blob_bytes() { return ((size_t)DL * 2 * DRING * DKVH * DHD + DH) * 4; }

static void draft_reset(Draft* d, int seq) { if (d) d->clen[seq] = 0; }

// Snapshots: the GPU is idle when these run, and the memory is shared, so they are copies on the CPU.
static void draft_save(Draft* d, int seq, uint8_t* dst) {
    if (!d) return;
    const size_t ring = (size_t)DRING * DKVH * DHD * 4;
    for (int l = 0; l < DL; l++) {
        memcpy(dst, d->ringk.cpu + ((size_t)seq * DL + l) * ring, ring); dst += ring;
        memcpy(dst, d->ringv.cpu + ((size_t)seq * DL + l) * ring, ring); dst += ring;
    }
    memcpy(dst, d->anchor.cpu + (size_t)seq * DH * 4, DH * 4);
}

static void draft_restore(Draft* d, int seq, const uint8_t* src, int clen) {
    if (!d) return;
    const size_t ring = (size_t)DRING * DKVH * DHD * 4;
    for (int l = 0; l < DL; l++) {
        memcpy(d->ringk.cpu + ((size_t)seq * DL + l) * ring, src, ring); src += ring;
        memcpy(d->ringv.cpu + ((size_t)seq * DL + l) * ring, src, ring); src += ring;
    }
    memcpy(d->anchor.cpu + (size_t)seq * DH * 4, src, DH * 4);
    d->clen[seq] = clen;
}

static int draft_clen(Draft* d, int seq) { return d ? d->clen[seq] : 0; }
static void draft_skip(Draft* d, int seq, int upto) { if (d && upto > d->clen[seq]) d->clen[seq] = upto; }

// Absorb the target features of n rows (cap rows [row][6][4096]) at positions [pos0, pos0 + n) of one sequence, or
// (rowseq) rows of several sequences of one block. Encodes on the open command buffer.
static void draft_ingest_rows(Draft* d, DPtr cap, int nrows, int seq, int pos0, bool multi) {
    gemv(d->fc, cap, 6 * DH, d->ictx, DH, nrows);
    rms(d->ictx, d->hid_norm, d->ictx, DH, nrows);
    for (int l = 0; l < DL; l++) {
        DLayer& a = d->L[l];
        {
            Par par;
            gemv(a.k, d->ictx, DH, d->ik, DKVH * DHD, nrows);
            gemv(a.v, d->ictx, DH, d->iv, DKVH * DHD, nrows);
        }
        RopeArgs ra{d->ik.gpu, a.kn.gpu, 0, multi ? d->irow_pos.gpu : 0, DKVH, pos0, multi ? 2 : 0};
        disp("d_qk_norm_rope", ra, nrows, DKVH, 1, 64);
        RingArgs wa{d->ik.gpu, d->iv.gpu, d->ringk.gpu, d->ringv.gpu, multi ? d->irow_seq.gpu : 0, multi ? d->irow_pos.gpu : 0, seq, l, DL, pos0};
        disp("d_write_kv", wa, nrows, 1, 1, 256);
    }
}

static void draft_ingest(Draft* d, int seq, DPtr cap, int pos0, int n) {
    if (!d || n <= 0) return;
    draft_ingest_rows(d, cap, n, seq, pos0, false);
    copyf(d->anchor.at((size_t)seq * DH * 4), d->ictx.at((size_t)(n - 1) * DH * 4), DH);
    d->clen[seq] = pos0 + n;
}

static void draft_ingest_multi(Draft* d, DPtr cap, int nrows, int nseq, const int* seqs, const int* row0, const int* n, const int* pos0) {
    if (!d || nseq < 1) return;
    int* rs = (int*)d->irow_seq.cpu;
    int* rp = (int*)d->irow_pos.cpu;
    for (int r = 0; r < nrows; r++) { rs[r] = -1; rp[r] = 0; }
    for (int i = 0; i < nseq; i++)
        for (int j = 0; j < n[i]; j++) { rs[row0[i] + j] = seqs[i]; rp[row0[i] + j] = pos0[i] + j; }
    draft_ingest_rows(d, cap, nrows, 0, 0, true);
    for (int i = 0; i < nseq; i++) {
        copyf(d->anchor.at((size_t)seqs[i] * DH * 4), d->ictx.at((size_t)(row0[i] + n[i] - 1) * DH * 4), DH);
        d->clen[seqs[i]] = pos0[i] + n[i];
    }
}

// Seven draft tokens each for the blocks starting with anchors[i] at positions pos[i] of sequences seqs[i]; every weight
// read is shared by the streams. Encodes, runs and waits. got[i] is 7, or 0 when stream i's context is not at pos[i].
static void draft_propose_multi(Draft* d, int n, const int* seqs, const int* anchors, const int* pos, int* out, int* got) {
    for (int i = 0; i < n; i++) got[i] = 0;
    if (!d || n < 1) { finish(); return; }
    int idx[SEG_MAX], B = 0;
    for (int i = 0; i < n && B < d->nseq; i++) {
        if (d->clen[seqs[i]] <= 0 || pos[i] != d->clen[seqs[i]]) continue;
        idx[B++] = i;
    }
    if (B == 0) { finish(); return; }
    int* sd = (int*)d->seqs_d.cpu; int* cl = (int*)d->clen_d.cpu; int* ps = (int*)d->pos_d.cpu; int* an = (int*)d->anc_d.cpu;
    for (int b = 0; b < B; b++) {
        sd[b] = seqs[idx[b]];
        cl[b] = d->clen[seqs[idx[b]]];
        ps[b] = pos[idx[b]];
        an[b] = anchors[idx[b]];
    }
    const int R = B * DBLOCK, NN = B * DNODES;
    DraftMiscArgs m{};
    m.table = d->embed.gpu; m.ids = d->anc_d.gpu; m.y = d->x.gpu;
    misc("d_block_embed", m, R, 1, 256);
    copyf(d->r, d->x, (size_t)R * DH);
    for (int l = 0; l < DL; l++) {
        DLayer& a = d->L[l];
        if (l > 0) {
            copyf(d->x, d->r, (size_t)R * DH, 1);
            copyf(d->r, d->x, (size_t)R * DH);
        }
        rms(d->x, a.in_norm, d->h, DH, R);
        gemv(a.att_kp, d->h, DH, d->y, 1024, R);
        GconvArgs ga{d->h.gpu, a.base_a.gpu, d->y.gpu, d->z.gpu, R, 0};
        disp("d_gconv", ga, (R * DH + 255) / 256, 1, 1, 256);
        {
            Par par;
            gemv(a.q, d->z, DH, d->q, DH, R);
            gemv(a.k, d->z, DH, d->k, 256, R);
            gemv(a.v, d->z, DH, d->v, 256, R);
        }
        RopeArgs rq{d->q.gpu, a.qn.gpu, d->pos_d.gpu, 0, DHEADS, 0, 1};
        {
            Par par;
            disp("d_qk_norm_rope", rq, R, DHEADS, 1, 64);
            RopeArgs rk{d->k.gpu, a.kn.gpu, d->pos_d.gpu, 0, DKVH, 0, 1};
            disp("d_qk_norm_rope", rk, R, DKVH, 1, 64);
        }
        DattnArgs da{d->q.gpu, d->k.gpu, d->v.gpu, d->ringk.gpu, d->ringv.gpu, a.sink.gpu, d->h.gpu, d->seqs_d.gpu, d->clen_d.gpu, d->pos_d.gpu, l, DL};
        disp("d_attn", da, DHEADS, R, 1, 128);
        gemv(a.o, d->h, DH, d->z, DH, R);
        GconvArgs gb{d->z.gpu, a.base_a.gpu, d->y.gpu, d->x.gpu, R, 1};
        disp("d_gconv", gb, (R * DH + 255) / 256, 1, 1, 256);
        copyf(d->x, d->r, (size_t)R * DH, 1);
        copyf(d->r, d->x, (size_t)R * DH);
        rms(d->x, a.post_norm, d->h, DH, R);
        gemv(a.mlp_kp, d->h, DH, d->y, 1024, R);
        GconvArgs gc{d->h.gpu, a.base_m.gpu, d->y.gpu, d->z.gpu, R, 0};
        disp("d_gconv", gc, (R * DH + 255) / 256, 1, 1, 256);
        {
            Par par;
            gemv(a.gate, d->z, DH, d->g, DINTER, R);
            gemv(a.up, d->z, DH, d->u, DINTER, R);
        }
        DraftMiscArgs sm{}; sm.a = d->g.gpu; sm.b = d->u.gpu; sm.y = d->g.gpu; sm.n = R * DINTER;
        misc("d_silu_mul", sm, (R * DINTER + 255) / 256, 1, 256);
        gemv(a.down, d->g, DINTER, d->h, DH, R);
        GconvArgs gd{d->h.gpu, a.base_m.gpu, d->y.gpu, d->x.gpu, R, 1};
        disp("d_gconv", gd, (R * DH + 255) / 256, 1, 1, 256);
    }
    copyf(d->x, d->r, (size_t)R * DH, 1);
    rms(d->x, d->final_norm, d->hid, DH, R);
    // candidates of the seven draft slots from the target's head, with their log-probabilities
    gemv(d->head, d->hid, DH, d->dlog, VOCAB, R);
    candidates(d->dlog, R, DTOPK, d->cvals, d->cids, d->csv, d->csi);
    m = {}; m.a = d->dlog.gpu; m.y = d->lse.gpu;
    misc("d_lse", m, R, 1, 256);
    // the lattice head scores each stream's 56 (slot, rank) candidates
    m = {}; m.table = d->embed.gpu; m.ids = d->cids.gpu; m.y = d->gathered.gpu;
    misc("d_gather_cand", m, NN, 1, 256);
    gemv(d->tok_p, d->gathered, DH, d->tokstate, DLH, NN, 0, d->tok_b);
    gemv(d->pass_p, d->hid, DH, d->passstate, DLH, R, 0, d->pass_b);
    m = {}; m.a = d->cvals.gpu; m.b = d->lse.gpu; m.c = d->feat_nw.gpu; m.d = d->feat_nb.gpu; m.y = d->feat.gpu;
    misc("d_features", m, B, 1, 64);
    m = {}; m.table = d->feat_up.gpu; m.a = d->feat_up_b.gpu; m.b = d->feat.gpu; m.y = d->fh.gpu;
    misc("d_feat_up", m, NN, 1, 256);
    gemv(d->feat_down, d->fh, DLH, d->tmp, DLH, NN, 0, d->feat_down_b);
    m = {}; m.a = d->tokstate.gpu; m.b = d->passstate.gpu; m.c = d->tmp.gpu; m.d = d->slot.gpu; m.e = d->rank.gpu; m.y = d->nodes.gpu;
    misc("d_nodes", m, NN, 1, 256);
    for (int i = 0; i < 2; i++) {
        LLayer& a = d->ll[i];
        rms(d->nodes, a.an, d->ln, DLH, NN);
        gemv(a.qkv, d->ln, DLH, d->lqkv, 3 * DLH, NN, 0, a.qkv_b);
        m = {}; m.a = d->lqkv.gpu; m.b = d->abias.gpu; m.y = d->lo.gpu;
        misc("d_lat_attn", m, DLHEADS, NN, 128);
        gemv(a.out, d->lo, DLH, d->nodes, DLH, NN, 0, a.out_b, 1);
        rms(d->nodes, a.mn, d->ln, DLH, NN);
        gemv(a.up, d->ln, DLH, d->tmp, 2 * DLH, NN, 2, a.up_b);
        gemv(a.down, d->tmp, 2 * DLH, d->nodes, DLH, NN, 0, a.down_b, 1);
    }
    rms(d->nodes, d->out_norm, d->ln, DLH, NN);
    for (int b = 0; b < B; b++) copyf(d->acat.at((size_t)b * DH * 4), d->anchor.at((size_t)seqs[idx[b]] * DH * 4), DH);
    gemv(d->ctx_p, d->acat, DH, d->astate, DLH, B, 0, d->ctx_b);
    rms(d->astate, d->anc_norm, d->astate, DLH, B);
    m = {}; m.a = d->ln.gpu; m.b = d->astate.gpu; m.y = d->cat.gpu;
    misc("d_cat3", m, NN, 1, 256);
    gemv(d->factor, d->cat, 3 * DLH, d->fh, DLH, NN, 2, d->factor_b);
    {
        Par par;
        gemv(d->out_h, d->fh, DLH, d->edges_out, DLH, NN, 0, d->out_b);
        gemv(d->in_h, d->fh, DLH, d->edges_in, DLH, NN, 0, d->in_b);
    }
    m = {}; m.y = d->edges_out.gpu; m.eps = VEPS;
    misc("d_l2norm", m, NN, 1, 256);
    m.y = d->edges_in.gpu;
    misc("d_l2norm", m, NN, 1, 256);
    gemv(d->anc_h, d->astate, DLH, d->aout, DLH, B, 0, d->anc_b);
    m.y = d->aout.gpu;
    misc("d_l2norm", m, B, 1, 256);
    m = {}; m.a = d->aout.gpu; m.b = d->edges_out.gpu; m.c = d->edges_in.gpu; m.ids = d->cids.gpu; m.out = d->outp.gpu;
    misc("d_walk", m, B, 1, 256);
    finish();
    const int* o = (const int*)d->outp.cpu;
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < DSLOTS; j++) out[idx[b] * 8 + j] = o[b * DSLOTS + j];
        got[idx[b]] = DSLOTS;
    }
}

// ---------------------------------------------------------------------------------------------------------
// The target
// ---------------------------------------------------------------------------------------------------------

namespace {

struct Layer {
    int type = 0;  // 0 mamba, 1 moe, 2 attention
    int mi = -1;   // index among the Mamba layers
    DPtr norm;
    // mamba
    Mat in_proj, out_proj;
    DPtr conv_w, conv_b, A_log, D, dt_bias, gnorm;
    DPtr conv;  // [seq][CONV_SLOTS][CONV_ELEMS]
    // moe
    Mat fc1, fc2, gate, up_s, down_s;
    DPtr gbias;
    int experts = 0, width = 0;
    Slab up, down;
    // attention
    Mat q, k, v, o;
    DPtr kc, vc;  // [seq][context][256] bf16
};

}  // namespace

struct Bolt {
    int context = 65536;
    float eps = 1e-5f, route_scale = 5.f;
    DPtr embed, xemb, norm_f;
    Mat head;
    std::vector<Layer> layers;
    int mamba_layers = 0;
    DPtr ssm, hist_x, hist_dt, A_log_all, dt_bias_all;  // every Mamba layer's, see ReplayArgs

    DPtr x, xn, r1, r2, proj, xbc, ybuf, gout, qb, kb, vb, apart, aout;
    DPtr glog, lat, lat2, rw, act, mo, sact, ridx, order, wfirst, wcount, wexp, nwork;
    DPtr logits, cap, cvals, cids, csv, csi, msel, mmask, mvals, mids;
    DPtr ctl;

    struct Seq {
        bool used = false;
        int len = 0;
        int ssm = 0, conv = 0;
        int last_rows = 0, last_row0 = 0, last_ssm = 0, last_conv = 0, last_len = 0, last_record = 0;
        int ing_pos0 = 0, ing_n = 0, ing_row0 = 0;
        int ingest_floor = 0;
    };
    int nseq = 1;
    std::vector<Seq> seqs;
    int last_cand_k = 0;
    Draft* draft = nullptr;
    double cost_ms[VERIFY_MAX + 1] = {};
    std::vector<std::pair<int, double>> shared_ms;
    double draft_ms = 0, draft_multi_ms = 0;
    bool prof = false;
    double prof_ms = 0;
    int prof_n = 0;
};

namespace {

uint64_t kv_token_bytes() { return 8ull * 2 * 512; }

// The attention keys one split covers for a segment starting at base with rows rows: 128, doubled until 32 splits
// reach the segment's end. For decode and verify windows (rows <= 8) it follows the sequence's length alone.
int chunk_of(int base, int rows) {
    int k = 128;
    while (k * BOLT_ATT_SPLITS < base + std::max(rows, VERIFY_MAX)) k *= 2;
    return k;
}

void encode_block(Bolt* b, int T, int record, int cand_k, int last_only, int nseg, int splits) {
    Ctl* ctl = (Ctl*)b->ctl.cpu;
    const uint64_t ctl_g = b->ctl.gpu;
    EmbedArgs ea{b->embed.gpu, b->xemb.gpu, ctl_g, b->x.gpu};
    disp("embed", ea, T, 1, 1, 256);
    const int L = (int)b->layers.size();
    for (int l = 0; l < L; l++) {
        Layer& ly = b->layers[l];
        DPtr r1 = l == 0 ? DPtr{} : b->r1;
        DPtr r2 = (l > 0 && b->layers[l - 1].type == 1) ? b->r2 : DPtr{};
        DPtr cap;
        if (l >= 1 && aux_slot(l - 1) >= 0) cap = b->cap.at((size_t)aux_slot(l - 1) * HID * 4);
        AddRmsArgs aa{b->x.gpu, r1.gpu, r2.gpu, ly.norm.gpu, b->xn.gpu, cap.gpu, 6 * HID, b->eps};
        disp("add_rms", aa, T, 1, 1, 256);
        if (ly.type == 0) {
            gemv(ly.in_proj, b->xn, HID, b->proj, PROJ, T);
            DPtr hx = record ? b->hist_x.at((size_t)ly.mi * BATCH_MAX * 10240 * 4) : DPtr{};
            DPtr hd = record ? b->hist_dt.at((size_t)ly.mi * BATCH_MAX * 128 * 4) : DPtr{};
            ConvArgs ca{ctl_g, ly.conv.gpu, ly.conv_w.gpu, ly.conv_b.gpu, b->proj.gpu, b->xbc.gpu, hx.gpu, hd.gpu, PROJ, record, T};
            if (nseg == 1 && T >= 64 && !record) disp("conv_long", ca, 10240 / 128, (T + 15) / 16, 1, 128);
            else disp("conv", ca, 40, nseg, 1, 256);
            DPtr pool = b->ssm.at((size_t)ly.mi * b->nseq * 2 * SSM_ELEMS * 4);
            ScanArgs sa{ctl_g, pool.gpu, b->xbc.gpu, b->proj.gpu, ly.A_log.gpu, ly.dt_bias.gpu, ly.D.gpu, b->ybuf.gpu, PROJ, T};
            if (nseg == 1 && T >= 64 && !record) disp("scan_long", sa, 128, 1, 1, 128);
            else disp("scan", sa, 128 * 8, nseg, 1, 256);
            GateArgs gg{b->ybuf.gpu, b->proj.gpu, ly.gnorm.gpu, b->gout.gpu, PROJ, b->eps};
            disp("gate_rms", gg, T * 8, 1, 1, 256);
            gemv(ly.out_proj, b->gout, 8192, b->r1, HID, T);
        } else if (ly.type == 2) {
            {
                Par par;
                gemv(ly.q, b->xn, HID, b->qb, 4096, T);
                gemv(ly.k, b->xn, HID, b->kb, 256, T);
                gemv(ly.v, b->xn, HID, b->vb, 256, T);
            }
            KvArgs ka{ctl_g, b->kb.gpu, b->vb.gpu, ly.kc.gpu, ly.vc.gpu, b->qb.gpu, b->apart.gpu, b->aout.gpu, b->context, T};
            disp("kv_store", ka, T, 1, 1, 128);
            disp("attn_part", ka, splits, 2, T, 256);
            disp("attn_comb", ka, T, 32, 1, 128);
            gemv(ly.o, b->aout, 4096, b->r1, HID, T);
        } else {
            // shared expert (reads the block input's norm), latent projection, router
            {
                Par par;
                gemv(ly.up_s, b->xn, HID, b->sact, ly.up_s.n, T, 1);
                gemv(ly.fc1, b->xn, HID, b->lat, 1024, T);
                gemv(ly.gate, b->xn, HID, b->glog, 512, T);
            }
            {
                Par par;
                gemv(ly.down_s, b->sact, ly.up_s.n, b->r2, HID, T);
                RouteArgs ra{b->glog.gpu, ly.gbias.gpu, b->ridx.gpu, b->rw.gpu, 512, ly.experts, b->route_scale};
                disp("route", ra, T, 1, 1, 32);
            }
            const int pairs = T * TOPN;
            const bool big = T >= 33;   // prompt chunks: items of 32 pairs on the tiled kernel
            const int item_b = pairs <= 9 ? 1 : big ? 32 : 8;
            GroupArgs ga{b->ridx.gpu, b->order.gpu, b->wfirst.gpu, b->wcount.gpu, b->wexp.gpu, b->nwork.gpu, pairs, item_b};
            disp("moe_group", ga, 1, 1, 1, 512);
            const int items = std::min(pairs, ly.experts + pairs / item_b);
            static const int moe_r = getenv("BOLT_MOE_R") ? atoi(getenv("BOLT_MOE_R")) : 2;
            const char* mk = big ? "moe_gemm" : item_b == 1 ? "moe_gemv_1" : moe_r >= 4 ? "moe_gemv_8r4" : "moe_gemv_8";
            const int rpt = big ? 64 : item_b == 1 ? 16 : moe_r >= 4 ? 16 : 8;
            MoeArgs up{ly.up.qs.gpu, ly.up.d.gpu, b->order.gpu, b->wfirst.gpu, b->wcount.gpu, b->wexp.gpu, b->nwork.gpu,
                       b->lat.gpu, b->act.gpu, ly.up.qs_stride, ly.up.d_stride, ly.width, 1024, 1024, ly.width, 0, 1};
            disp(mk, up, (ly.width + rpt - 1) / rpt, items, 1, 128);
            MoeArgs dn{ly.down.qs.gpu, ly.down.d.gpu, b->order.gpu, b->wfirst.gpu, b->wcount.gpu, b->wexp.gpu, b->nwork.gpu,
                       b->act.gpu, b->mo.gpu, ly.down.qs_stride, ly.down.d_stride, 1024, ly.width, ly.width, 1024, 1, 0};
            disp(mk, dn, (1024 + rpt - 1) / rpt, items, 1, 128);
            CombineArgs cb{b->mo.gpu, b->rw.gpu, b->lat2.gpu};
            disp("combine", cb, T, 1, 1, 256);
            gemv(ly.fc2, b->lat2, 1024, b->r1, HID, T);
        }
    }
    AddRmsArgs fa{b->x.gpu, b->r1.gpu, b->r2.gpu, b->norm_f.gpu, b->xn.gpu, 0, 0, b->eps};
    disp("add_rms", fa, T, 1, 1, 256);
    if (cand_k > 0) {
        const int rows = last_only ? 1 : T;
        DPtr hx = last_only ? b->xn.at((size_t)(T - 1) * HID * 4) : b->xn;
        gemv(b->head, hx, HID, b->logits, VOCAB, rows);
        candidates(b->logits, rows, cand_k, b->cvals, b->cids, b->csv, b->csi);
    }
    (void)ctl;
}

void flush_ingest(Bolt* b, int id) {
    Bolt::Seq& q = b->seqs[id];
    if (q.ing_n <= 0) return;
    int pos0 = q.ing_pos0, n = q.ing_n, row0 = q.ing_row0;
    q.ing_n = 0;
    if (!b->draft) return;
    int skip = 0;
    if (pos0 < q.ingest_floor) skip = std::min(q.ingest_floor - pos0, n);
    if (skip >= n) { draft_skip(b->draft, id, pos0 + n); return; }
    if (skip > 0) draft_skip(b->draft, id, pos0 + skip);
    draft_ingest(b->draft, id, b->cap.at((size_t)(row0 + skip) * 6 * HID * 4), pos0 + skip, n - skip);
}

void flush_all_ingest(Bolt* b) {
    int seqs[SEG_MAX], row0[SEG_MAX], n[SEG_MAX], pos0[SEG_MAX], k = 0, nrows = 0;
    for (int id = 0; id < b->nseq; id++) {
        Bolt::Seq& q = b->seqs[id];
        if (q.ing_n <= 0) continue;
        if (b->draft && q.last_record && q.ing_pos0 >= q.ingest_floor) {
            seqs[k] = id; row0[k] = q.ing_row0; n[k] = q.ing_n; pos0[k] = q.ing_pos0;
            nrows = std::max(nrows, q.ing_row0 + q.ing_n);
            k++;
            q.ing_n = 0;
        } else {
            flush_ingest(b, id);
        }
    }
    if (k > 0) draft_ingest_multi(b->draft, b->cap, nrows, k, seqs, row0, n, pos0);
}

struct SegIn { int seq, rows; const int* tokens; };
struct EmbedPatch { int row; const uint16_t* src; int n; };

int run_blocks(Bolt* b, const SegIn* sg, int nseg, int record, int cand_k, int last_only = 0, const EmbedPatch* patches = nullptr, int npatch = 0) {
    if (nseg < 1 || nseg > SEG_MAX || (!record && nseg != 1)) return -1;
    int T = 0;
    for (int i = 0; i < nseg; i++) {
        if (sg[i].seq < 0 || sg[i].seq >= b->nseq || sg[i].rows < 1) return -1;
        for (int j = 0; j < i; j++) if (sg[j].seq == sg[i].seq) return -1;
        if (record && sg[i].rows > VERIFY_MAX) return -1;
        if (b->seqs[sg[i].seq].len + sg[i].rows > b->context) return -2;
        T += sg[i].rows;
    }
    if (T > TMAX || (record && T > BATCH_MAX)) return -1;
    if (cand_k > 0 && !last_only && T > LOGIT_ROWS) return -1;
    const double t0 = b->prof ? now_ms() : 0;
    begin();
    flush_all_ingest(b);
    Ctl* h = (Ctl*)b->ctl.cpu;
    int row = 0, splits = 1;
    for (int i = 0; i < nseg; i++) {
        Bolt::Seq& q = b->seqs[sg[i].seq];
        h->seg_row0[i] = row;
        h->seg_base[i] = q.len;
        h->seg_seq[i] = sg[i].seq;
        h->seg_ssm[i] = q.ssm;
        h->seg_conv[i] = q.conv;
        h->seg_chunk[i] = chunk_of(q.len, sg[i].rows);
        splits = std::max(splits, (q.len + sg[i].rows + h->seg_chunk[i] - 1) / h->seg_chunk[i]);
        for (int t = 0; t < sg[i].rows; t++) h->tokens[row + t] = sg[i].tokens[t];
        row += sg[i].rows;
    }
    h->seg_row0[nseg] = row;
    h->nrows = T;
    h->nseg = nseg;
    for (int i = 0; i < npatch; i++) memcpy(b->xemb.cpu + (size_t)patches[i].row * HID * 2, patches[i].src, (size_t)patches[i].n * HID * 2);
    encode_block(b, T, record, cand_k, last_only, nseg, splits);
    if (!finish()) return -3;
    if (b->prof) { b->prof_ms += now_ms() - t0; b->prof_n++; }
    b->last_cand_k = cand_k;
    for (int i = 0; i < nseg; i++) {
        Bolt::Seq& q = b->seqs[sg[i].seq];
        const int rows = sg[i].rows;
        q.last_rows = rows;
        q.last_row0 = h->seg_row0[i];
        q.last_ssm = q.ssm;
        q.last_conv = q.conv;
        q.last_len = q.len;
        q.last_record = record;
        q.ing_pos0 = q.len;
        q.ing_n = rows;
        q.ing_row0 = h->seg_row0[i];
        q.conv = record ? (q.conv + rows) % CONV_SLOTS : (q.conv + 1) % CONV_SLOTS;
        q.ssm ^= 1;
        q.len += rows;
    }
    return 0;
}

int run_block(Bolt* b, int seq, const int* tokens, int T, int record, int cand_k, int last_only = 0, const EmbedPatch* patches = nullptr, int npatch = 0) {
    SegIn s{seq, T, tokens};
    return run_blocks(b, &s, 1, record, cand_k, last_only, patches, npatch);
}

}  // namespace

static void bolt_calibrate(Bolt* b);

extern "C" Bolt* bolt_open(const char* dir, int context) {
    if (!gpu_init()) return nullptr;
    const double t0 = now_ms();
    fprintf(stderr, "bolt: loading %s\n", dir);
    std::string cfg = slurp(std::string(dir) + "/config.json");
    auto types = block_types(cfg);
    if (types.size() != 88) { fprintf(stderr, "bolt: expected 88 blocks, got %zu\n", types.size()); return nullptr; }
    Store* store = open_store(dir, "language_model.");
    Bolt* b = new Bolt();
    b->context = context > 0 ? std::min(context, 262144) : 65536;
    {
        const char* ns = getenv("BOLT_SEQS");
        b->nseq = std::clamp(ns ? atoi(ns) : 1, 1, SEG_MAX);
        b->seqs.assign(b->nseq, Bolt::Seq());
    }
    b->embed = dev_raw(store, "language_model.backbone.embeddings.weight");
    if (!b->embed) return nullptr;
    b->norm_f = dev_f32(store, "language_model.backbone.norm_f.weight");
    b->head = load_q8(store, "language_model.lm_head");
    b->layers.resize(types.size());
    for (auto& t : types) b->mamba_layers += t == "mamba";
    const int ML = b->mamba_layers;
    b->ssm = alloc((size_t)ML * b->nseq * 2 * SSM_ELEMS * 4);
    b->hist_x = alloc((size_t)ML * BATCH_MAX * 10240 * 4);
    b->hist_dt = alloc((size_t)ML * BATCH_MAX * 128 * 4);
    b->A_log_all = alloc((size_t)ML * 128 * 4);
    b->dt_bias_all = alloc((size_t)ML * 128 * 4);
    int mi = 0, ai = 0, oi = 0;
    for (size_t li = 0; li < types.size(); li++) {
        Layer& ly = b->layers[li];
        std::string p = "language_model.backbone.layers." + std::to_string(li);
        ly.norm = dev_f32(store, p + ".norm.weight");
        if (types[li] == "mamba") {
            ly.type = 0;
            ly.mi = mi;
            ly.in_proj = load_nv4(store, p + ".mixer.in_proj");
            ly.out_proj = load_nv4(store, p + ".mixer.out_proj");
            ly.conv_w = dev_f32(store, p + ".mixer.conv1d.weight");
            ly.conv_b = dev_f32(store, p + ".mixer.conv1d.bias");
            DPtr al = dev_f32(store, p + ".mixer.A_log", 128), db = dev_f32(store, p + ".mixer.dt_bias", 128);
            memcpy(b->A_log_all.cpu + (size_t)mi * 512, al.cpu, 512);
            memcpy(b->dt_bias_all.cpu + (size_t)mi * 512, db.cpu, 512);
            ly.A_log = b->A_log_all.at((size_t)mi * 512);
            ly.dt_bias = b->dt_bias_all.at((size_t)mi * 512);
            ly.D = dev_f32(store, p + ".mixer.D");
            ly.gnorm = dev_f32(store, p + ".mixer.norm.weight");
            ly.conv = alloc((size_t)b->nseq * CONV_SLOTS * CONV_ELEMS * 4);
            mi++;
        } else if (types[li] == "moe") {
            ly.type = 1;
            moe_shape(cfg, oi, ly.experts, ly.width);
            if (ly.experts < 9 || ly.experts > 512 || ly.width % 256 != 0) { fprintf(stderr, "bolt: layer %zu odd moe shape %d x %d\n", li, ly.experts, ly.width); return nullptr; }
            ly.fc1 = load_q8(store, p + ".mixer.fc1_latent_proj");
            ly.fc2 = load_q8(store, p + ".mixer.fc2_latent_proj");
            ly.gate = load_bf16(store, p + ".mixer.gate.weight");
            if (ly.gate.n != ly.experts) { fprintf(stderr, "bolt: layer %zu router has %d rows, config says %d experts\n", li, ly.gate.n, ly.experts); return nullptr; }
            ly.gbias = dev_f32(store, p + ".mixer.gate.e_score_correction_bias");
            ly.up_s = load_iq2(store, p + ".mixer.shared_experts.up_proj");
            ly.down_s = load_iq2(store, p + ".mixer.shared_experts.down_proj");
            if (!load_slab(store, p, "up_proj", ly.experts, ly.width, 1024, ly.up) ||
                !load_slab(store, p, "down_proj", ly.experts, 1024, ly.width, ly.down)) return nullptr;
            oi++;
            if (oi % 10 == 0) fprintf(stderr, "bolt: moe layers loaded %d (%.0f s)\n", oi, (now_ms() - t0) / 1000);
        } else {
            ly.type = 2;
            ly.q = load_q8(store, p + ".mixer.q_proj");
            ly.k = load_q8(store, p + ".mixer.k_proj");
            ly.v = load_q8(store, p + ".mixer.v_proj");
            ly.o = load_q8(store, p + ".mixer.o_proj");
            const size_t bytes = (size_t)b->nseq * b->context * 512;
            ly.kc = alloc(bytes, false);   // pages are committed as the cache fills
            ly.vc = alloc(bytes, false);
            ai++;
        }
    }
    delete store;
    auto fa = [](size_t floats) { return alloc(floats * 4); };
    b->xemb = alloc((size_t)TMAX * HID * 2);
    b->x = fa((size_t)TMAX * HID); b->xn = fa((size_t)TMAX * HID); b->r1 = fa((size_t)TMAX * HID); b->r2 = fa((size_t)TMAX * HID);
    b->proj = fa((size_t)TMAX * PROJ); b->xbc = fa((size_t)TMAX * 10240); b->ybuf = fa((size_t)TMAX * 8192); b->gout = fa((size_t)TMAX * 8192);
    b->qb = fa((size_t)TMAX * 4096); b->kb = fa((size_t)TMAX * 256); b->vb = fa((size_t)TMAX * 256);
    b->apart = fa((size_t)TMAX * 2 * BOLT_ATT_SPLITS * BOLT_ATT_PART); b->aout = fa((size_t)TMAX * 4096);
    b->glog = fa((size_t)TMAX * 512); b->lat = fa((size_t)TMAX * 1024); b->lat2 = fa((size_t)TMAX * 1024);
    b->rw = fa((size_t)TMAX * TOPN); b->act = fa((size_t)TMAX * TOPN * 2560); b->mo = fa((size_t)TMAX * TOPN * 1024);
    b->sact = fa((size_t)TMAX * 5376);
    b->ridx = fa((size_t)TMAX * TOPN); b->order = fa((size_t)TMAX * TOPN); b->wfirst = fa((size_t)TMAX * TOPN);
    b->wcount = fa((size_t)TMAX * TOPN); b->wexp = fa((size_t)TMAX * TOPN); b->nwork = fa(4);
    b->logits = fa((size_t)LOGIT_ROWS * VOCAB); b->cap = fa((size_t)TMAX * 6 * HID);
    b->cvals = fa((size_t)LOGIT_ROWS * CAND); b->cids = fa((size_t)LOGIT_ROWS * CAND);
    b->csv = fa((size_t)LOGIT_ROWS * BOLT_CAND_BLOCKS * CAND); b->csi = fa((size_t)LOGIT_ROWS * BOLT_CAND_BLOCKS * CAND);
    b->msel = fa(LOGIT_ROWS); b->mmask = fa((size_t)LOGIT_ROWS * (VOCAB / 32)); b->mvals = fa((size_t)LOGIT_ROWS * CAND); b->mids = fa((size_t)LOGIT_ROWS * CAND);
    b->ctl = alloc(sizeof(Ctl));
    b->prof = getenv("BOLT_PROF") != nullptr;
    fprintf(stderr, "bolt: weights resident in %.1f s, %d mamba %d moe %d attention, %d sequence%s of context %d, %.1f GiB of Metal buffers\n",
            (now_ms() - t0) / 1000, mi, oi, ai, b->nseq, b->nseq == 1 ? "" : "s", b->context, G.total / 1073741824.0);
    // the drafter: BOLT_DRAFT, else a LiLiCorr folder beside the checkpoint
    std::string ddir = getenv("BOLT_DRAFT") ? getenv("BOLT_DRAFT") : "";
    if (ddir.empty()) {
        std::string d(dir);
        while (!d.empty() && d.back() == '/') d.pop_back();
        const std::string parent = d.substr(0, d.rfind('/'));
        for (std::string c : {d + "-LiLiCorr", parent + "/LiLiCorr", parent + "/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"})
            if (access((c + "/model.safetensors").c_str(), R_OK) == 0) { ddir = c; break; }
    }
    if (!getenv("BOLT_DRAFT_OFF") && !ddir.empty()) b->draft = draft_open(ddir.c_str(), b->embed, b->head, b->nseq);
    if (!b->draft) fprintf(stderr, "bolt: no drafter (BOLT_DRAFT names the LiLiCorr folder)\n");
    if (!getenv("BOLT_NOCAL")) bolt_calibrate(b);
    return b;
}

extern "C" Bolt* bolt_load(const char* dir) {
    const char* c = getenv("BOLT_CONTEXT");
    return bolt_open(dir, c ? atoi(c) : 65536);
}

extern "C" int bolt_abi() { return 2; }
extern "C" int bolt_seqs(Bolt* b) { return b->nseq; }
extern "C" int bolt_context(Bolt* b) { return b->context; }

extern "C" uint64_t bolt_seq_bytes(Bolt* b) {
    return (uint64_t)b->mamba_layers * (2 * (uint64_t)SSM_ELEMS + (uint64_t)CONV_SLOTS * CONV_ELEMS) * 4 + (uint64_t)b->context * kv_token_bytes() +
           (b->draft ? draft_blob_bytes() : 0);
}

extern "C" int bolt_propose_seq(Bolt* b, int seq, int anchor, int* out) {
    if (!b->draft) return 0;
    begin();
    flush_ingest(b, seq);
    int got = 0, pos = b->seqs[seq].len;
    draft_propose_multi(b->draft, 1, &seq, &anchor, &pos, out, &got);
    return got;
}

extern "C" void bolt_propose_multi(Bolt* b, int n, const int* seqs, const int* anchors, int* out, int* got) {
    int pos[SEG_MAX];
    if (!b->draft || n < 1 || n > SEG_MAX) { for (int i = 0; i < n; i++) got[i] = 0; return; }
    begin();
    flush_all_ingest(b);
    for (int i = 0; i < n; i++) pos[i] = b->seqs[seqs[i]].len;
    draft_propose_multi(b->draft, n, seqs, anchors, pos, out, got);
}

extern "C" int bolt_has_draft(Bolt* b) { return b->draft != nullptr; }
extern "C" double bolt_cost(Bolt* b, int width) {
    if (width == 0) return b->draft_ms;
    if (width < 1 || width > VERIFY_MAX) return 0;
    return b->cost_ms[width];
}
extern "C" double bolt_draft_multi_cost(Bolt* b) { return b->draft_multi_ms; }
extern "C" int bolt_shared_costs(Bolt* b, int* rows, double* ms, int cap) {
    int n = 0;
    for (auto& e : b->shared_ms) {
        if (n >= cap) break;
        rows[n] = e.first;
        ms[n] = e.second;
        n++;
    }
    return n;
}

extern "C" void bolt_free(Bolt* b) { (void)b; }

extern "C" void bolt_reset_seq(Bolt* b, int seq) {
    finish();
    Bolt::Seq& q = b->seqs[seq];
    const bool used = q.used;
    q = Bolt::Seq();
    q.used = used;
    for (Layer& ly : b->layers) if (ly.type == 0) {
        memset(b->ssm.cpu + ((size_t)ly.mi * b->nseq + seq) * 2 * SSM_ELEMS * 4, 0, SSM_ELEMS * 4);
        memset(ly.conv.cpu + (size_t)seq * CONV_SLOTS * CONV_ELEMS * 4, 0, CONV_ELEMS * 4);
    }
    draft_reset(b->draft, seq);
}

extern "C" int bolt_seq_alloc(Bolt* b) {
    for (int i = 0; i < b->nseq; i++) {
        if (b->seqs[i].used) continue;
        bolt_reset_seq(b, i);
        b->seqs[i].used = true;
        return i;
    }
    return -1;
}
extern "C" void bolt_seq_release(Bolt* b, int seq) {
    if (seq < 0 || seq >= b->nseq) return;
    b->seqs[seq].used = false;
}

extern "C" int bolt_len_seq(Bolt* b, int seq) { return b->seqs[seq].len; }

extern "C" int bolt_forward_multi(Bolt* b, int nseg, const int* seqs, const int* rows, const int* tokens, int cand_k) {
    SegIn sg[SEG_MAX];
    if (nseg < 1 || nseg > SEG_MAX) return -1;
    int at = 0;
    for (int i = 0; i < nseg; i++) {
        sg[i] = {seqs[i], rows[i], tokens + at};
        at += rows[i];
    }
    return run_blocks(b, sg, nseg, 1, cand_k);
}

extern "C" int bolt_mask_rows(Bolt* b, int n, const int* rows, const uint32_t* masks, int words, int k) {
    if (n < 1 || n > LOGIT_ROWS || words < 1 || words > VOCAB / 32 || k < 1) return -1;
    if (k > CAND) k = CAND;
    for (int i = 0; i < n; i++) if (rows[i] < 0 || rows[i] >= LOGIT_ROWS) return -2;
    memcpy(b->msel.cpu, rows, (size_t)n * 4);
    memcpy(b->mmask.cpu, masks, (size_t)n * words * 4);
    candidates(b->logits, n, k, b->mvals, b->mids, b->csv, b->csi, b->msel, b->mmask, words);
    if (!finish()) return -3;
    for (int i = 0; i < n; i++) {
        memcpy(b->cvals.cpu + (size_t)rows[i] * CAND * 4, b->mvals.cpu + (size_t)i * CAND * 4, (size_t)k * 4);
        memcpy(b->cids.cpu + (size_t)rows[i] * CAND * 4, b->mids.cpu + (size_t)i * CAND * 4, (size_t)k * 4);
    }
    return 0;
}

extern "C" void bolt_candidates(Bolt* b, int row, int k, float* vals, int* ids) {
    if (k > CAND) k = CAND;
    memcpy(vals, b->cvals.cpu + (size_t)row * CAND * 4, (size_t)k * 4);
    memcpy(ids, b->cids.cpu + (size_t)row * CAND * 4, (size_t)k * 4);
}

extern "C" void bolt_keep_seq(Bolt* b, int seq, int n) {
    Bolt::Seq& q = b->seqs[seq];
    n = std::clamp(n, 0, q.last_rows);
    if (!q.last_record) return;
    if (n < q.last_rows) {
        if (n > 0) {
            ReplayArgs ra{b->ssm.gpu, b->hist_x.gpu, b->hist_dt.gpu, b->A_log_all.gpu, b->dt_bias_all.gpu, b->nseq, seq, q.last_ssm, q.last_row0, n};
            disp("scan_replay", ra, 128 * 8, b->mamba_layers, 1, 256);
            finish();
        }
        q.ssm = n == 0 ? q.last_ssm : (q.last_ssm ^ 1);
        q.last_rows = n;
    }
    q.conv = (q.last_conv + n) % CONV_SLOTS;
    q.len = q.last_len + n;
    q.ing_n = n;
}

extern "C" void bolt_prompt_end(Bolt* b, int seq, int total) { b->seqs[seq].ingest_floor = total > 1100 ? total - 1100 : 0; }

extern "C" int bolt_prefill_seq(Bolt* b, int seq, const int* tokens, int n, int cand_k) {
    int done = 0;
    while (done < n) {
        const int t = std::min(n - done, TMAX);
        const int last = done + t == n;
        const int rc = run_block(b, seq, tokens + done, t, 0, last ? cand_k : 0, 1);
        if (rc) return rc;
        done += t;
    }
    return 0;
}

struct BoltRowRun { int pos, count; const void* rows; };

// Prompt rows whose embeddings are given (bf16 [count][4096] in memory the CPU can read): images and video.
extern "C" int bolt_prefill_seq_rows(Bolt* b, int seq, const int* tokens, int n, int cand_k, const BoltRowRun* runs, int nruns) {
    int done = 0;
    std::vector<int> toks(tokens, tokens + n);
    std::vector<EmbedPatch> patches;
    while (done < n) {
        const int t = std::min(n - done, TMAX);
        patches.clear();
        for (int i = 0; i < nruns; i++) {
            const int lo = std::max(runs[i].pos, done), hi = std::min(runs[i].pos + runs[i].count, done + t);
            if (hi <= lo) continue;
            patches.push_back({lo - done, (const uint16_t*)runs[i].rows + (size_t)(lo - runs[i].pos) * HID, hi - lo});
            for (int p = lo; p < hi; p++) toks[p] = -1;
        }
        const int last = done + t == n;
        const int rc = run_block(b, seq, toks.data() + done, t, 0, last ? cand_k : 0, 1, patches.data(), (int)patches.size());
        if (rc) return rc;
        done += t;
    }
    return 0;
}

extern "C" void bolt_aux(Bolt* b, int row, float* out) { memcpy(out, b->cap.cpu + (size_t)row * 6 * HID * 4, 6 * HID * 4); }

// ---- single-sequence API (sequence 0) for the bench tools ----

extern "C" void bolt_reset(Bolt* b) { bolt_reset_seq(b, 0); }
extern "C" int bolt_len(Bolt* b) { return b->seqs[0].len; }
extern "C" int bolt_propose(Bolt* b, int anchor, int* out) { return bolt_propose_seq(b, 0, anchor, out); }
extern "C" void bolt_keep(Bolt* b, int n) { bolt_keep_seq(b, 0, n); }
extern "C" int bolt_prefill(Bolt* b, const int* tokens, int n, int cand_k) {
    bolt_prompt_end(b, 0, b->seqs[0].len + n);
    return bolt_prefill_seq(b, 0, tokens, n, cand_k);
}
extern "C" int bolt_forward_seq(Bolt* b, int seq, const int* tokens, int T, int record, int cand_k) { return run_block(b, seq, tokens, T, record, cand_k); }
extern "C" int bolt_forward(Bolt* b, const int* tokens, int T, int record, int cand_k) { return run_block(b, 0, tokens, T, record, cand_k); }
extern "C" void bolt_push(Bolt* b, uint32_t token) {
    int t = (int)token;
    run_block(b, 0, &t, 1, 0, 1);
}
extern "C" int bolt_sample(Bolt* b) { return ((int*)b->cids.cpu)[0]; }

// The full logits of a row of the last block (for checks against a reference).
extern "C" void bolt_logits_row(Bolt* b, int row, float* out) { memcpy(out, b->logits.cpu + (size_t)row * VOCAB * 4, (size_t)VOCAB * 4); }

extern "C" float bolt_logprob_row(Bolt* b, int row, int target) {
    const float* h = b->logits.f() + (size_t)row * VOCAB;
    float m = -1e30f;
    for (int i = 0; i < VOCAB; i++) m = std::max(m, h[i]);
    double s = 0;
    for (int i = 0; i < VOCAB; i++) s += exp((double)h[i] - m);
    return (float)(h[target] - m - log(s));
}
extern "C" float bolt_logprob(Bolt* b, int target) { return bolt_logprob_row(b, 0, target); }

extern "C" void bolt_probe(Bolt* b, const int* toks, int n, int* stats) {
    for (int i = 0; i < n; i++) {
        if (i > 0 && i + 8 <= n) {
            int out[8];
            if (bolt_propose(b, toks[i], out) == 7) {
                int a = 0;
                for (int j = 0; j < 7; j++) if (out[j] == toks[i + 1 + j]) stats[j]++;
                while (a < 7 && out[a] == toks[i + 1 + a]) a++;
                stats[8 + a]++;
                stats[20]++;
            }
        }
        bolt_push(b, (uint32_t)toks[i]);
    }
}

// Time each verify width and one proposal so the round loop's cost model starts from measurements, and shared
// blocks of several sequences' windows.
static void bolt_calibrate(Bolt* b) {
    int toks[BATCH_MAX];
    for (int i = 0; i < BATCH_MAX; i++) toks[i] = 1000 + 37 * (i % 61);
    for (int w = 1; w <= VERIFY_MAX; w++) {
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = now_ms();
            if (run_block(b, 0, toks, w, 1, 20)) break;
            double t1 = now_ms();
            bolt_keep_seq(b, 0, 0);
            b->seqs[0].ing_n = 0;
            if (rep >= 2 && t1 - t0 < best) best = t1 - t0;
        }
        b->cost_ms[w] = best;
    }
    if (b->nseq > 1) {
        int mix[BATCH_MAX];
        for (int i = 0; i < BATCH_MAX; i++) mix[i] = 1000 + (int)((i * 2654435761u + 12345u) % 90000u);
        for (int R : {2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128}) {
            int nseg = std::min(R, b->nseq);
            if (R > nseg * VERIFY_MAX) continue;
            int seqs[SEG_MAX], rows[SEG_MAX];
            for (int i = 0; i < nseg; i++) { seqs[i] = i; rows[i] = R / nseg + (i < R % nseg ? 1 : 0); }
            double best = 1e30;
            for (int rep = 0; rep < 4; rep++) {
                double t0 = now_ms();
                if (bolt_forward_multi(b, nseg, seqs, rows, mix, 64)) break;
                double t1 = now_ms();
                for (int i = 0; i < nseg; i++) { bolt_keep_seq(b, seqs[i], 0); b->seqs[seqs[i]].ing_n = 0; }
                if (rep >= 1 && t1 - t0 < best) best = t1 - t0;
            }
            if (best < 1e29) b->shared_ms.push_back({R, best});
        }
    }
    if (b->draft) {
        bolt_forward(b, toks, 8, 1, 1);
        bolt_keep(b, 8);
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            int out[8];
            double t0 = now_ms();
            bolt_propose(b, toks[0], out);
            double t1 = now_ms();
            if (rep >= 1 && t1 - t0 < best) best = t1 - t0;
        }
        b->draft_ms = best;
        if (b->nseq > 1) {
            int seqs[SEG_MAX], anchors[SEG_MAX], out[SEG_MAX * 8], got[SEG_MAX];
            for (int i = 0; i < b->nseq; i++) {
                seqs[i] = i;
                anchors[i] = toks[0];
                if (i > 0) { bolt_forward_seq(b, i, toks, 8, 1, 1); bolt_keep_seq(b, i, 8); }
            }
            double bestn = 1e30;
            for (int rep = 0; rep < 4; rep++) {
                double t0 = now_ms();
                bolt_propose_multi(b, b->nseq, seqs, anchors, out, got);
                double t1 = now_ms();
                if (rep >= 1 && t1 - t0 < bestn) bestn = t1 - t0;
            }
            b->draft_multi_ms = bestn;
        }
    }
    for (int i = 0; i < b->nseq; i++) bolt_reset_seq(b, i);
    fprintf(stderr, "bolt: window ms");
    for (int w = 1; w <= VERIFY_MAX; w++) fprintf(stderr, " %d:%.1f", w, b->cost_ms[w]);
    fprintf(stderr, "; proposal %.1f ms", b->draft_ms);
    if (b->draft_multi_ms > 0) fprintf(stderr, " (%d streams %.1f ms)", b->nseq, b->draft_multi_ms);
    if (!b->shared_ms.empty()) {
        fprintf(stderr, "; shared rows");
        for (auto& e : b->shared_ms) fprintf(stderr, " %d:%.1f", e.first, e.second);
    }
    fprintf(stderr, "\n");
}

extern "C" void bolt_prof_report(Bolt* b) {
    if (b->prof_n) fprintf(stderr, "bolt prof: %d blocks, %.2f ms each on the wall, %.2f ms of GPU time, %.0f dispatches\n", b->prof_n,
                           b->prof_ms / b->prof_n, G.gpu_ms / b->prof_n, (double)G.dispatches / b->prof_n);
    std::vector<std::pair<std::string, std::pair<double, int>>> v(G.by_kernel.begin(), G.by_kernel.end());
    std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.second.first > y.second.first; });
    double tot = 0;
    for (auto& e : v) tot += e.second.first;
    for (auto& e : v)
        fprintf(stderr, "  %-16s %9.2f ms %5.1f%%  %7d calls  %7.1f us each\n", e.first.c_str(), e.second.first, 100 * e.second.first / tot, e.second.second, 1000 * e.second.first / e.second.second);
    G.by_kernel.clear();
    G.gpu_ms = 0;
    G.dispatches = 0;
    b->prof_ms = 0;
    b->prof_n = 0;
}

// ---- prompt-cache snapshots: the state after `at` prompt tokens, copied out and back ----

struct Snap {
    int at = 0, draft_clen = 0;
    std::vector<uint8_t> data;   // ssm, conv, every attention layer's K and V rows [0, at), the drafter's rings
};

extern "C" uint64_t bolt_snap_bytes(Bolt* b, int at) {
    return (uint64_t)b->mamba_layers * (SSM_ELEMS + CONV_ELEMS) * 4 + (uint64_t)at * kv_token_bytes() + (b->draft ? draft_blob_bytes() : 0);
}

extern "C" void* bolt_snap_save_seq(Bolt* b, int seq, int at) {
    Bolt::Seq& q = b->seqs[seq];
    if (at != q.len || at <= 0) return nullptr;
    begin();
    flush_ingest(b, seq);
    finish();
    Snap* sn = new Snap();
    sn->at = at;
    sn->draft_clen = draft_clen(b->draft, seq);
    sn->data.resize(bolt_snap_bytes(b, at));
    uint8_t* p = sn->data.data();
    const size_t kvb = (size_t)at * 512, kv0 = (size_t)seq * b->context * 512;
    for (Layer& ly : b->layers) {
        if (ly.type == 0) {
            memcpy(p, b->ssm.cpu + (((size_t)ly.mi * b->nseq + seq) * 2 + q.ssm) * SSM_ELEMS * 4, SSM_ELEMS * 4); p += SSM_ELEMS * 4;
            memcpy(p, ly.conv.cpu + ((size_t)seq * CONV_SLOTS + q.conv) * CONV_ELEMS * 4, CONV_ELEMS * 4); p += CONV_ELEMS * 4;
        } else if (ly.type == 2) {
            memcpy(p, ly.kc.cpu + kv0, kvb); p += kvb;
            memcpy(p, ly.vc.cpu + kv0, kvb); p += kvb;
        }
    }
    draft_save(b->draft, seq, p);
    return sn;
}

extern "C" int bolt_snap_restore_seq(Bolt* b, int seq, void* handle) {
    Snap* sn = (Snap*)handle;
    if (!sn || sn->at > b->context) return -1;
    finish();
    Bolt::Seq& q = b->seqs[seq];
    const uint8_t* p = sn->data.data();
    const size_t kvb = (size_t)sn->at * 512, kv0 = (size_t)seq * b->context * 512;
    q.ssm = 0;
    q.conv = 0;
    for (Layer& ly : b->layers) {
        if (ly.type == 0) {
            memcpy(b->ssm.cpu + ((size_t)ly.mi * b->nseq + seq) * 2 * SSM_ELEMS * 4, p, SSM_ELEMS * 4); p += SSM_ELEMS * 4;
            memcpy(ly.conv.cpu + (size_t)seq * CONV_SLOTS * CONV_ELEMS * 4, p, CONV_ELEMS * 4); p += CONV_ELEMS * 4;
        } else if (ly.type == 2) {
            memcpy(ly.kc.cpu + kv0, p, kvb); p += kvb;
            memcpy(ly.vc.cpu + kv0, p, kvb); p += kvb;
        }
    }
    draft_restore(b->draft, seq, p, sn->draft_clen);
    q.len = sn->at;
    q.last_rows = 0;
    q.ing_n = 0;
    return 0;
}

extern "C" void* bolt_snap_save(Bolt* b, int at) { return bolt_snap_save_seq(b, 0, at); }
extern "C" int bolt_snap_restore(Bolt* b, void* handle) { return bolt_snap_restore_seq(b, 0, handle); }
extern "C" void bolt_snap_drop(Bolt*, void* handle) { delete (Snap*)handle; }

// The server's idle keepalive (--keep-warm): one tiny command buffer on Bolt's queue, so the GPU and the wired weights
// stay ready between requests (a lone request after a second of quiet otherwise starts about 170 ms later on an M5 Ultra).
extern "C" void bolt_tick(Bolt*) {
    @autoreleasepool {
        static id<MTLBuffer> scratch = [G.dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        static id<MTLComputePipelineState> copy = pso("copy_f32");   // first tick: idle, the engine thread is not encoding
        if (getenv("BOLT_TICK_BLIT")) {
            id<MTLCommandBuffer> cb = [G.q commandBuffer];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            [e fillBuffer:scratch range:NSMakeRange(0, 64) value:0];
            [e endEncoding];
            [cb commit];
            return;
        }
        id<MTLCommandBuffer> cb = [G.q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e useResources:G.bufs.data() count:G.bufs.size() usage:MTLResourceUsageRead];
        [e useResource:scratch usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        CopyArgs a{};
        a.src = scratch.gpuAddress;
        a.dst = scratch.gpuAddress + 32;
        a.n = 1;
        [e setComputePipelineState:copy];
        [e setBytes:&a length:sizeof(a) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
    }
}

