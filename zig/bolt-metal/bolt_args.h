// Argument blocks shared by libbolt's Metal host (bolt_metal.mm) and its kernels (bolt.metal).
// Device addresses are the buffers' gpuAddress plus an offset; in a kernel they are `device` pointers.
// Every struct puts its pointers first so both compilers lay it out the same way.
#pragma once

#ifdef __METAL_VERSION__
#define DP(T) device T*
#define U64 ulong
#else
#include <cstdint>
#define DP(T) uint64_t
#define U64 uint64_t
#endif

#define BOLT_TMAX 1024       // most rows one block forward carries (prompt chunks)
#define BOLT_BATCH_MAX 128   // most rows of a block of several sequences' verify windows
#define BOLT_LOGIT_ROWS 128  // rows whose logits are kept
#define BOLT_SEG_MAX 16      // most sequences in one block
#define BOLT_VERIFY_MAX 8    // most rows of one sequence's verify window
#define BOLT_CONV_SLOTS 9    // conv state slots a sequence keeps
#define BOLT_HID 4096
#define BOLT_VOCAB 131072
#define BOLT_TOPN 9
#define BOLT_CAND 64
#define BOLT_PROJ 18560      // Mamba in_proj rows: z 8192, xBC 10240, dt 128
#define BOLT_SSM_ELEMS (128 * 64 * 128)
#define BOLT_CONV_ELEMS (10240 * 3)
#define BOLT_ATT_SPLITS 32
#define BOLT_ATT_PART (16 + 16 + 16 * 128)  // m[16], l[16], o[16][128]
#define BOLT_CAND_BLOCKS 128

// What changes between blocks: tokens, and per segment its first row, cache length, sequence and state slots.
struct Ctl {
    int tokens[BOLT_TMAX];
    int nrows;
    int nseg;
    int seg_row0[BOLT_SEG_MAX + 1];
    int seg_base[BOLT_SEG_MAX];
    int seg_seq[BOLT_SEG_MAX];
    int seg_ssm[BOLT_SEG_MAX];
    int seg_conv[BOLT_SEG_MAX];
    int seg_chunk[BOLT_SEG_MAX];  // attention keys a split covers for this segment's rows
};

// y[t][row] = epi(sum_k W[row][k] * x[t][k] + bias[row]); epi 0 none, 1 relu^2, 2 silu.
struct GemvArgs {
    DP(const uchar) w;
    DP(const uchar) s;
    DP(const float) x;
    DP(float) y;
    DP(const float) bias;
    float s2;
    int n, k, ldx, ldy, T, epi, accum;
};

// Routed experts: every work item is one expert and up to 8 of its (token, expert) pairs.
struct MoeArgs {
    DP(const uchar) qs;
    DP(const half) d;
    DP(const int) order;
    DP(const int) wfirst;
    DP(const int) wcount;
    DP(const int) wexp;
    DP(const int) nwork;
    DP(const float) x;
    DP(float) y;
    U64 qs_stride, d_stride;  // bytes / halves between experts
    int n, k, ldx, ldy, x_per_pair, epi;
};

struct GroupArgs {
    DP(const int) ridx;
    DP(int) order;
    DP(int) wfirst;
    DP(int) wcount;
    DP(int) wexp;
    DP(int) nwork;
    int pairs, item_b;
};

struct EmbedArgs {
    DP(const ushort) table;
    DP(const ushort) extra;
    DP(const Ctl) ctl;
    DP(float) x;
};

struct AddRmsArgs {
    DP(float) x;
    DP(const float) r1;
    DP(const float) r2;
    DP(const float) w;
    DP(float) xn;
    DP(float) cap;
    int cap_stride;
    float eps;
};

struct ConvArgs {
    DP(const Ctl) ctl;
    DP(float) pool;  // this layer's ring: [seq][CONV_SLOTS][CONV_ELEMS]
    DP(const float) w;
    DP(const float) b;
    DP(const float) proj;
    DP(float) xbc;
    DP(float) hist_x;
    DP(float) hist_dt;
    int ldp, record, T;
};

struct ScanArgs {
    DP(const Ctl) ctl;
    DP(float) pool;  // this layer's SSM state: [seq][2][SSM_ELEMS]
    DP(const float) xbc;
    DP(const float) proj;
    DP(const float) A_log;
    DP(const float) dt_bias;
    DP(const float) D;
    DP(float) y;
    int ldp, T;
};

struct GateArgs {
    DP(const float) y;
    DP(const float) proj;
    DP(const float) w;
    DP(float) out;
    int ldp;
    float eps;
};

struct RouteArgs {
    DP(const float) logits;
    DP(const float) bias;
    DP(int) ridx;
    DP(float) rw;
    int ldl, E;
    float scale;
};

struct CombineArgs {
    DP(const float) o;
    DP(const float) rw;
    DP(float) out;
};

struct KvArgs {
    DP(const Ctl) ctl;
    DP(const float) k;
    DP(const float) v;
    DP(ushort) kc;
    DP(ushort) vc;
    DP(const float) q;
    DP(float) part;
    DP(float) out;
    int ctx, T;
};

struct CandArgs {
    DP(const float) logits;
    DP(const int) sel;
    DP(const uint) masks;
    DP(float) sv;
    DP(int) si;
    DP(float) vals;
    DP(int) ids;
    int topk, words;
};

// Mamba state of every layer in one buffer each: ssm [layer][seq][2][SSM_ELEMS], hist_x [layer][BATCH_MAX][10240],
// hist_dt [layer][BATCH_MAX][128], A_log and dt_bias [layer][128].
struct ReplayArgs {
    DP(float) ssm;
    DP(const float) hist_x;
    DP(const float) hist_dt;
    DP(const float) A_log;
    DP(const float) dt_bias;
    int nseq, seq, slot, row0, n;
};

struct CopyArgs {
    DP(const float) src;
    DP(float) dst;
    uint n;
    int add;  // 1: dst += src
};

// ---- LiLiCorr drafter ----

struct RmsArgs {
    DP(const float) x;
    DP(const float) w;
    DP(float) y;
    int n;
    float eps;
};

struct GconvArgs {
    DP(const float) x;
    DP(const float) base;
    DP(const float) delta;
    DP(float) y;
    int T, side;
};

struct RopeArgs {
    DP(float) x;
    DP(const float) w;
    DP(const int) posv;    // mode 1: block of row t starts at posv[t / 8]
    DP(const int) rowpos;  // mode 2: row t at rowpos[t]
    int heads, pos0, mode;
};

struct DattnArgs {
    DP(const float) q;
    DP(const float) bk;
    DP(const float) bv;
    DP(const float) ringk;  // [seq][layer][RING][2][128]
    DP(const float) ringv;
    DP(const float) sink;
    DP(float) out;
    DP(const int) seqs;  // [stream] sequence of each stream in the call
    DP(const int) clen;
    DP(const int) pos;
    int layer, nlayers;
};

struct DraftMiscArgs {
    DP(const float) a;
    DP(const float) b;
    DP(const float) c;
    DP(const float) d;
    DP(const float) e;
    DP(float) y;
    DP(const ushort) table;
    DP(const int) ids;
    DP(int) out;
    int n, rows;
    float eps;
};

struct RingArgs {
    DP(const float) k;
    DP(const float) v;
    DP(float) ringk;
    DP(float) ringv;
    DP(const int) rowseq;
    DP(const int) rowpos;
    int seq, layer, nlayers, pos0;
};
