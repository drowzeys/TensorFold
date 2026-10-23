//! The Drama backbone's per-call activations: regions of one device allocation, sized for the stored formats.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("backbone_config.zig");
const weights = @import("backbone_weights.zig");
const kernels = @import("backbone_kernels.zig");
const gdn = @import("backbone_gdn.zig");

const H = config.hidden;
const top = config.top_k;

/// Bump offsets in one device allocation, 256-byte aligned.
const Bump = struct {
    used: usize = 0,

    fn take(b: *Bump, bytes: usize) usize {
        const at = std.mem.alignForward(usize, b.used, 256);
        b.used = at + bytes;
        return at;
    }
};

/// Per-call activations for up to `max_rows` rows (offsets until `place`, then device addresses).
pub const Scratch = struct {
    ids: u64 = 0, // i32 [R]
    x: u64 = 0, // bf16 [R][H] layer delta (embedding, mixer or MoE output)
    residual: u64 = 0, // bf16 [R][H]
    normed: u64 = 0, // bf16 [R][H]
    proj: u64 = 0, // bf16 [R][max(gdn_in_rows, attn_in_rows)]
    conv: u64 = 0, // bf16 [R][conv_dim]
    core: u64 = 0, // bf16 [R][value_dim] recurrence output
    gated: u64 = 0, // bf16 [R][4096] gated core (DeltaNet) or gated attention, the out projection's input
    q: u64 = 0, // bf16 [R][heads][head_dim]
    moe_in: u64 = 0, // bf16 [R][moe_in_rows]
    picks: u64 = 0, // i32 [P]
    routed: u64 = 0, // f32 [P]
    xq: u64 = 0, // e4m3 [R][H]
    xs: u64 = 0, // f32 [R][H / 128]
    members: u64 = 0, // i32 [P]
    items: u64 = 0, // i32 [3 * max items]
    counts: u64 = 0, // i32 [2]
    rank: u64 = 0, // i32 [P]
    hist: u64 = 0, // i32 [ceil(P / 1024) * experts]
    gu: u64 = 0, // bf16 [P][gu_rows]
    hq: u64 = 0, // e4m3 [P][expert_inner]
    hs: u64 = 0, // f32 [P][expert_inner / 128]
    down: u64 = 0, // bf16 [P][H]
    act: u64 = 0, // bf16 [P][expert_inner] NVFP4 experts' SwiGLU rows, laid out only when a layer is NVFP4
    down_f32: u64 = 0, // f32 [P][H] NVFP4 experts' unweighted down rows, likewise
    shared_h: u64 = 0, // bf16 [R][shared_inner]
    shared_out: u64 = 0, // bf16 [R][H]
    gdn: u64 = 0, // the DeltaNet prefill workspace (gdn.Workspace), shared by every layer
    ba: u64 = 0, // bf16 [R][64] b | a beside an FP8 gdn_in's qkv | z, laid out only when a layer stores one
    router: u64 = 0, // bf16 [R][experts] router logits beside an FP8 shared gate | up, likewise
    part: u64 = 0, // the lane's slice partials for its largest FP8 call when it reduces them, likewise
    slots: u64 = 0, // i32 [P] the picks' held slots, laid out only on a compact checkpoint

    pub fn layout(rows: usize, x: Extra) struct { Scratch, usize } {
        const pairs = rows * top;
        const proj_cols = @max(weights.gdn_in_rows, weights.attn_in_rows);
        const items = cuda.grouped.maxItems(pairs, config.experts, tile);
        const blocks = (pairs + 1023) / 1024;
        var b: Bump = .{};
        var s: Scratch = .{};
        s.ids = b.take(rows * 4);
        s.x = b.take(rows * H * 2);
        s.residual = b.take(rows * H * 2);
        s.normed = b.take(rows * H * 2);
        s.proj = b.take(rows * proj_cols * 2);
        s.conv = b.take(rows * config.conv_dim * 2);
        s.core = b.take(rows * config.value_dim * 2);
        s.gated = b.take(rows * @max(config.value_dim, config.query_heads * config.head_dim) * 2);
        s.q = b.take(rows * config.query_heads * config.head_dim * 2);
        s.moe_in = b.take(rows * weights.moe_in_rows * 2);
        s.picks = b.take(pairs * 4);
        s.routed = b.take(pairs * 4);
        s.xq = b.take(rows * H);
        s.xs = b.take(rows * (H / config.block) * 4);
        s.members = b.take(pairs * 4);
        s.items = b.take(3 * items * 4);
        s.counts = b.take(2 * 4);
        s.rank = b.take(pairs * 4);
        s.hist = b.take(blocks * config.experts * 4);
        s.gu = b.take(pairs * weights.gu_rows * 2);
        s.hq = b.take(pairs * config.expert_inner);
        s.hs = b.take(pairs * (config.expert_inner / config.block) * 4);
        s.down = b.take(pairs * H * 2);
        if (x.fp4) {
            s.act = b.take(pairs * config.expert_inner * 2);
            s.down_f32 = b.take(pairs * H * 4);
        }
        s.shared_h = b.take(rows * config.shared_inner * 2);
        s.shared_out = b.take(rows * H * 2);
        s.gdn = b.take(gdnWorkspace(rows)[1]);
        if (x.ba) s.ba = b.take(rows * 2 * config.linear_value_heads * 2);
        if (x.router) s.router = b.take(rows * config.experts * 2);
        if (x.part > 0) s.part = b.take(x.part);
        if (x.slots) s.slots = b.take(pairs * 4);
        return .{ s, b.used };
    }

    pub fn place(s: *Scratch, base: u64) void {
        inline for (@typeInfo(Scratch).@"struct".field_names) |name| @field(s, name) += base;
    }
};

/// Scratch the stored formats add: NVFP4 experts' rows, the rows split off FP8 groups, the lane's partials, held slots.
pub const Extra = struct { fp4: bool = false, ba: bool = false, router: bool = false, part: usize = 0, slots: bool = false };

pub fn extra(k: *const kernels.Kernels, w: *const weights.Weights, rows: usize) Extra {
    var x: Extra = .{ .fp4 = w.nvfp4Layers() > 0, .slots = w.held != null };
    for (w.layers) |l| {
        x.ba = x.ba or l.fp8.gdn_in != null;
        x.router = x.router or l.fp8.shared_in != null;
    }
    if (k.lane) |lane| x.part = partBound(lane, w, rows);
    return x;
}

/// The lane partials of the largest FP8 call up to `rows` rows: they grow until the fused form, which needs none.
fn partBound(lane: cuda.qmmf.Lane, w: *const weights.Weights, rows: usize) usize {
    const most = @min(rows, cuda.qmmf.fused_rows - 1);
    var part: usize = 0;
    for (w.layers) |l| inline for (weights.dense.group_names) |g| if (@field(l.fp8, g)) |q| {
        part = @max(part, lane.partBytes(most, q.fp8g));
    };
    return part;
}

pub const tile = 16; // routed pairs an expert item takes (the plan's decode tile; the expert kernel's 16 mma rows)

pub fn gdnWorkspace(rows: usize) struct { gdn.Workspace, usize } {
    return gdn.Workspace.layout(rows, config.linear_key_heads, config.linear_value_heads);
}

test "scratch layout keeps every region 256-byte aligned and disjoint; format regions only when asked, at the end" {
    const formats = .{ Extra{}, Extra{ .fp4 = true }, Extra{ .ba = true, .part = 4096 }, Extra{ .fp4 = true, .router = true }, Extra{ .part = 64, .slots = true } };
    inline for (formats) |x| {
        const s, const total = Scratch.layout(128, x);
        var last: u64 = 0;
        inline for (@typeInfo(Scratch).@"struct".field_names, 0..) |name, i| {
            const at = @field(s, name);
            const eql = struct {
                fn f(comptime a: []const u8, comptime b: []const u8) bool {
                    return comptime std.mem.eql(u8, a, b);
                }
            }.f;
            const skipped = (!x.fp4 and (eql(name, "act") or eql(name, "down_f32"))) or (!x.ba and eql(name, "ba")) or
                (!x.router and eql(name, "router")) or (x.part == 0 and eql(name, "part")) or (!x.slots and eql(name, "slots"));
            try std.testing.expect(at % 256 == 0);
            if (i > 0 and !skipped) try std.testing.expect(at > last);
            if (!skipped) last = at;
        }
        try std.testing.expect(total > last);
    }
    // a model without FP8 groups keeps every offset and the size it had before they existed
    const plain, const plain_total = Scratch.layout(128, .{ .fp4 = true });
    const fp8, const fp8_total = Scratch.layout(128, .{ .fp4 = true, .ba = true, .router = true, .part = 1 << 20 });
    inline for (@typeInfo(Scratch).@"struct".field_names) |name| {
        if (comptime !(std.mem.eql(u8, name, "ba") or std.mem.eql(u8, name, "router") or std.mem.eql(u8, name, "part")))
            try std.testing.expectEqual(@field(plain, name), @field(fp8, name));
    }
    try std.testing.expect(fp8.ba >= plain_total and fp8_total > plain_total);
    // and one that holds a compact expert map adds only its slots, past every other region
    const held, const held_total = Scratch.layout(128, .{ .fp4 = true, .slots = true });
    inline for (@typeInfo(Scratch).@"struct".field_names) |name| {
        if (comptime !std.mem.eql(u8, name, "slots")) try std.testing.expectEqual(@field(plain, name), @field(held, name));
    }
    try std.testing.expect(held.slots >= plain_total and held_total == held.slots + 128 * top * 4);
}

test "part holds every FP8 call's partials at every row count up to the capacity, past the fused transition too" {
    var w: weights.Weights = .{ .gpa = std.testing.allocator };
    const fp8 = cuda.fp8.weight;
    w.layers[0].fp8 = .{ .gdn_in = .{ .fp8g = fp8(0, 0, 12288, H) }, .gdn_out = .{ .fp8g = fp8(0, 0, H, 4096) }, .shared_in = .{ .fp8g = fp8(0, 0, 1024, H) }, .shared_out = .{ .fp8g = fp8(0, 0, H, 512) } };
    w.layers[3].fp8 = .{ .attn_in = .{ .fp8g = fp8(0, 0, 9216, H) }, .attn_out = .{ .fp8g = fp8(0, 0, H, 4096) } };
    for ([_]bool{ false, true }) |clusters| {
        // partBytes reads only the plan's inputs; the module handles are never touched
        const lane: cuda.qmmf.Lane = .{ .mod = undefined, .fns = undefined, .reduce = undefined, .clusters = clusters };
        for ([_]usize{ 1, 7, 16, 17, 128, 255, 256, 300, 1024 }) |cap| {
            const bound = partBound(lane, &w, cap);
            try std.testing.expect(clusters == (bound == 0)); // GB10's clusters reduce in shared memory
            for (1..cap + 1) |m| for (w.layers) |l| inline for (weights.dense.group_names) |g| if (@field(l.fp8, g)) |q| {
                try std.testing.expect(lane.partBytes(m, q.fp8g) <= bound);
            };
        }
    }
}
