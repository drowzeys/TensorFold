//! Full GLM-5.3 (model_type glm_moe_dsa) dimensions from config.json, as the Python engine's config.py reads them.
const std = @import("std");

pub const max_layers = 96;

pub const Config = struct {
    hidden: usize, // hidden_size
    layers: usize, // num_hidden_layers (the MTP layer follows at this index)
    first_dense: usize, // first_k_dense_replace: layers below it run a dense MLP
    dense_width: usize, // intermediate_size
    expert_width: usize, // moe_intermediate_size
    experts: usize, // n_routed_experts
    shared_experts: usize, // n_shared_experts
    top_k: usize, // num_experts_per_tok
    routed_scaling: f32,
    norm_topk: bool,
    heads: usize, // num_attention_heads (all ranks)
    q_lora: usize,
    kv_lora: usize,
    nope: usize, // qk_nope_head_dim
    rope: usize, // qk_rope_head_dim
    v_dim: usize, // v_head_dim
    rope_theta: f64,
    index_heads: usize,
    index_dim: usize,
    index_topk: usize,
    full_index: [max_layers]bool = @splat(false), // indexer_types[i] == "full"
    eps: f32, // rms_norm_eps
    vocab: usize,
    mtp_layers: usize = 0, // num_nextn_predict_layers: the MTP layer sits at index `layers`

    /// One token's MLA cache row: the kv latent and the shared RoPE key (576).
    pub fn latentWidth(self: Config) usize {
        return self.kv_lora + self.rope;
    }

    pub fn qkDim(self: Config) usize {
        return self.nope + self.rope;
    }

    pub fn isMoe(self: Config, layer: usize) bool {
        return layer >= self.first_dense;
    }

    /// Whether a layer runs its own indexer (else it reuses the last full layer's selection).
    pub fn fullIndexer(self: Config, layer: usize) bool {
        return layer < max_layers and self.full_index[layer];
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        var root = parsed.value;
        if (root != .object) return error.BadConfig;
        if (root.object.get("text_config")) |t| {
            if (t == .object) root = t;
        }
        const o = root.object;
        const mt = o.get("model_type") orelse return error.BadConfig;
        if (mt != .string or !std.mem.eql(u8, mt.string, "glm_moe_dsa")) return error.NotGlmMoeDsa;
        if ((try optInt(o, "n_group", 1)) != 1 or (try optInt(o, "topk_group", 1)) != 1) return error.GroupedRoutingUnsupported;
        var c: Config = .{
            .hidden = try int(o, "hidden_size"),
            .layers = try int(o, "num_hidden_layers"),
            .first_dense = try int(o, "first_k_dense_replace"),
            .dense_width = try int(o, "intermediate_size"),
            .expert_width = try int(o, "moe_intermediate_size"),
            .experts = try int(o, "n_routed_experts"),
            .shared_experts = try int(o, "n_shared_experts"),
            .top_k = try int(o, "num_experts_per_tok"),
            .routed_scaling = @floatCast(try float(o, "routed_scaling_factor")),
            .norm_topk = try boolean(o, "norm_topk_prob"),
            .heads = try int(o, "num_attention_heads"),
            .q_lora = try int(o, "q_lora_rank"),
            .kv_lora = try int(o, "kv_lora_rank"),
            .nope = try int(o, "qk_nope_head_dim"),
            .rope = try int(o, "qk_rope_head_dim"),
            .v_dim = try int(o, "v_head_dim"),
            .rope_theta = 10000.0,
            .index_heads = try int(o, "index_n_heads"),
            .index_dim = try int(o, "index_head_dim"),
            .index_topk = try int(o, "index_topk"),
            .eps = @floatCast(try float(o, "rms_norm_eps")),
            .vocab = try int(o, "vocab_size"),
            .mtp_layers = try optInt(o, "num_nextn_predict_layers", 0),
        };
        // rope_parameters (or rope_scaling) .rope_theta, else rope_theta; only the default rope type
        const rope_obj = o.get("rope_parameters") orelse o.get("rope_scaling");
        var theta_set = false;
        if (rope_obj) |r| {
            if (r == .object) {
                if (r.object.get("rope_type")) |t| {
                    if (t != .string or !std.mem.eql(u8, t.string, "default")) return error.RopeScalingUnsupported;
                }
                if (r.object.get("rope_theta") != null) {
                    c.rope_theta = try float(r.object, "rope_theta");
                    theta_set = true;
                }
            }
        }
        if (!theta_set and o.get("rope_theta") != null) c.rope_theta = try float(o, "rope_theta");
        if (c.layers > max_layers) return error.TooManyLayers;
        const types = o.get("indexer_types") orelse return error.BadConfig;
        if (types != .array or types.array.items.len != c.layers) return error.BadConfig;
        for (types.array.items, 0..) |t, i| {
            if (t != .string) return error.BadConfig;
            if (std.mem.eql(u8, t.string, "full")) {
                c.full_index[i] = true;
            } else if (!std.mem.eql(u8, t.string, "shared")) return error.BadConfig;
        }
        if (!c.full_index[0]) return error.BadConfig;
        return c;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        return parse(gpa, text);
    }
};

fn int(o: std.json.ObjectMap, name: []const u8) !usize {
    const v = o.get(name) orelse {
        std.log.err("config.json has no {s}", .{name});
        return error.BadConfig;
    };
    return switch (v) {
        .integer => |i| std.math.cast(usize, i) orelse error.BadConfig,
        else => error.BadConfig,
    };
}

fn optInt(o: std.json.ObjectMap, name: []const u8, default: usize) !usize {
    if (o.get(name) == null) return default;
    return int(o, name);
}

fn float(o: std.json.ObjectMap, name: []const u8) !f64 {
    const v = o.get(name) orelse return error.BadConfig;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn boolean(o: std.json.ObjectMap, name: []const u8) !bool {
    const v = o.get(name) orelse return error.BadConfig;
    return switch (v) {
        .bool => |b| b,
        else => error.BadConfig,
    };
}

test "GLM-5.3's config.json fields" {
    const text =
        \\{"model_type": "glm_moe_dsa", "hidden_size": 6144, "num_hidden_layers": 4, "first_k_dense_replace": 3,
        \\ "intermediate_size": 12288, "moe_intermediate_size": 2048, "n_routed_experts": 256, "n_shared_experts": 1,
        \\ "num_experts_per_tok": 8, "routed_scaling_factor": 2.5, "norm_topk_prob": true, "num_attention_heads": 64,
        \\ "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
        \\ "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"}, "index_n_heads": 32, "index_head_dim": 128,
        \\ "index_topk": 2048, "indexer_types": ["full", "full", "full", "shared"], "rms_norm_eps": 1e-05,
        \\ "vocab_size": 154880, "n_group": 1, "topk_group": 1}
    ;
    const c = try Config.parse(std.testing.allocator, text);
    try std.testing.expectEqual(@as(usize, 576), c.latentWidth());
    try std.testing.expectEqual(@as(f64, 8000000), c.rope_theta);
    try std.testing.expect(c.fullIndexer(2) and !c.fullIndexer(3));
    try std.testing.expect(!c.isMoe(2) and c.isMoe(3));
    try std.testing.expectEqual(@as(f32, 1e-5), c.eps);
    try std.testing.expectError(error.NotGlmMoeDsa, Config.parse(std.testing.allocator, "{\"model_type\": \"x\"}"));
}
