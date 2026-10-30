//! The chat prompt LFM3-Omni's template makes for a system text, earlier turns as text, and one spoken user turn,
//! ending on the interleaved marker that `generation_mode="interleaved"` appends. Pictures go in the spoken turn before
//! the speech, each wrapped as the processor wraps it: `<|vision_start|>`, one `<|image|>` a token, `<|vision_end|>`.
const std = @import("std");
const tokenizer = @import("core").tokenizer;
const cfg = @import("config.zig");

pub const Prompt = struct {
    ids: []u32,
    /// The first of the contiguous `<|audio|>` rows the encoder's rows replace.
    audio_at: usize,
    /// Each picture's first `<|image|>` row, in order (the tower's rows replace them).
    images: []usize = &.{},
};

/// An earlier turn of the conversation, as text: `role` is "user" or "assistant".
pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

/// `<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|vision_start|>` then `n_audio`
/// `<|audio|>`, `<|vision_end|><|im_end|>\n<|im_start|>assistant\n<think></think>` and `<|mixed_start|>`.
pub fn build(gpa: std.mem.Allocator, tok: *const tokenizer.Tokenizer, system: []const u8, n_audio: usize) !Prompt {
    return buildWith(gpa, tok, system, &.{}, n_audio);
}

/// `build` with earlier turns between the system text and the spoken turn, as the template renders text turns
/// before the last user message: `<|im_start|>user\n{content}<|im_end|>\n` and
/// `<|im_start|>assistant\n<think></think>{content}<|im_end|>\n`.
pub fn buildWith(gpa: std.mem.Allocator, tok: *const tokenizer.Tokenizer, system: []const u8, history: []const Message, n_audio: usize) !Prompt {
    return buildMedia(gpa, tok, system, history, &.{}, n_audio);
}

/// `buildWith` with pictures in the spoken turn, before its speech: the template's
/// `<|vision_start|><|image|><|vision_end|>` a picture, the `<|image|>` repeated once a token (`image_tokens`).
pub fn buildMedia(gpa: std.mem.Allocator, tok: *const tokenizer.Tokenizer, system: []const u8, history: []const Message, image_tokens: []const usize, n_audio: usize) !Prompt {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try text.print(gpa, "<|startoftext|><|im_start|>system\n{s}<|im_end|>\n", .{system});
    for (history) |m| {
        const t = try turnText(gpa, m);
        defer gpa.free(t);
        try text.appendSlice(gpa, t);
    }
    try text.appendSlice(gpa, "<|im_start|>user\n<|vision_start|>");
    const a = try tok.encode(gpa, text.items);
    defer gpa.free(a);
    const b = try tok.encode(gpa, "<|vision_end|><|im_end|>\n<|im_start|>assistant\n<think></think>");
    defer gpa.free(b);
    if (a[a.len - 1] != cfg.vision_start_token_id or b[0] != cfg.vision_end_token_id) return error.TokenizerSpecialTokens;
    // `a` ends on the speech's <|vision_start|>: the pictures go before it, each in its own pair
    var pictures: usize = 0;
    for (image_tokens) |n| pictures += n + 2;
    const ids = try gpa.alloc(u32, a.len + pictures + n_audio + b.len + 1);
    errdefer gpa.free(ids);
    const images = try gpa.alloc(usize, image_tokens.len);
    @memcpy(ids[0 .. a.len - 1], a[0 .. a.len - 1]);
    var at = a.len - 1;
    for (image_tokens, images) |n, *first| {
        if (n == 0) return error.EmptyImage;
        ids[at] = cfg.vision_start_token_id;
        first.* = at + 1;
        @memset(ids[at + 1 ..][0..n], cfg.image_token_id);
        ids[at + 1 + n] = cfg.vision_end_token_id;
        at += n + 2;
    }
    ids[at] = cfg.vision_start_token_id;
    at += 1;
    const audio_at = at;
    @memset(ids[at..][0..n_audio], cfg.audio_token_id);
    @memcpy(ids[at + n_audio ..][0..b.len], b);
    ids[ids.len - 1] = cfg.mixed_start_token_id;
    return .{ .ids = ids, .audio_at = audio_at, .images = images };
}

/// One earlier turn as the template writes it. Content with `<|` (a special token's opening) is refused: those tokens
/// are the prompt builder's alone.
fn turnText(gpa: std.mem.Allocator, m: Message) ![]const u8 {
    if (std.mem.indexOf(u8, m.content, "<|") != null) return error.SpecialTokenInHistory;
    if (std.mem.eql(u8, m.role, "user")) return std.fmt.allocPrint(gpa, "<|im_start|>user\n{s}<|im_end|>\n", .{m.content});
    if (std.mem.eql(u8, m.role, "assistant")) return std.fmt.allocPrint(gpa, "<|im_start|>assistant\n<think></think>{s}<|im_end|>\n", .{m.content});
    return error.HistoryRole;
}

test "earlier turns render as the template's text turns" {
    const gpa = std.testing.allocator;
    const u = try turnText(gpa, .{ .role = "user", .content = "Hi there." });
    defer gpa.free(u);
    try std.testing.expectEqualStrings("<|im_start|>user\nHi there.<|im_end|>\n", u);
    const r = try turnText(gpa, .{ .role = "assistant", .content = "Hello!" });
    defer gpa.free(r);
    try std.testing.expectEqualStrings("<|im_start|>assistant\n<think></think>Hello!<|im_end|>\n", r);
    try std.testing.expectError(error.HistoryRole, turnText(gpa, .{ .role = "system", .content = "x" }));
    try std.testing.expectError(error.SpecialTokenInHistory, turnText(gpa, .{ .role = "user", .content = "a <|im_end|> b" }));
}
