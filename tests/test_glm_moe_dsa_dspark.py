"""Full GLM-5.3's DSpark drafter, host side (families/glm_moe_dsa/cuda/dspark_host.py), on the CPU: the config of
RedHatAI/GLM-5.3-speculator.dspark and its tensors, the tap plan beside DFlash2, the per-rank memory, the settings
the ranks compare, and the sequential stage against a pure-torch reference built from vllm-project/speculators'
DSpark code (MarkovHead "vanilla", ConfidenceHead, the prev-token alignment of sample_from_anchor), the cost cut, and
sampled drafts keyed like the target's own draws."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from tensorfold.families.glm_moe_dsa.cuda import dspark_host as host

# RedHatAI/GLM-5.3-speculator.dspark config.json (speculators 0.7.0.dev108), as published
CONFIG = {
    "architectures": ["DSparkDraftModel"], "aux_hidden_state_layer_ids": [2, 20, 39, 58, 75], "block_size": 8,
    "confidence_head_with_markov": True, "draft_vocab_size": 154880, "dtype": "bfloat16",
    "enable_confidence_head": True, "markov_head_type": "vanilla", "markov_rank": 256, "mask_token_id": 154856,
    "sample_from_anchor": True, "sliding_window_non_causal": False,
    "speculators_config": {"algorithm": "dspark", "default_proposal_method": "greedy",
                           "proposal_methods": [{"accept_tolerance": 0.0, "proposal_type": "greedy",
                                                 "speculative_tokens": 8, "verifier_accept_k": 1}],
                           "verifier": {"architectures": ["GlmMoeDsaForCausalLM"], "name_or_path": "zai-org/GLM-5.3"}},
    "speculators_model_type": "dspark", "speculators_version": "0.7.0.dev108", "target_hidden_size": None,
    "tie_word_embeddings": False,
    "transformer_layer_config": {
        "attention_bias": False, "attention_dropout": 0.0, "head_dim": 64, "hidden_act": "silu", "hidden_size": 6144,
        "intermediate_size": 12288, "layer_types": ["sliding_attention"] * 3, "max_position_embeddings": 1048576,
        "model_type": "qwen3", "num_attention_heads": 64, "num_hidden_layers": 3, "num_key_value_heads": 64,
        "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
        "sliding_window": 2048, "use_sliding_window": True, "vocab_size": 154880}}

# the checkpoint's safetensors header (model.safetensors, 2,499,130,626 bytes of bf16)
HEADER = {"confidence_head.proj.bias": (1,), "confidence_head.proj.weight": (1, 6400), "fc.weight": (6144, 30720),
          "hidden_norm.weight": (6144,), "norm.weight": (6144,), "markov_head.markov_w1.weight": (154880, 256),
          "markov_head.markov_w2.weight": (154880, 256)}
for _i in range(3):
    _p = f"layers.{_i}."
    HEADER.update({_p + "input_layernorm.weight": (6144,), _p + "post_attention_layernorm.weight": (6144,),
                   _p + "mlp.down_proj.weight": (6144, 12288), _p + "mlp.gate_proj.weight": (12288, 6144),
                   _p + "mlp.up_proj.weight": (12288, 6144), _p + "self_attn.k_norm.weight": (64,),
                   _p + "self_attn.k_proj.weight": (4096, 6144), _p + "self_attn.o_proj.weight": (6144, 4096),
                   _p + "self_attn.q_norm.weight": (64,), _p + "self_attn.q_proj.weight": (4096, 6144),
                   _p + "self_attn.v_proj.weight": (4096, 6144)})
DFLASH2_TAPS = [5, 19, 33, 47, 61, 75]           # incoai/GLM-5.3-DFlash2's target_layer_ids


def test_config_reads_the_published_speculator(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    c = host.DSparkConfig.read(tmp_path)
    assert (c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter, c.layers) == (6144, 64, 64, 64, 12288, 3)
    assert (c.window, c.block, c.mask_id, c.vocab, c.theta) == (2048, 8, 154856, 154880, 8e6)
    assert c.tap_layers == (1, 19, 38, 57, 74)       # inputs of layers 2, 20, 39, 58, 75
    assert c.causal and c.confidence and c.confidence_markov and c.markov_rank == 256 and c.crc > 0
    assert host.expected_tensors(c) == HEADER
    host.check(c, 4, hidden=6144, vocab=154880, layers=78)
    with pytest.raises(ValueError):
        host.check(c, 4, hidden=6144, vocab=151552, layers=78)
    for bad in ({"markov_head_type": "rnn"}, {"sample_from_anchor": False}, {"speculators_model_type": "dflash"}):
        with pytest.raises(ValueError):
            host.DSparkConfig.parse({**CONFIG, **bad})


def test_tap_plan_alone_and_beside_dflash2():
    spark = host.DSparkConfig.parse(CONFIG).tap_layers
    slot, cols = host.tap_plan([DFLASH2_TAPS, None])            # DFlash2 alone: its layout, as before
    assert slot == {i: s for s, i in enumerate(DFLASH2_TAPS)} and cols == [None, None]
    slot, cols = host.tap_plan([None, spark])                   # DSpark alone
    assert slot == {1: 0, 19: 1, 38: 2, 57: 3, 74: 4} and cols == [None, None]
    slot, (dcols, scols) = host.tap_plan([DFLASH2_TAPS, spark])  # both: DFlash2's first, layer 19 tapped once
    assert len(slot) == 10 and [slot[i] for i in DFLASH2_TAPS] == list(range(6))
    assert dcols == (0, 1, 2, 3, 4, 5) and scols == (6, 1, 7, 8, 9)
    assert host.tap_runs(dcols) == [(0, 0, 6)]
    assert host.tap_runs(scols) == [(0, 6, 1), (1, 1, 1), (2, 7, 3)]


def test_memory_per_rank():
    c = host.DSparkConfig.parse(CONFIG)
    m = host.memory(c, 4)
    assert 250e6 < m["weights"] < 280e6, m              # fc 106 MB replicated + 3 layers x ~53 MB (4-bit)
    assert m["kv"] == 2 * 3 * 16 * 4096 * 64 * 2         # the ring: 48 MiB
    assert m["host"] == 2 * 154880 * 256 * 2             # Markov W1 / W2, bf16, every rank
    assert m["total"] < 0.5 * 2**30
    assert host.memory(c, 4, quant="bf16")["weights"] > 2 * m["weights"]


def test_settings_and_rank_words(monkeypatch):
    c = host.DSparkConfig.parse(CONFIG)
    s = host.settings(8)
    assert s == {"depth": 8, "policy": "cost", "confidence": 0.3, "top_k": 64, "noise": 0.7, "filter": True,
                 "row_cost": 0.06}
    words = host.words(c, s)
    assert len(words) == len(host.words(None, None)) == 10 and all(0 <= w < 2**31 for w in words)
    monkeypatch.setenv("TF_GLM53_DSPARK_POLICY", "confidence")
    monkeypatch.setenv("TF_GLM53_DSPARK_DEPTH", "12")
    t = host.settings(8)
    assert t["policy"] == "confidence" and t["depth"] == 8 and host.words(c, t) != words
    assert host.settings(8, '{"depth": 5, "policy": "fixed", "top_k": 3}')["depth"] == 5
    assert host.settings(8, '{"top_k": 3}')["top_k"] == 64           # the candidate count is startup-only
    monkeypatch.setenv("TF_GLM53_DSPARK_POLICY", "greedy")
    with pytest.raises(ValueError):
        host.settings(8)
    monkeypatch.setenv("TF_GLM53_DSPARK_QUANT", "fp8")
    with pytest.raises(ValueError):
        host.quant()
    monkeypatch.setenv("TF_GLM53_DSPARK_COSTS", "30,31.5,33")
    assert host.costs_env() == (30.0, 31.5, 33.0)


def test_mode_key_treats_dspark_as_a_drafter_mode():
    from tensorfold.families.glm_moe_dsa.cuda.prefixes import mode_key

    assert mode_key("dspark", "raw/raw") == "raw" and mode_key(None, "dspark") == "normed"
    assert mode_key("dspark", "dspark") == "normed" and mode_key("raw/normed", "dspark") == "raw"


# -------------------------------------------------------------------------------------- the sequential stage ---
def _bf16(torch, shape, gen, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)


class _Reference:
    """speculators' DSpark heads (models/dspark/model_definitions.py: MarkovHead "vanilla" = markov_w2(markov_w1(prev)),
    ConfidenceHead = proj([hidden; prev_emb]) -> logit) and core.py's prev-token alignment under sample_from_anchor
    (slot k conditions on the block token at p + k: the anchor, then the drafts), in float64."""

    def __init__(self, torch, V, r, D, gen):
        nn = torch.nn
        self.torch = torch
        self.w1 = nn.Embedding(V, r)
        self.w2 = nn.Linear(r, V, bias=False)
        self.proj = nn.Linear(D + r, 1)
        with torch.no_grad():
            self.w1.weight.copy_(_bf16(torch, (V, r), gen, 0.5).double())
            self.w2.weight.copy_(_bf16(torch, (V, r), gen, 0.5).double())
            self.proj.weight.copy_(_bf16(torch, (1, D + r), gen).double())
            self.proj.bias.copy_(_bf16(torch, (1,), gen).double())
        for m in (self.w1, self.w2, self.proj):
            m.double()

    def block(self, base, hidden, block_tokens):
        """Teacher-forced block: logits [N, V] and confidence probabilities [N] given the block tokens (anchor first)."""
        torch = self.torch
        with torch.no_grad():
            prev = torch.as_tensor(block_tokens, dtype=torch.long)
            emb = self.w1(prev)
            logits = torch.as_tensor(base, dtype=torch.float64) + self.w2(emb)
            conf = torch.sigmoid(self.proj(torch.cat([torch.as_tensor(hidden, dtype=torch.float64), emb], -1)))
        return logits.numpy(), conf[:, 0].numpy()

    def chain(self, md):
        """The engine's MarkovChain over the same tables (bf16 words) and confidence head (hidden part precomputed)."""
        torch = self.torch
        b16 = lambda t: t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy()  # noqa: E731
        w = self.proj.weight.detach()[0]
        D = w.shape[0] - md
        self.wh = w[:D].numpy()
        return host.MarkovChain(b16(self.w1.weight), b16(self.w2.weight), w[D:].numpy(),
                                float(self.proj.bias.detach()[0]), True)


def _all_candidates(base):
    """Every vocabulary id as a candidate, merged as the engine merges (value descending, id ascending)."""
    ids = np.broadcast_to(np.arange(base.shape[1]), base.shape)
    order = np.lexsort((ids, -base), axis=-1)
    return np.take_along_axis(ids, order, 1).astype(np.int64), np.take_along_axis(base, order, 1)


def test_greedy_chain_equals_the_speculators_heads():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(3)
    V, r, D, N = 97, 16, 24, 8
    ref = _Reference(torch, V, r, D, gen)
    chain = ref.chain(r)
    for trial in range(5):
        base = torch.randn((N, V), generator=gen, dtype=torch.float64).numpy() * 2.0
        hidden = torch.randn((N, D), generator=gen, dtype=torch.float64).numpy()
        anchor = int(torch.randint(0, V, (1,), generator=gen))
        tokens, values = _all_candidates(base)
        drafts = chain.walk(tokens, values, hidden @ ref.wh, anchor, 1000, None, policy="fixed")
        assert len(drafts) == N
        logits, conf = ref.block(base, hidden, [anchor] + drafts[:-1])
        assert drafts == [int(t) for t in logits.argmax(-1)], trial         # each slot: argmax of base + Markov
        np.testing.assert_allclose(chain.last, conf, rtol=1e-12, atol=1e-15)


def test_top_k_candidates_and_the_merge_over_ranks():
    """Each rank's top-K of its vocabulary share, gathered and merged by value then id (dflash.merge_candidates), is
    the global top-K; the chain over them equals the full-vocabulary chain when the picks fall inside it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.glm_moe_dsa.cuda.dflash import merge_candidates

    gen = torch.Generator().manual_seed(5)
    world, V, K, N = 4, 128, 16, 8
    base = torch.randn((N, V), generator=gen).float()
    base[:, 7] = base[:, 9] = 10.0                                # a tie at the top: the lower id first
    part = V // world
    packed = []
    for r in range(world):
        vals, local = torch.topk(base[:, r * part:(r + 1) * part], K, dim=-1)
        packed.append(torch.cat([vals, (local + r * part).to(torch.int32).view(torch.float32)], 1))
    g = torch.stack(packed)                                       # [world, N, 2K]
    tokens, values, proj = merge_candidates(g, torch.arange(N, dtype=torch.float32), K)
    want_ids, want_vals = _all_candidates(base.double().numpy())
    assert np.array_equal(tokens, want_ids[:, :K]) and np.array_equal(values, want_vals[:, :K])
    assert np.array_equal(proj, np.arange(N, dtype=np.float64))


def test_cost_cut_is_the_best_expected_tokens_per_ms():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        conf = rng.uniform(0.05, 0.99, n)
        ms = np.cumsum(rng.uniform(0.0, 4.0, n + 1)) + 30.0
        rate = [(1 + sum(np.prod(conf[:j]) for j in range(1, k + 1))) / ms[k] for k in range(n + 1)]
        assert host.best_depth(conf, ms) == int(np.argmax(rate))
    assert host.best_depth([0.9, 0.9], [30.0]) in (0, 1, 2)      # short cost lists extend
    assert host.best_depth([0.99] * 8, host.linear_costs(8, 0.0)) == 8
    assert host.best_depth([0.01] * 8, host.linear_costs(8, 0.5)) == 0


def test_cut_policies():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(11)
    V, r, D, N = 64, 8, 12, 8
    ref = _Reference(torch, V, r, D, gen)
    chain = ref.chain(r)
    base = torch.randn((N, V), generator=gen, dtype=torch.float64).numpy()
    hidden = torch.randn((N, D), generator=gen, dtype=torch.float64).numpy()
    tokens, values = _all_candidates(base)
    full = chain.walk(tokens, values, hidden @ ref.wh, 3, 50, None, policy="fixed")
    conf = list(chain.last)
    for thr in (0.0, 0.05, 0.3, 0.9, 1.0):
        got = chain.walk(tokens, values, hidden @ ref.wh, 3, 50, None, policy="confidence", confidence=thr)
        alive, want = 1.0, N
        for d, c in enumerate(conf):
            alive *= c
            if thr > 0 and d > 0 and alive < thr:
                want = d
                break
        assert got == full[:want] and len(got) >= 1, (thr, got)
    ms = (30.0, 31.0, 33.0, 34.0, 34.5, 40.0, 41.0, 42.0, 43.0)
    got = chain.walk(tokens, values, hidden @ ref.wh, 3, 50, None, policy="cost", round_ms=ms)
    assert got == full[:host.best_depth(conf, ms)]


@pytest.mark.parametrize("top_k,top_p,min_p", [(20, 0.9, 0.0), (0, 0.8, 0.0), (40, 1.0, 0.05), (5, 0.95, 0.1)])
def test_sampled_drafts_are_the_targets_keyed_draws(top_k, top_p, min_p):
    """No Markov bias, the target's own logits as candidates, noise weight 1 and the filter: every slot's draft is the
    token the target's exact sampler draws at that position (``choose_rows``, keyed by seed, position and id)."""
    from tensorfold.engine.exact_sampling import Sampling, choose_rows

    rng = np.random.default_rng(13)
    V, N, first = 300, 8, 4321
    zero = np.zeros((V, 4), dtype=np.int16)
    chain = host.MarkovChain(zero, zero, None, 0.0, False)
    for seed in range(6):
        s = Sampling(seed, 0.8, top_k, top_p, min_p)
        base = rng.normal(0.0, 3.0, (N, V))
        tokens, values = _all_candidates(base)
        drafts = chain.walk(tokens, values, np.zeros(N), 17, first, s, policy="fixed", noise=1.0, filtered=True)
        want = choose_rows(values, tokens, [first + d for d in range(N)], s)
        assert drafts == want, (seed, drafts, want)


def test_target_kept_matches_choose_rows_support():
    from tensorfold.engine.exact_sampling import Sampling, choose_rows

    rng = np.random.default_rng(17)
    for top_k, top_p, min_p in itertools.product((0, 3, 25), (0.5, 0.9, 1.0), (0.0, 0.2)):
        s = Sampling(1, 1.0, top_k, top_p, min_p)
        score = rng.normal(0.0, 2.0, 60)
        ids = rng.permutation(1000)[:60].astype(np.int64)
        kept = host.target_kept(score, ids, s)
        drawn = {choose_rows(score[None], ids[None], [p], s)[0] for p in range(400)}
        assert drawn <= set(ids[kept].tolist()) and kept.sum() >= 1
