# Scoring GLM-5.3 as it is served

`tf-glm53-generate --mode score` measures how far the served engine's next-token distributions are from a reference
model's (the unquantized checkpoint): KL(reference || served) at temperature 1 and top-1 agreement, over fixed token
rows, per set and per row. It is the measurement a quantization is tuned against: the numbers include everything
between the pack on disk and the logits a client's tokens are drawn from. It ran on four DGX Sparks on 2026-10-12 (the table below); the
formats, the math and the tools' agreement with an independent numpy implementation are host-tested.

## What runs

Each row runs as a served prompt does (`Runner.prefillChunk`, the engine booted as the server boots it, with no
drafts), and the head runs on every row instead of the last one. `--score-rows 0` cuts the row into the chunks the
server would (the prompt GEMM, cuBLAS, prompt experts: what a client's prompt goes through); `--score-rows 16` runs
it 16 rows at a time through the decode windows' kernels (what generated tokens go through).

The head is split over the ranks and no rank sees another's logits, so each rank reduces every position to a few sums
over its own vocabulary share and writes them to a partials file. `--mode score-combine` (no GPU) adds the four files
up. Nothing is exchanged between the ranks while scoring.

## The reference and its bounds

`tools/glm53/score_reference.py` runs the reference model with exllamav3, one module at a time, and keeps for every
position its `--topk` most likely tokens (default 4096), the probability mass of the rest, the real next token's
log-probability and the entropy. Keeping every token would be 1.2 MB a position.

With the tail merged into one outcome the tool reports two numbers that bracket the exact KL:

- `KL`: the KL of the two distributions with the unkept tokens merged. Never above the exact value.
- `KL upper`: adds the worst the unkept tokens could contribute (none is likelier than the last kept token, none is
  less likely under the served model than its least likely token). Never below the exact value; loose by design.

Both equal the exact KL when every token is kept (host-tested), and the gap closes as more are kept.

## Files

| File | Made by | Holds |
|---|---|---|
| rows (`GLM53RF1`, no positions) | `tools/glm53/score_rows.py` | named sets of token rows |
| reference (`GLM53RF1`) | `tools/glm53/score_reference.py` | the rows and each position's reference record |
| partials (`GLM53PT1`) | `--mode score`, one a rank | each position's sums over that rank's vocabulary share |
| dump | `--mode score --score-dump` | the 16 largest local logits of the dumped rows' positions |

`score_rows.py` builds five sets from openly licensed sources: `wiki` (wikitext-2 test, the rows exllamav3's
`model_diff.py` scores), `code` (HumanEval), `math` (GSM8K), `chat` (UltraChat through the chat template) and `tools`
(Hermes function calling through the chat template, with tool schemas, calls and replies). A `.json` beside the file
records each source, its licence and the tokenizer's hash.

## Running it

```sh
python3 tools/glm53/score_rows.py --model MODEL --out rows.bin
python3 tools/glm53/score_reference.py --model BF16_MODEL --rows rows.bin --out reference.bin      # a GPU, hours
# on every rank, the engine's usual flags (see --mode 3b):
tf-glm53-generate --mode score --ref reference.bin --out part-rR.bin --rank R --world 4 --master IP:PORT \
    --model MODEL --aot AOT [--score-rows 0|16] [--score-dump dump-rR.bin --score-dump-rows 0,48]
tf-glm53-generate --mode score-combine --ref reference.bin --parts part-r0.bin,part-r1.bin,part-r2.bin,part-r3.bin --out score.json
python3 tools/glm53/score_dump.py --ref reference.bin --parts ... --dumps ... --model MODEL     # the worst positions
python3 tools/glm53/score_selftest.py zig-out/bin/tf-glm53-generate                              # no GPU
```

## Not done

- No GPU run: the scorer's two head launches on up to 128 rows, the engine booted with no MTP layer, and the speed
  are unverified.
- `score_reference.py` has not been run; it follows `model_diff.py`'s loop.
- Only KL(reference || served) is reported. The other direction needs the reference's whole distribution.

## Measured (2026-10-12, four DGX Sparks, GLM-5.3 EXL3 2.75 bpw against the unquantized checkpoint)

137,490 positions, the reference keeping 4,096 tokens a position, the served prompt path (`--score-rows 0`):

| Set | Tokens | KL | KL upper | Top-1 agreement |
|---|---|---|---|---|
| wiki | 65,504 | 0.1244 | 0.1376 | 89.5 % |
| code | 16,376 | 0.0496 | 0.0549 | 98.1 % |
| math | 16,376 | 0.1094 | 0.1158 | 93.8 % |
| chat | 22,501 | 0.1794 | 0.2252 | 88.0 % |
| tools | 16,733 | 0.5591 | 0.7318 | 80.0 % |
| all | 137,490 | 0.1756 | 0.2118 | 89.6 % |

The decode windows' kernels (`--score-rows 16`) score 0.1774 over all rows: the two paths agree to 0.002.
