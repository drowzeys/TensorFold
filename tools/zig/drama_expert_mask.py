#!/usr/bin/env python3
"""An --expert-mask file from routing mass: in each layer, keep the experts that took the most of it."""
import argparse
import json
import math
import sys
from pathlib import Path

EXPERTS, TOP_K, LAYERS = 256, 8, 40


def keep(mass, ratio, whole):
    """The mask: per layer of `mass`, the ids of its kept experts in ascending order."""
    count = max(TOP_K, math.ceil(ratio * EXPERTS))
    mask = {}
    for layer, values in mass.items():
        if not layer.isdigit() or int(layer) >= LAYERS or str(int(layer)) != layer:
            raise ValueError(f'layer "{layer}" is not a layer index from 0 to {LAYERS - 1}')
        if len(values) != EXPERTS or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in values):
            raise ValueError(f'layer {layer}: {EXPERTS} finite, non-negative masses')
        ranked = sorted(range(EXPERTS), key=lambda e: (-values[e], e))
        mask[layer] = list(range(EXPERTS)) if int(layer) in whole else sorted(ranked[:count])
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mass', type=Path, help='{"layer": [the routing mass each of the 256 experts took]}; layers it leaves out keep every expert')
    parser.add_argument('--ratio', type=float, required=True, help='each layer keeps ceil(R * 256) experts, at least the 8 a row picks, most mass first, the lower id on a tie')
    parser.add_argument('--whole', default='', help='layers that keep every expert, comma separated')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.ratio <= 1:
        parser.error('--ratio: a share above 0 to 1')
    try:
        whole = {int(layer) for layer in args.whole.split(',') if layer.strip()}
        mask = keep(json.loads(args.mass.read_text()), args.ratio, whole)
    except ValueError as e:
        sys.exit(f'drama_expert_mask.py: {args.mass}: {e}')
    args.out.write_text(json.dumps(mask, separators=(',', ':')) + '\n')
    kept = sum(len(ids) for ids in mask.values()) + EXPERTS * (LAYERS - len(mask))
    print(f'{args.out}: {kept} of {EXPERTS * LAYERS} experts kept')


if __name__ == '__main__':
    main()
